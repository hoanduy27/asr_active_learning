#!/bin/bash
set -e
set -u
set -o pipefail

MAIN_ROOT=$(pwd)/..

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

exit 

stage=1
stop_stage=10000

# feat options 
round=1
mode=rewind # Can be `rewind` or `finetune`

# Pretrained model config
asr_tag=asr_rnnt_conformer_round_1
asr_pretrained_model=valid.loss.ave_10best.pth
asr_model_config=config.yaml

# Dataset config
data_path=/

# Clustering config 
k=1000 # Number of cluster

seed=1

. local/parse_options.sh

egs_dir=$(pwd)
asr_model_path=${egs_dir}/exp/${asr_tag}/${asr_pretrained_model}
asr_model_config_path=${egs_dir}/conf/${asr_model_config}

feats_dir=${egs_dir}/feats/round_${round}
report_name=vivos_conformer

cluster_ckpt=${feats_dir}/kmedoids_target-pseudo_scale-none_k-${k}.pkl
cluster_feats_dir=${feats_dir}

report_path=${feats_dir}/${report_name}

# Path to save the training data 
data_output_dir=${egs_dir}/data

cd ${MAIN_ROOT}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Extract model features"
    _opts=" "
    _opts+="--asr_model_file ${asr_model_path}"
    _opts+="--asr_model_config ${asr_model_config_path} "
    _opts+="--data_path ${data_path} "
    _opts+="--wav_meta_path ${feats_dir}/wav.csv "
    _opts+="--report_dir ${feats_dir}"
    _opts+="--report_name ${report_name} "

    if [ ! -f ${feats_dir}/feats.done ]; then 
        # Load report checkpoint if any 
        if ls ${feats_dir}/*${report_name}*.csv 1> /dev/null 2>&1; then
            for file in ${feats_dir}/*${report_name}*.csv; do
                _opts+="--report_chkpt ${file}"
                break 
            done 
        fi
        
        python -m asr_evaluate.active_learning.statistics.conformer_lb_grad_decompose_faster ${_opts}
        touch ${feats_dir}/feats.done 
    else
        log Features exist. Skipping...
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: Clustering for sample selection"
    if [ ! -f ${feats_dir}/sampling.done ]; then
        _opts=" "
        _opts+="--dataset ${feats_dir}/${report_name}.csv; "
        _opts+="--feat ${cluster_feats_dir} "
        _opts+="--ckpt ${cluster_ckpt} "
        _opts+="--k ${k} "
        _opts+="--seed ${seed} "

        python -m asr_evaluate.active_learning.kmedoids_real ${_opts}

        touch ${feats_dir}/sampling.done
    else
        echo "Cluster model exist. Skipping..."
    fi
else
    echo "Stage 2 is skipped as per the stage settings."
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Stage 3: Convert to ESPNET dataset format"

    if [ "$mode" == "rewind" ]; then
        # TODO: Get all files matching feats/round_*/pseudo_cdist_dtw/sample.csv up to round 5
        files=$(find feats/round*/pseudo_cdist_dtw -name sample.csv | head -n 5 | tr '\n' ' ')
        python -m utils.merge_dataset "${data_output_dir}/train_nodev_${mode}_round_${round}" $files
    else 
        python -m asr_evaluate.active_learning.dataio.exporter -from aal -to kaldi -src "${feats_dir}/" -dst "${data_output_dir}/train_nodev_${mode}_round_${round}"
    fi
fi