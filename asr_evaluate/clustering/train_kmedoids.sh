dataset='/home/duy/github/asr_model_tesing/tmp/OAD2208/OAD2208-conformer_OAD350v2.0_OAD2204v0.9_lb_loss_grad_decompose-configID-Apr_04_2023.csv'
k=3000
n=10000

run_times=3
for i in $(seq 1 $run_times)
do
    seed=$i
    checkpoint="/home/duy/github/asr_model_tesing/tmp/OAD2208/kmeans/exp/kmedoids_target-pseudo_scale-none_n-${n}_k-${k}_set_${i}.pkl"
    feat_path="/home/duy/github/asr_model_tesing/tmp/OAD2208/kmeans/feats/pseudo_cdist_dtw_${n}_set_$i"
    echo "Run time: $i"
    echo "Checkpoint: $checkpoint"
    echo "Feat: $feat_path"

    python kmedoids_real.py --dataset $dataset --feat $feat_path --ckpt $checkpoint --n $n --k $k --seed $i
    echo "----"
done