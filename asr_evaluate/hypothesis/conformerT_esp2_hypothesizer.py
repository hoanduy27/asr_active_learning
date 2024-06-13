import argparse
import json
import time
from torch.utils.data import DataLoader
from espnet2.bin import asr_train, asr_inference
from asr_evaluate.CONFIG import *
import tqdm
import soundfile as sf
import csv

from asr_evaluate.hypothesis.base_hypothesizer import BaseHypothersizerFromDirectory, BaseHypothesizer, BaseHypothesizerFromFormat

class ConformerTransducerESP2Hypothesizer(BaseHypothesizer):
    def __init__(self, asr_model_file, asr_model_config, model_code, data_path, report_dir, report_chkpt=None, lm_file=None, lm_config=None):
        self.model_code = model_code 
        self.data_path = data_path
        self.ds = None
        self.asr_model_file = asr_model_file 
        self.asr_model_config = asr_model_config
        self.lm_file = lm_file
        self.lm_config = lm_config
        self.report_dir = report_dir 
        self.report_chkpt = report_chkpt
        self.load_model()

    def load_model(self):
        lm_inference_config = {}
        if self.lm_file is None or self.lm_config is None:
            self.lm_file = None 
            self.lm_config = None 
        else:
            # with open(lm_inference_config, 'r', encoding='utf-8') as f:
            #     lm_conf = json.load(f)
            # Future: load from config
            lm_inference_config = {
                'lm_weight': 0.5,
                'beam_size': 10,
                'penalty': 1,
                'ctc_weight': 0.5
            }  
        print(lm_inference_config)
        self.speech2text = asr_inference.Speech2Text(
            self.asr_model_config, self.asr_model_file, 
            beam_size=2,
            ctc_weight=0.0,
            transducer_conf=dict(search_type='default', nbest=2),
            lm_train_config = self.lm_config,
            lm_file = self.lm_file,
            nbest=2,
            **lm_inference_config,
        )
        self.speech2text.asr_model.error_calculator = None 
        self.speech2text.asr_model.error_calculator_trans = None

    def predict(self, wave_file):
        speech, rate = sf.read(wave_file)
        start_s = time.time()
        nbests = self.speech2text(speech)
        text, *_ = nbests[0]
        inference_time = time.time() - start_s 
        return text, inference_time
    
    def get_hypothesis(self, *args, **kwargs):
        write_mode = 'a' if self.report_chkpt is not None else 'w'
        header = self.header + [HYPOTHESIS, INFERENCE_TIME]
        with open(self.report_outfile, write_mode, encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header, delimiter=',', quotechar='"')
            if write_mode == 'w':
                writer.writeheader()
            for utt in tqdm.tqdm(self.ds):
                utt_id = utt[self.pkey]
                # Only process utterances that are not in checkpoint
                if utt_id in self.utt_ids:
                    full_path = self.make_full_path(utt[PATH])
                    hyp, inference_time = self.predict(full_path, *args, **kwargs) 
                    utt[HYPOTHESIS] = hyp
                    utt[INFERENCE_TIME] = inference_time
                    writer.writerow(utt)

class ConformerTransducerESP2HypothesizerFromFormat(ConformerTransducerESP2Hypothesizer, BaseHypothesizerFromFormat):
    def __init__(self, wav_meta_path, *args, **kwargs):
        self.wav_meta_path = wav_meta_path
        super(ConformerTransducerESP2HypothesizerFromFormat, self).__init__(*args, **kwargs)

class ConformerTransducerESP2HypothesizerFromDirectory(ConformerTransducerESP2Hypothesizer, BaseHypothersizerFromDirectory):
    pass 


def main(args):
    if args.wav_meta_path is not None:
        evaluator = ConformerTransducerESP2HypothesizerFromFormat(
            args.wav_meta_path, 
            args.asr_model_file,
            args.asr_model_config,
            args.model_code,
            args.data_path,
            args.report_dir,
            args.report_chkpt,
            args.lm_file,
            args.lm_config,
        )
    else:
        evaluator = ConformerTransducerESP2HypothesizerFromDirectory(
            args.asr_model_file,
            args.asr_model_config,
            args.model_code,
            args.data_path,
            args.report_dir,
            args.report_chkpt,
            args.lm_file,
            args.lm_config,
        )

    evaluator()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--asr_model_file',
        required=True, 
        help='Model path'
    )
    parser.add_argument(
        '--asr_model_config',
        required=True, 
        help='Model config path'
    )
    parser.add_argument(
        '--lm_file',
        default=None, 
        help='Language model path'
    )
    parser.add_argument(
        '--lm_config',
        default=None,
        help='Language model config path'
    )
    parser.add_argument(
        '--model_code',
        default='ConformerT',
        required=True,
        help='Model code'
    )
    parser.add_argument(
        '--data_path',
        required='True',
        help='Data path'
    )
    parser.add_argument(
        '--wav_meta_path',
        default=None,
        help='Path to wav metadata'
    )
    parser.add_argument(
        '--report_dir',
        required=True,
        help='Path to store report'
    )
    parser.add_argument(
        '--report_chkpt',
        default=None,
        help='Path to restore report'
    )
    args = parser.parse_args()
    main(args)
