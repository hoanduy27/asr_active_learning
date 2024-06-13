
import argparse
from asr_evaluate.hypothesis.base_hypothesizer import *
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import time
import soundfile as sf
import gzip

class CompressRatioCalculator(BaseHypothesizer):
    def load_model(self):
        pass

    def predict(self, utt: dict):
        wave_file = self.make_full_path(utt[PATH])
        with open(wave_file, 'rb') as f:
            data = f.read()

        compressed_data = gzip.compress(data)

        compress_ratio = 1 - len(compressed_data) / len(data)

        return dict(compress_ratio=compress_ratio)

    def get_hypothesis(self, *args, **kwargs):
        write_mode = 'a' if self.report_chkpt is not None else 'w'
        header = self.header + list(self.report_columns)
        with open(self.report_outfile, write_mode, encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header, delimiter=',', quotechar='"')
            if write_mode == 'w':
                writer.writeheader()
            for utt in tqdm.tqdm(self.ds):
                utt_id = utt[self.pkey]
                # Only process utterances that are not in checkpoint
                if utt_id in self.utt_ids:
                    ret = self.predict(utt, *args, **kwargs) 
                    utt.update(ret)
                    writer.writerow(utt)

    @property
    def report_columns(self):
        return ['compress_ratio']

class CompressRatioCalculatorFromFormat(BaseHypothesizerFromFormat, CompressRatioCalculator):
    pass 

class CompressRatioCalculatorFromDirectory(BaseHypothersizerFromDirectory, CompressRatioCalculator):
    pass

def main(args):
    if args.wav_meta_path is not None:
        evaluator = CompressRatioCalculatorFromFormat(
            args.wav_meta_path, 
            None,
            args.model_code, 
            args.data_path, 
            args.report_dir,
            args.report_chkpt
        )
    else:
        evaluator = CompressRatioCalculatorFromDirectory(
            None, 
            args.model_code, 
            args.data_path,
            args.report_dir,
            args.report_chkpt
        )

    evaluator()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_code',
        default='Wav2Vec2',
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
