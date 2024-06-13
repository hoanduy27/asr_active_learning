
import argparse
from asr_evaluate.hypothesis.base_hypothesizer import *
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import time
import soundfile as sf

class Wav2Vec2Hypothesizer(BaseHypothesizer):
    def map_to_array(self, batch):
        speech, _ = sf.read(batch["file"])
        batch["speech"] = speech
        return batch

    def load_model(self):
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_path)

    def predict(self, wave_file):
        ds = self.map_to_array({
            "file": wave_file
        })
        start_time = time.time()
        # tokenize
        input_values = self.processor(ds["speech"], return_tensors="pt", padding="longest", sampling_rate=SAMPLE_RATE).input_values  # Batch size 1

        # retrieve logits
        logits = self.model(input_values).logits

        # take argmax and decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)
        decode_time = time.time() - start_time
        
        return transcription[0].upper(), decode_time


class Wav2Vec2HypothesizerFromFormat(BaseHypothesizerFromFormat, Wav2Vec2Hypothesizer):
    pass 

class Wav2Vec2HypothesizerFromDirectory(BaseHypothersizerFromDirectory, Wav2Vec2Hypothesizer):
    pass

def main(args):
    if args.wav_meta_path is not None:
        evaluator = Wav2Vec2HypothesizerFromFormat(
            args.wav_meta_path, 
            args.model_path, 
            args.model_code, 
            args.data_path, 
            args.report_dir,
            args.report_chkpt
        )
    else:
        evaluator = Wav2Vec2HypothesizerFromDirectory(
            args.model_path, 
            args.model_code, 
            args.data_path,
            args.report_dir,
            args.report_chkpt
        )

    evaluator()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        default="nguyenvulebinh/wav2vec2-base-vietnamese-250h", 
        required=True, 
        help='model path'
    )
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
