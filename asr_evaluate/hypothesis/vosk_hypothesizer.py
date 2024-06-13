import argparse
from asr_evaluate.hypothesis.base_hypothesizer import *
from vosk import Model, KaldiRecognizer
import wave
import json
import time

class HybridHypothesizer(BaseHypothesizer):
    def load_model(self):
        if not os.path.exists(self.model_path):
            print(f"Model `{self.model_path}` does not exists")
            exit (1)
        self.model = Model(self.model_path)
        self.rec = KaldiRecognizer(
            self.model, 
            SAMPLE_RATE,
            '["oh one two three four five six seven eight nine zero", "[unk]"]'
        )
    def predict(self, wave_file):
        wf = wave.open(wave_file, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            print ("Audio file must be WAV format mono PCM.")
            exit (1)        
        
        start_time = time.time()
        #while True:
        #    data = wf.readframes(4000)
        #    if len(data) == 0:
        #        break
        #    _ =  self.rec.AcceptWaveform(data)

        with open(wave_file, "rb") as f:
            data = f.read()
            _ = self.rec.AcceptWaveform(data)
        # Stop recording time
        res = json.loads(self.rec.FinalResult().strip())
        
        print(self.rec.FinalResult())

        decode_time = time.time() - start_time
        return res.get('text', ''), decode_time

class HybridHypothesizerFromFormat(BaseHypothesizerFromFormat, HybridHypothesizer):
    pass 

class HybridHypothesizerFromDirectory(BaseHypothersizerFromDirectory, HybridHypothesizer):
    pass

def main(args):
    if args.wav_meta_path is not None:
        evaluator = HybridHypothesizerFromFormat(
            args.wav_meta_path, 
            args.model_path, 
            args.model_code, 
            args.data_path,
            args.report_dir,
            args.report_chkpt
        )
    else:
        evaluator = HybridHypothesizerFromDirectory(
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
        default="", 
        required=True, 
        help='model path'
    )
    parser.add_argument(
        '--model_code',
        default='Hybrid',
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
