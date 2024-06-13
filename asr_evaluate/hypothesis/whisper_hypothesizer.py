import argparse
import io
import logging
import os
import time
from functools import partial

import openai
import whisper
import yaml
from faster_whisper import WhisperModel

from asr_evaluate.hypothesis.base_hypothesizer import (
    BaseHypothersizerFromDirectory, BaseHypothesizer,
    BaseHypothesizerFromFormat)

OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY', '')
openai.api_key=OPENAI_API_KEY

class WhisperHypothesizer(BaseHypothesizer):
    def __init__(
            self, 
            model_code,
            model_path_or_tag, 
            backend,
            model_config,
            decode_config,
            data_path, 
            report_dir, 
            report_chkpt=None
        ):

        self.model_code = model_code 
        self.model_path_or_tag = model_path_or_tag
        self.backend = backend
        self.model_config = model_config
        self.decode_config = decode_config
        self.data_path = data_path
        self.report_dir = report_dir 
        self.report_chkpt = report_chkpt
        self.ds = None
        self.load_model()

    def load_model(self):
        if self.backend == "whisper":
            logging.info("Load model using pretrained path")
            model = whisper.load_model(self.model_path_or_tag, **self.model_config)
            self.transcriber = model.transcribe


        elif self.backend == "api":
            logging.info("Load model using API")
            if self.model_path_or_tag is None:
                self.model_path_or_tag = 'whisper-1'

            self.transcriber = partial(openai.Audio.transcribe, self.model_path_or_tag)
            
        elif self.backend == "whisper-faster":
            logging.info("Load model using pretrained path (faster)")
            model = WhisperModel(self.model_path_or_tag, **self.model_config)
            self.transcriber = model.transcribe

    def predict(self, wav_file):
        start_s = time.time()
        try:
            with open(wav_file, 'rb') as f:
                response = self.transcriber(f, **self.decode_config)
        except TypeError:
            response = self.transcriber(wav_file, **self.decode_config)
        
        if self.backend == "api" or self.backend == "whisper":
            text = response['text'].upper()
        else:
            segments, info = response
            text = ""
            for segment in segments:
                text += " " + segment.text
            text = " ".join(text.split()).upper()
        # logging.info(response)
        inference_time = time.time() - start_s

        logging.info(text)
       
        return text, inference_time

class WhisperHypothesizerFromFormat(WhisperHypothesizer, BaseHypothesizerFromFormat):
    def __init__(self, wav_meta_path, *args, **kwargs):
        self.wav_meta_path = wav_meta_path
        super(WhisperHypothesizerFromFormat, self).__init__(*args, **kwargs)

class WhisperHypothesizerFromDirectory(WhisperHypothesizer, BaseHypothersizerFromDirectory):
    pass 

def main(args):
    config = {}
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
    model_config = config.get('model_config', {})
    decode_config = config.get('decode_config', {})

    if args.wav_meta_path is not None:
        evaluator = WhisperHypothesizerFromFormat(
            args.wav_meta_path,
            args.model_code, 
            args.model_path_or_tag, 
            args.backend, 
            model_config,
            decode_config,
            args.data_path, 
            args.report_dir,
            args.report_chkpt
        )
    else:
        evaluator = WhisperHypothesizerFromDirectory(
            args.model_code, 
            args.model_path_or_tag, 
            args.backend, 
            model_config,
            decode_config,
            args.data_path, 
            args.report_dir,
            args.report_chkpt
        )

    evaluator()
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_code',
        default='whisper',
        required=True,
        help='Model code'
    )
    # group = parser.add_mutually_exclusive_group()
    parser.add_argument(
        '--model_path_or_tag',
        default=None,
        help='Path to model file'
    )
    parser.add_argument(
        '--backend',
        default="whisper",
        choices=["whisper", "whisper-faster", "api"],
        help="Which whisper backend to use"
    )
    parser.add_argument(
        '--config',
        default=None,
        help="Path to config file"
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
