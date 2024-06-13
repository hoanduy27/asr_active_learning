import argparse
import os
import time
from asr_evaluate.hypothesis.base_hypothesizer import BaseHypothersizerFromDirectory, BaseHypothesizer, BaseHypothesizerFromFormat
from google.cloud import speech
import io

class GoogleASRHypothesizer(BaseHypothesizer):
    def load_model(self):
        pass

    def predict(self, wav_file):
        client = speech.SpeechClient()

        # [START speech_python_migration_sync_request]
        # [START speech_python_migration_config]
        with io.open(wav_file, "rb") as audio_file:
            content = audio_file.read()
        start_s = time.time()
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
        )
        # [END speech_python_migration_config]

        # [START speech_python_migration_sync_response]
        response = client.recognize(config=config, audio=audio)
        inference_time = time.time() - start_s
        text = ""

        # [END speech_python_migration_sync_request]
        # Each result is for a consecutive portion of the audio. Iterate through
        # them to get the transcripts for the entire audio file.
        for result in response.results:
            # The first alternative is the most likely one for this portion.
            partial = result.alternatives[0].transcript
            text += " %s"%(partial)
        # [END speech_python_migration_sync_response]
        text = text.upper()
        text = ' '.join(text.split())
        return text, inference_time

class GoogleASRHypothesizerFromFormat(BaseHypothesizerFromFormat, GoogleASRHypothesizer):
    pass 

class GoogleASRHypothesizerFromDirectory(BaseHypothersizerFromDirectory, GoogleASRHypothesizer):
    pass 

def main(args):
    if args.wav_meta_path is not None:
        evaluator = GoogleASRHypothesizerFromFormat(
            args.wav_meta_path, 
            None, 
            args.model_code, 
            args.data_path, 
            args.report_dir,
            args.report_chkpt
        )
    else:
        evaluator = GoogleASRHypothesizerFromDirectory(
            args.model_path, 
            None, 
            args.data_path,
            args.report_dir,
            args.report_chkpt
        )

    evaluator()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_code',
        default='GoogleASR',
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
