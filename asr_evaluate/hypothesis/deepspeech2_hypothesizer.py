import argparse
import io
import time 
from asr_evaluate.hypothesis.base_hypothesizer import *
from urllib import response
import grpc
# import the generated classes
from asr_evaluate.hypothesis.deepspeech2_utils import asr_pb2
from asr_evaluate.hypothesis.deepspeech2_utils import asr_pb2_grpc

CHUNK_SIZE = 1024
class DeepSpeech2Hypothesizer(BaseHypothesizer):
    def load_model(self):
        self.channel = grpc.insecure_channel(self.model_path)
        self.stub = asr_pb2_grpc.speechRecognizationStub(self.channel)

    def predict(self, wav_file):
        with open(wav_file, 'rb') as file:
            content = file.read()
        
        content = io.BytesIO(content)
        n_bytes = content.getbuffer().nbytes
        n_chunks = (n_bytes // CHUNK_SIZE) + 1
        
        def get_chunk():
            for i in range(n_chunks):
                n_byte = min(CHUNK_SIZE, n_bytes - i * CHUNK_SIZE)
                yield content.read(n_byte)

        start_time = time.time()
        req = (asr_pb2.ASRRequest(utterance=chunk) for chunk in get_chunk())
        response = self.stub.speechRecognize(req)
        decode_time = time.time() - start_time
        response = response.transcription

        return response, decode_time

class DeepSpeech2HypothesizerFromFormat(BaseHypothesizerFromFormat, DeepSpeech2Hypothesizer):
    pass 

class DeepSpeech2HypothesizerFromDirectory(BaseHypothersizerFromDirectory, DeepSpeech2Hypothesizer):
    pass 

def main(args):
    if args.wav_meta_path is not None:
        evaluator = DeepSpeech2HypothesizerFromFormat(
            args.wav_meta_path, 
            args.endpoint, 
            args.model_code, 
            args.data_path,
            args.report_dir,
            args.report_chkpt
        )
    else:
        evaluator = DeepSpeech2HypothesizerFromDirectory(
            args.endpoint, 
            args.model_code, 
            args.data_path,
            args.report_dir,
            args.report_chkpt
        )

    evaluator()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--endpoint',
        default="192.168.1.210:50051", 
        required=True, 
        help='Endpoint to model server'
    )

    parser.add_argument(
        '--model_code',
        default='DeepSpeech2',
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
