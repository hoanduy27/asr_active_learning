from espnet2.bin.asr_inference import Speech2Text 
from espnet2.tasks.asr import ASRTask
import soundfile as sf
from torch import batch_norm
import torch
from tqdm import tqdm
import yaml
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.train.dataset import ESPnetDataset
from espnet2.samplers.unsorted_batch_sampler import UnsortedBatchSampler
from espnet2.iterators.sequence_iter_factory import SequenceIterFactory
from espnet2.train.collate_fn import CommonCollateFn
import pandas as pd
import os
import argparse

# config_file = '/home/duy/tmp/config.yaml'
# model_file = '/home/duy/tmp/15epoch.pth'
# wav_scp='/home/duy/github/espnet/egs2/oad_2204/asr1/data/test/wav.scp'
# text_file='/home/duy/github/espnet/egs2/oad_2204/asr1/data/test/text'
TOKEN_TYPE = 'char'

class ConformerTransducerLoss:
    def __init__(self, asr_model_file, asr_model_config, wav_scp, text_file):
        self.asr_model_file = asr_model_file 
        self.asr_model_config = asr_model_config
        self.wav_scp = wav_scp
        self.text_file = text_file
        self.load_model()

    def load_model(self):
        self.asr_model, self.asr_train_args = ASRTask.build_model_from_file(self.asr_model_config, self.asr_model_file)
        self.asr_model.to('cpu').eval()

    def load_data(self):
        with open(self.asr_model_config, 'r', encoding='utf-8') as f:
            conf = yaml.safe_load(f)
        token_list = conf['token_list']

        preprocessor = CommonPreprocessor(train=False, token_type=TOKEN_TYPE, token_list=token_list)
        dataset = ESPnetDataset(
            [# file, name (key), type
                (self.wav_scp, 'speech', 'sound'), 
                (self.text_file, 'text', 'text')
            ],
            preprocess=preprocessor,
        )
        batches = UnsortedBatchSampler(batch_size=1, key_file=self.wav_scp)
        batches = list(batches)
        # bs_list = [len(batch) for batch in batches]

        return SequenceIterFactory(
            dataset, batches, collate_fn=CommonCollateFn(0.0, 0)
        )


    def __call__(self, out_csv):
        # <utt_id>, <loss>
        dataloader = self.load_data().build_iter(0)
        if not os.path.exists(out_csv):
            df = pd.DataFrame(columns=['utt_id', 'loss'])
            df.to_csv(out_csv, index=False)
        else:
            df = pd.read_csv(out_csv)
            assert set(df.columns) == set(['utt_id', 'loss']), "Mismatch columns"

        for i,batch in tqdm(enumerate(dataloader)):
            # batch: Tuple(utt_id: str, model_input: Dict)
            utt_id = batch[0]
            try:
                if utt_id not in df.utt_id.values:
                    with torch.no_grad():
                        loss, *_ = self.asr_model(**batch[1])
                        loss = loss.item()

                    new_row = pd.DataFrame({'utt_id': utt_id, 'loss': [loss]}, )

                    new_row.to_csv(out_csv, mode='a', index=False, header=False)
            except:
                continue
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--asr_model_file', 
        type=str, 
        required=True, 
        help='Path to the trained model file'
    )
    parser.add_argument(
        '--asr_model_config',
        type=str,
        required=True,
        help='Path to the model config file'
    )
    parser.add_argument(
        '--wav_scp',
        type=str,
        required=True,
        help='Path to the wav.scp file'
    )
    parser.add_argument(
        '--text_file',
        type=str,
        required=True,
        help='Path to the text file'
    )
    parser.add_argument(
        '--out_csv',
        type=str,
        required=True,
        help='Path to the output csv file'
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    loss_evaluator = ConformerTransducerLoss(args.asr_model_file, args.asr_model_config, args.wav_scp, args.text_file)
    loss_evaluator(args.out_csv)
