import argparse
import csv
import time
import numpy as np
import soundfile as sf
import tqdm
import torch

from asr_evaluate.CONFIG import *
from asr_evaluate.hypothesis.conformerT_esp2_hypothesizer import \
    ConformerTransducerESP2Hypothesizer

from asr_evaluate.hypothesis.base_hypothesizer import \
    BaseHypothersizerFromDirectory, BaseHypothesizerFromFormat

from espnet2.torch_utils.device_funcs import to_device

# config_file = '/home/duy/tmp/config.yaml'
# model_file = '/home/duy/tmp/15epoch.pth'
# wav_scp='/home/duy/github/espnet/egs2/oad_2204/asr1/data/test/wav.scp'
# text_file='/home/duy/github/espnet/egs2/oad_2204/asr1/data/test/text'
TOKEN_TYPE = 'char'

class ConformerUncertainty(ConformerTransducerESP2Hypothesizer):
    def predict(self, utt: dict):
        wave_file = self.make_full_path(utt[PATH])
        text = utt[TEXT]
        tokens = self.speech2text.tokenizer.text2tokens(text)
        text_ints = self.speech2text.converter.tokens2ids(tokens)
        speech, rate = sf.read(wave_file)

        # Create batch
        batch = {
            "speech": torch.tensor(speech, dtype=torch.float32).unsqueeze(0),
            "speech_lengths": torch.tensor([speech.shape[0]], dtype=torch.long),
            "text": torch.tensor(text_ints, dtype=torch.long).unsqueeze(0),
            "text_lengths": torch.tensor([len(text_ints)], dtype=torch.long)
        }

        batch = to_device(batch, device=self.speech2text.device)

        # Compute loss
        loss, *_ = self.speech2text.asr_model(**batch)
        loss = loss.item()

        results = self.speech2text(speech)

        # Get text
        top_text, _, _, _ = results[0]

        # Get pseudo loss

        inv_hyp_scores = np.array([1 - np.exp(result[-1].score) for result in results])

        uncertainty_dict = {}
        uncertainty_dict[HYPOTHESIS] = top_text
        uncertainty_dict.update(dict(
            loss=loss
        ))

        uncertainty_dict.update(dict(map(
            lambda x: (
                "inv_score_" + str(x[0])
                , x[1]
            ), enumerate(inv_hyp_scores))))

        return uncertainty_dict
    
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
        inv_score_col = list(map(lambda x: "inv_score_" + str(x), range(self.speech2text.nbest)))

        return [HYPOTHESIS, 'loss'] + inv_score_col

class ConformerUncertaintyFromFormat(ConformerUncertainty, BaseHypothesizerFromFormat):
    def __init__(self, wav_meta_path, *args, **kwargs):
        self.wav_meta_path = wav_meta_path
        super(ConformerUncertaintyFromFormat, self).__init__(*args, **kwargs)

class ConformerUncertaintyFromDirectory(ConformerUncertainty, BaseHypothersizerFromDirectory):
    pass 


def main(args):
    if args.wav_meta_path is not None:
        evaluator = ConformerUncertaintyFromFormat(
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
        evaluator = ConformerUncertaintyFromDirectory(
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
