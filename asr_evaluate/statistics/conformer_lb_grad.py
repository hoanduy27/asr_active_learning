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

class ConformerPseudoLoss(ConformerTransducerESP2Hypothesizer):
    def compute_loss_and_grad(self, speech, text):
        tokens = self.speech2text.tokenizer.text2tokens(text)
        text_ints = self.speech2text.converter.tokens2ids(tokens)

        batch = {
            "speech": torch.tensor(speech, dtype=torch.float32).unsqueeze(0),
            "speech_lengths": torch.tensor([speech.shape[0]], dtype=torch.long),
            "text": torch.tensor(text_ints, dtype=torch.long).unsqueeze(0),
            "text_lengths": torch.tensor([len(text_ints)], dtype=torch.long)
        }

        # Compute loss
        self.speech2text.asr_model.zero_grad()
        
        loss, *_ = self.speech2text.asr_model(**batch)

        loss.backward()

        loss = loss.item()
        
        # Compute grad norm
        for name,param in self.speech2text.asr_model.named_parameters():
            if "joint_network.lin_out.weight" in name:
                # grad_norm = torch.norm(param.grad, p=2).item()
                # grad_norm_frobenius = torch.norm(param.grad, p='fro').item()
                # grad_norm_frobenius = torch.norm(param.grad, p='nuc').item()
                grad_norm_nuclear = torch.linalg.matrix_norm(param.grad, ord='nuc').item()
                grad_norm_frobenius = torch.linalg.matrix_norm(param.grad, ord='fro').item()
                grad_norm_inf = torch.linalg.matrix_norm(param.grad, ord=torch.inf).item()
                grad_norm_1 = torch.linalg.matrix_norm(param.grad, ord=1).item()
                grad_norm_2 = torch.linalg.matrix_norm(param.grad, ord=2).item()
                grad_norm_inf_minus = torch.linalg.matrix_norm(param.grad, ord=-torch.inf).item()
                grad_norm_1_minus = torch.linalg.matrix_norm(param.grad, ord=-1).item()
                grad_norm_2_minus = torch.linalg.matrix_norm(param.grad, ord=-2).item()
                svd = torch.linalg.svd(param.grad).S.tolist()


                break

        return loss, grad_norm_nuclear, grad_norm_frobenius,\
            grad_norm_inf, grad_norm_1, grad_norm_2, \
            grad_norm_inf_minus, grad_norm_1_minus, grad_norm_2_minus, svd



    def predict(self, utt: dict):
        wave_file = self.make_full_path(utt[PATH])
        text = utt[TEXT]
        speech, rate = sf.read(wave_file)

        results = self.speech2text(speech)

        # Get text
        top_text, _, _, _ = results[0]
        
        # Real
        real_loss, real_grad_norm_nuclear, real_grad_norm_frobenius,\
            real_grad_norm_inf, real_grad_norm_1, real_grad_norm_2, \
            real_grad_norm_inf_minus, real_grad_norm_1_minus, real_grad_norm_2_minus, real_svd \
            = self.compute_loss_and_grad(speech, text)
      

        # Pseudo
        pseudo_loss, pseudo_grad_norm_nuclear, pseudo_grad_norm_frobenius, \
            pseudo_grad_norm_inf, pseudo_grad_norm_1, pseudo_grad_norm_2, \
            pseudo_grad_norm_inf_minus, pseudo_grad_norm_1_minus, pseudo_grad_norm_2_minus, pseudo_svd \
            = self.compute_loss_and_grad(speech, top_text)

        return dict(
            hypothesis=top_text, 
            real_loss = real_loss, 
            real_grad_norm_nuclear=real_grad_norm_nuclear,
            real_grad_norm_frobenius=real_grad_norm_frobenius,
            real_grad_norm_inf=real_grad_norm_inf,
            real_grad_norm_1=real_grad_norm_1,
            real_grad_norm_2=real_grad_norm_2,
            real_grad_norm_inf_minus=real_grad_norm_inf_minus,
            real_grad_norm_1_minus=real_grad_norm_1_minus,
            real_grad_norm_2_minus=real_grad_norm_2_minus,
            real_svd=real_svd,
            pseudo_loss = pseudo_loss,
            pseudo_grad_norm_nuclear = pseudo_grad_norm_nuclear,
            pseudo_grad_norm_frobenius = pseudo_grad_norm_frobenius,
            pseudo_grad_norm_inf = pseudo_grad_norm_inf,
            pseudo_grad_norm_1 = pseudo_grad_norm_1,
            pseudo_grad_norm_2 = pseudo_grad_norm_2,
            pseudo_grad_norm_inf_minus = pseudo_grad_norm_inf_minus,
            pseudo_grad_norm_1_minus = pseudo_grad_norm_1_minus,
            pseudo_grad_norm_2_minus = pseudo_grad_norm_2_minus,
            pseudo_svd = pseudo_svd
        )

    
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
        # inv_score_col = list(map(lambda x: "inv_score_" + str(x), range(self.speech2text.nbest)))
        return [
            HYPOTHESIS, 
            'real_loss', 
            'real_grad_norm_nuclear', 
            'real_grad_norm_frobenius', 
            'real_grad_norm_inf', 
            'real_grad_norm_1', 
            'real_grad_norm_2', 
            'real_grad_norm_inf_minus', 
            'real_grad_norm_1_minus', 
            'real_grad_norm_2_minus', 
            'real_svd',
            'pseudo_loss', 
            'pseudo_grad_norm_nuclear', 
            'pseudo_grad_norm_frobenius',
            'pseudo_grad_norm_inf',
            'pseudo_grad_norm_1',
            'pseudo_grad_norm_2',
            'pseudo_grad_norm_inf_minus',
            'pseudo_grad_norm_1_minus',
            'pseudo_grad_norm_2_minus',
            'pseudo_svd'
        ]


class ConformerPseudoLossFromFormat(ConformerPseudoLoss, BaseHypothesizerFromFormat):
    def __init__(self, wav_meta_path, *args, **kwargs):
        self.wav_meta_path = wav_meta_path
        super(ConformerPseudoLossFromFormat, self).__init__(*args, **kwargs)

class ConformerPseudoLossFromDirectory(ConformerPseudoLoss, BaseHypothersizerFromDirectory):
    pass 


def main(args):
    if args.wav_meta_path is not None:
        evaluator = ConformerPseudoLossFromFormat(
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
        evaluator = ConformerPseudoLossFromDirectory(
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
