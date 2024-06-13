import argparse
import csv
import os
import logging
import time

import numpy as np
import soundfile as sf
import torch
import tqdm
from espnet2.asr_transducer.utils import get_transducer_task_io
from espnet2.torch_utils.device_funcs import to_device
from matplotlib import pyplot as plt
from torch import linalg as LA
from warprnnt_pytorch import RNNTLoss

from asr_evaluate.CONFIG import *
from asr_evaluate.hypothesis.base_hypothesizer import (
    BaseHypothersizerFromDirectory, BaseHypothesizerFromFormat)
from asr_evaluate.hypothesis.conformerT_esp2_hypothesizer import \
    ConformerTransducerESP2Hypothesizer

# config_file = '/home/duy/tmp/config.yaml'
# model_file = '/home/duy/tmp/15epoch.pth'
# wav_scp='/home/duy/github/espnet/egs2/oad_2204/asr1/data/test/wav.scp'
# text_file='/home/duy/github/espnet/egs2/oad_2204/asr1/data/test/text'
TOKEN_TYPE = 'char'

# logging.basicConfig(level=logging.INFO)

class ConformerPseudoLoss(ConformerTransducerESP2Hypothesizer):
    # def _calc_transducer_loss(
    #     self,
    #     encoder_out: torch.Tensor,
    #     encoder_out_lens: torch.Tensor,
    #     labels: torch.Tensor,
    # ):
    #     """Compute Transducer loss.

    #     Args:
    #         encoder_out: Encoder output sequences. (B, T, D_enc)
    #         encoder_out_lens: Encoder output sequences lengths. (B,)
    #         labels: Label ID sequences. (B, L)

    #     Return:
    #         loss_transducer: Transducer loss value.
    #         cer_transducer: Character error rate for Transducer.
    #         wer_transducer: Word Error Rate for Transducer.

    #     """
    #     criterion_transducer = RNNTLoss(
    #         blank=self.speech2text.asr_model.blank_id,
    #         fastemit_lambda=0.0,
    #         reduction="mean",
    #     )
        
    #     decoder_in, target, t_len, u_len = get_transducer_task_io(
    #         labels,
    #         encoder_out_lens,
    #         ignore_id=self.speech2text.asr_model.ignore_id,
    #         blank_id=self.speech2text.asr_model.blank_id,
    #     )

    #     self.speech2text.asr_model.decoder.set_device(encoder_out.device)
    #     decoder_out = self.speech2text.asr_model.decoder(decoder_in)

    #     joint_out = self.speech2text.asr_model.joint_network(
    #         encoder_out.unsqueeze(2), decoder_out.unsqueeze(1)
    #     )

    #     loss_transducer = criterion_transducer(
    #         joint_out,
    #         target,
    #         t_len,
    #         u_len,
    #     )
    #     return loss_transducer
    


    def compute_loss(self, speech, text):
        """
        Returns:
            margin (T,U): top_pred - 2nd_pred
            joint_weight_grad (T,V,H): dJ/dJoint_w at each timestep
        """
        tokens = self.speech2text.tokenizer.text2tokens(text)
        text_ints = self.speech2text.converter.tokens2ids(tokens)

        
        speech = torch.tensor(speech, dtype=torch.float32, device=self.speech2text.device).unsqueeze(0)
        speech_lengths = torch.tensor([speech.shape[-1]], dtype=torch.long, device=self.speech2text.device)
        text = torch.tensor(text_ints, dtype=torch.long, device=self.speech2text.device).unsqueeze(0)
        text_lengths = torch.tensor([len(text_ints)], dtype=torch.long, device=self.speech2text.device)


        batch = dict(
            speech = speech,
            speech_lengths = speech_lengths,
            text = text,
            text_lengths = text_lengths
        )
        
        loss, *_ = self.speech2text.asr_model(**batch)

        return loss.item()

    def predict(self, utt: dict):
        wave_file = self.make_full_path(utt[PATH])
        text = utt[TEXT]
        speech, rate = sf.read(wave_file)

        results = self.speech2text(speech)

        # Get text
        top_text, _, _, _ = results[0]
        
        # Real
        real_loss  = self.compute_loss(speech, text)

        # Pseudo
        if text == top_text:
            pseudo_loss = real_loss
        else:    
            pseudo_loss = self.compute_loss(speech, top_text)

        return dict(
            hypothesis=top_text, 
            real_loss = real_loss, 
            pseudo_loss = pseudo_loss,
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
            'pseudo_loss',
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
