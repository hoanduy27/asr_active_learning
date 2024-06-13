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

class ConformerPseudoGradDecompose(ConformerTransducerESP2Hypothesizer):
    def _calc_transducer_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Compute Transducer loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            encoder_out_lens: Encoder output sequences lengths. (B,)
            labels: Label ID sequences. (B, L)

        Return:
            loss_transducer: Transducer loss value.
            cer_transducer: Character error rate for Transducer.
            wer_transducer: Word Error Rate for Transducer.

        """
        criterion_transducer = RNNTLoss(
            blank=self.speech2text.asr_model.blank_id,
            fastemit_lambda=0.0,
            reduction="mean",
        )
        
        decoder_in, target, t_len, u_len = get_transducer_task_io(
            labels,
            encoder_out_lens,
            ignore_id=self.speech2text.asr_model.ignore_id,
            blank_id=self.speech2text.asr_model.blank_id,
        )

        self.speech2text.asr_model.decoder.set_device(encoder_out.device)
        decoder_out = self.speech2text.asr_model.decoder(decoder_in)

        joint_out = self.speech2text.asr_model.joint_network(
            encoder_out.unsqueeze(2), decoder_out.unsqueeze(1)
        )

        loss_transducer = criterion_transducer(
            joint_out,
            target,
            t_len,
            u_len,
        )

        # print(joint_out.shape)
        joint_out.retain_grad()

        loss_transducer.backward(retain_graph=True)

        return joint_out, loss_transducer
    


    def compute_loss_and_grad(self, speech, text):
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
        
        # Reset gradient
        self.speech2text.asr_model.zero_grad()

        # Forward encoder
        text[text == -1] = self.speech2text.asr_model.ignore_id
        text = text[:, : text_lengths.max()]
        encoder_out, encoder_out_lens = self.speech2text.asr_model.encode(speech, speech_lengths)
  
        # self.speech2text.asr_model.train()
        joint_out, loss = self._calc_transducer_loss(encoder_out, encoder_out_lens, text)

        # Compute grad norm
        joint_out_grad = joint_out.grad
        # self.speech2text.asr_model.eval()

        B,T,U,V = joint_out.size()
        H = self.speech2text.asr_model.joint_network.lin_out.weight.size(1)
        
        # (T, U)
        margin = joint_out[0].topk(dim=-1, k=2).values.diff().abs().squeeze(-1)
        
        # print(joint_weight_grad.shape)

        joint_weight_grad = torch.zeros((T, V, H))

        logging.info("Compute joint weight grad - faster")
        for t in range(T):
            # (V,H)
            grad_t = torch.autograd.grad(
                outputs=joint_out[0][t], 
                inputs=self.speech2text.asr_model.joint_network.lin_out.weight, 
                grad_outputs=joint_out_grad[0][t], 
                retain_graph=True
            )
            joint_weight_grad[t] = grad_t[0]

        # logging.info('Compute grad norm - faster')
        # (T,)
        grad_norm_nuclear = LA.matrix_norm(joint_weight_grad, ord='nuc')
        grad_norm_1 = LA.matrix_norm(joint_weight_grad, ord=1)
        grad_norm_2 = LA.matrix_norm(joint_weight_grad, ord=2)

        #(T,V)
        joint_weight_grad_by_vocab = LA.vector_norm(joint_weight_grad, ord=2, dim=-1)

       
        return loss.item(), margin.detach(), joint_weight_grad_by_vocab, grad_norm_nuclear, grad_norm_1, grad_norm_2

    def predict(self, utt: dict):
        wave_file = self.make_full_path(utt[PATH])
        text = utt[TEXT]
        speech, rate = sf.read(wave_file)

        results = self.speech2text(speech)

        # Get text
        top_text, _, _, _ = results[0]
        
        # Real
        real_loss, real_margin, real_grad, real_grad_norm_nuclear, real_grad_norm_1, real_grad_norm_2 = self.compute_loss_and_grad(speech, text)

        # Pseudo
        if text == top_text:
            pseudo_loss, pseudo_margin, pseudo_grad, pseudo_grad_norm_nuclear, pseudo_grad_norm_1, pseudo_grad_norm_2 \
                  = real_loss, real_margin, real_grad, real_grad_norm_nuclear, real_grad_norm_1, real_grad_norm_2
        else:    
            pseudo_loss, pseudo_margin, pseudo_grad, pseudo_grad_norm_nuclear, pseudo_grad_norm_1, pseudo_grad_norm_2  = self.compute_loss_and_grad(speech, top_text)

        # Save grad_norm info to pt file
        data_name = os.path.basename(self.data_path)
        pt_dir = '-'.join([data_name, self.model_code + '-faster'])
        os.makedirs(os.path.join(self.report_dir, pt_dir), exist_ok=True)

        margin_grad_info_path = os.path.join(self.report_dir, pt_dir, utt[UTTERANCE_ID] + '.npy')

        margin_grad_info = dict(
            real_margin = real_margin.cpu().numpy(),
            real_grad = real_grad.cpu().numpy(),
            real_grad_norm_nuclear = real_grad_norm_nuclear.cpu().numpy(), 
            real_grad_norm_1 = real_grad_norm_1.cpu().numpy(), 
            real_grad_norm_2 = real_grad_norm_2.cpu().numpy(),
            pseudo_margin = pseudo_margin.cpu().numpy(),
            pseudo_grad = pseudo_grad.cpu().numpy(),
            pseudo_grad_norm_nuclear = pseudo_grad_norm_nuclear.cpu().numpy(), 
            pseudo_grad_norm_1 = pseudo_grad_norm_1.cpu().numpy(), 
            pseudo_grad_norm_2 = pseudo_grad_norm_2.cpu().numpy()
        )
        T, real_U= real_margin.size()
        _, pseudo_U = pseudo_margin.size()

        np.save(margin_grad_info_path, margin_grad_info)

        # torch.save(margin_grad_info, margin_grad_info_path)

        return dict(
            hypothesis=top_text, 
            real_loss = real_loss, 
            pseudo_loss = pseudo_loss,
            margin_grad_info_path = margin_grad_info_path,
            T=T,
            real_U = real_U,
            pseudo_U = pseudo_U,
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
            'margin_grad_info_path',
            'T', 'real_U', 'pseudo_U'
        ]


class ConformerPseudoGradDecomposeFromFormat(ConformerPseudoGradDecompose, BaseHypothesizerFromFormat):
    def __init__(self, wav_meta_path, *args, **kwargs):
        self.wav_meta_path = wav_meta_path
        super(ConformerPseudoGradDecomposeFromFormat, self).__init__(*args, **kwargs)

class ConformerPseudoGradDecomposeFromDirectory(ConformerPseudoGradDecompose, BaseHypothersizerFromDirectory):
    pass 


def main(args):
    if args.wav_meta_path is not None:
        evaluator = ConformerPseudoGradDecomposeFromFormat(
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
        evaluator = ConformerPseudoGradDecomposeFromDirectory(
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
