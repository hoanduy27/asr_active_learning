import torch 
from torch import nn 


import torch
from espnet2.asr_transducer.utils import get_transducer_task_io
from warprnnt_pytorch import RNNTLoss

from asr_evaluate.CONFIG import *
from espnet2.bin import asr_inference

class EspnetASRTransducerFeatureExtraction(nn.Module):
    def __init__(self, asr_train_config, asr_model_file, **kwargs):
        super(EspnetASRTransducerFeatureExtraction, self).__init__()

        self.speech2text = asr_inference.Speech2Text(
            asr_train_config=asr_train_config,
            asr_model_file=asr_model_file,
            **kwargs 
        )

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
            reduction=None,
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

        return joint_out, loss_transducer

    def forward(self, speech, speech_lengths, text, text_lengths):
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        # Forward encoder
        text[text == -1] = self.speech2text.asr_model.ignore_id
        text = text[:, : text_lengths.max()]
        encoder_out, encoder_out_lens = self.speech2text.asr_model.encode(speech, speech_lengths)

        joint_out, loss = self._calc_transducer_loss(encoder_out, encoder_out_lens, text)

        return joint_out, loss


if __name__ == "__main__":
    asr_model_file="exp/asr_rnnt_conformer_round_1/valid.loss.ave_10best.pth"
    asr_model_config="exp/asr_rnnt_conformer_round_1/config.yaml"

    f_ext = EspnetASRTransducerFeatureExtraction(
        asr_train_config=asr_model_config,
        asr_model_file=asr_model_file,
        transducer_conf={},

    )
    T=40
    U=10
    B=32
    batch = dict(
        speech = torch.rand(B,T),
        speech_lengths = torch.randint(10, T, (B,)),
        text = torch.randint(1, 10, (B,U)),
        text_lengths = torch.randint(3, U, (B,))
    )
    print(batch)
    x = f_ext(**batch)
    print(x)

