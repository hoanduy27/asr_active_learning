import argparse
import csv
import os
import numpy as np
import pandas as pd

from asr_evaluate.CONFIG import *
from asr_evaluate import utils
from asr_evaluate import metrics
from asr_evaluate.dataio.dataset import WavDataset

class ErrorRate:
    def __init__(self, hypothesis_report_path, wav_meta_path=None):
        self.hypo_report_path = hypothesis_report_path
        self.wav_meta_path = wav_meta_path

    def compute_errors(self):
        self.ret = []
        hyps = {}
        refs = {}
        CERs, WERs = [], []

        with open(self.hypo_report_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=',', quotechar='"')
            if UTTERANCE_ID in reader.fieldnames:
                self.pkey = UTTERANCE_ID
            elif PATH in reader.fieldnames:
                self.pkey = PATH
            else:
                raise RuntimeError('Cannot find %s or %s in %s'%(
                    UTTERANCE_ID, PATH, self.hypo_report_path))
            for row in reader:
                hyp = row[HYPOTHESIS]
                utt_id = row[self.pkey]
                hyps[utt_id] = hyp
                if self.wav_meta_path is None:
                    try:
                        ref = row[TEXT]
                        refs[utt_id] = ref
                    except:
                        raise RuntimeError('Missing `%s` column in %s, please specify `wav_meta_path`'%(
                            TEXT, self.hypo_report_path))
                    cer, wer = metrics.compute_error(ref, hyp)
                    row.update({'CER': cer, 'WER': wer})
                    self.ret.append(row)
                    CERs.append(cer)
                    WERs.append(wer)

        if self.wav_meta_path is not None:
            ds = WavDataset.from_aal_format(self.wav_meta_path)
            for data in ds:
                ref = data[TEXT]
                utt_id = data[self.pkey]
                refs[utt_id] = ref
                hyp = hyps[data[self.pkey]]
                cer, wer = metrics.compute_error(ref, hyp)
                data.update({HYPOTHESIS: hyp, 'CER': ERR_FORMAT%cer, 'WER': ERR_FORMAT%wer})
                self.ret.append(data)
                CERs.append(cer)
                WERs.append(wer)

        self.mean_cer = np.mean(np.array([cer for cer in CERs if pd.notna(cer)]))
        self.mean_wer = np.mean(np.array([wer for wer in WERs if pd.notna(wer)]))

        refs = [ref for id,ref in refs.items()]
        hyps = [hyp for id,hyp in hyps.items()]

        g_cer,g_wer,g_ser=metrics.compute_errors(refs, hyps)
        return g_cer, g_wer, g_ser
    
    
    def __call__(self, report_dir):
        g_cer,g_wer,g_ser = self.compute_errors()
        if self.ret == 0:
            exit('No line to write. Exiting...')
        else:
            header = self.ret[0].keys()
        
        os.makedirs(report_dir, exist_ok=True)

        # Calculate error rates (CER, WER) for each utterance

        report_filepath = utils.add_postfix(self.hypo_report_path, postfix='-err', return_path=False)
        report_filepath = os.path.join(report_dir, report_filepath)

        with open(report_filepath, 'w', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header, delimiter=',', quotechar='"')
            writer.writeheader()
            writer.writerows(self.ret)

        # Calculate global error rates
        report_filepath = utils.add_postfix(self.hypo_report_path, postfix='-glb_err', return_path=False)
        report_filepath = os.path.join(report_dir, report_filepath)

        with open(report_filepath, 'w', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['CER', 'WER', 'SER'], delimiter=',', quotechar='"')
            writer.writeheader()
            writer.writerow({
                'CER': ERR_FORMAT%g_cer, 
                'WER': ERR_FORMAT%g_wer, 
                'SER': ERR_FORMAT%g_ser})

        # write mean cer, mean wer of all samples
        mean_report_filepath = utils.add_postfix(self.hypo_report_path, postfix='-mean_err', return_path=False)
        mean_report_filepath = os.path.join(report_dir, mean_report_filepath)

        with open(mean_report_filepath, 'w', encoding='utf-8') as f:
            f.write(f"Average CERs: {self.mean_cer}%\nAverage WERs: {self.mean_wer}%")

def main(args):
    evaluator = ErrorRate(args.hyp_path, args.wav_meta_path)
    evaluator(args.report_dir)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hyp_path',
        required=True, 
        help='Path to hypothesis report file'
    )
    parser.add_argument(
        '--wav_meta_path',
        default=None,
        help='Path to wav metadata file'
    )
    parser.add_argument(
        '--report_dir',
        required=True,
        help='Path to store report'
    )
    args = parser.parse_args()
    main(args)