import csv
from asr_evaluateluate import utils
import os

from joblib import Parallel, delayed
from torch import quantize_per_tensor
from asr_evaluateluate.dataio.dataset import WavDataset
from asr_evaluateluate.CONFIG import *
import tqdm
from datetime import datetime
from asr_evaluateluate import metrics

class BaseHypothesizer:
    def __init__(self, model_path, model_code, data_path, report_dir, report_chkpt=None):
        self.model_path = model_path
        self.model_code = model_code 
        self.data_path = data_path
        self.ds = None
        self.report_dir = report_dir 
        self.report_chkpt = report_chkpt
        self.load_model()

    def load_model(self):
        raise NotImplementedError

    def predict(self, wave_file, *args, **kwargs):
        raise NotImplementedError

    def load_dataset(self, *args, **kwargs):
        raise NotImplementedError

    def make_full_path(self, relpath):
        return os.path.join(self.data_path, relpath)

    def get_hypothesis(self, *args, **kwargs):
        write_mode = 'a' if self.report_chkpt is not None else 'w'
        header = self.header + [HYPOTHESIS, INFERENCE_TIME]
        with open(self.report_outfile, write_mode, encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header, delimiter=',', quotechar='"')
            if write_mode == 'w':
                writer.writeheader()
            for utt in tqdm.tqdm(self.ds):
                utt_id = utt[self.pkey]
                # Only process utterances that are not in checkpoint
                if utt_id in self.utt_ids:
                    full_path = self.make_full_path(utt[PATH])
                    hyp, inference_time = self.predict(full_path, *args, **kwargs) 
                    utt[HYPOTHESIS] = hyp
                    utt[INFERENCE_TIME] = inference_time
                    writer.writerow(utt)

    def prepare_report(self, *args, **kwargs):
        if len(self.ds) == 0:
            exit('No line to write. Exiting...')

        os.makedirs(self.report_dir, exist_ok=True)
        
        # Prepare report filename
        dataset = os.path.basename(self.data_path)
        model_code = self.model_code
        configID = 'configID'
        date = datetime.today().strftime(TIME_FORMAT)
        self.report_outfile = '%s-%s-%s-%s.csv'%(dataset, model_code, configID, date)
        self.report_outfile = os.path.join(self.report_dir, self.report_outfile)

        self.utt_ids = list(map(lambda data: data[self.pkey], self.ds))

        if self.report_chkpt is not None:
            # Rename report checkpoint if new report has the same path
            # if os.path.samefile(self.report_chkpt, self.report_outfile):
            if os.path.realpath(self.report_chkpt) == os.path.realpath(self.report_outfile):
                new_report_chkpt = utils.add_postfix(self.report_chkpt, '-old', return_path = True)
                os.rename(self.report_chkpt, new_report_chkpt)
                self.report_chkpt = new_report_chkpt
            
            with open(self.report_chkpt, 'r', encoding='utf-8') as fin, \
                open(self.report_outfile, 'w', encoding='utf-8') as fout:
                reader = csv.DictReader(fin, delimiter=',', quotechar='"')
                chkpt_header = reader.fieldnames
                # Validate report checkpoint
                assert set(self.header) == set(chkpt_header) - set(self.report_columns), \
                     "Mismatch columns with report checkpoint"

                # Restore from checkpoint 
                writer = csv.DictWriter(fout, fieldnames=chkpt_header, delimiter=',', quotechar='"')
                writer.writeheader()
                for row in reader:
                    utt_id = row[self.pkey]
                    if utt_id in self.utt_ids:
                        writer.writerow(row)
                        self.utt_ids.remove(utt_id)

    def __call__(self, *args, **kwargs):
        self.load_dataset(*args, **kwargs)
        self.prepare_report(*args, **kwargs)
        self.get_hypothesis(*args, **kwargs)

        # header = self.ds[0].keys()
        # os.makedirs(report_dir, exist_ok=True)
        
        # dataset = os.path.basename(self.data_path)
        # model_code = self.model_code
        # configID = 'configID'
        # date = datetime.today().strftime(TIME_FORMAT)
        # report_outfile = '%s-%s-%s-%s.csv'%(dataset, model_code, configID, date)
        # report_outfile = os.path.join(report_dir, report_outfile)

        # with open(report_outfile, 'w', encoding='utf-8') as f:
        #     writer = csv.DictWriter(f, fieldnames=header)
        #     writer.writeheader()
        #     writer.writerows(self.ds)

    @property
    def pkey(self):
        raise NotImplementedError
    
    @property
    def report_columns(self):
        return [HYPOTHESIS, INFERENCE_TIME]

    @property
    def header(self):
        if len(self.ds) == 0:
            return []
        else:
            return [col for col in self.ds[0].keys() if col not in self.report_columns]

class BaseHypothesizerFromFormat(BaseHypothesizer):
    def __init__(self, wav_meta_path, *args, **kwargs):
        self.wav_meta_path = wav_meta_path
        super(BaseHypothesizerFromFormat, self).__init__(*args, **kwargs)
    def load_dataset(self):
        self.ds = WavDataset.from_aal_format(self.wav_meta_path)
    @property
    def pkey(self):
        return UTTERANCE_ID

class BaseHypothersizerFromDirectory(BaseHypothesizer):
    def load_dataset(self):
        self.ds = WavDataset.from_dir(self.data_path)
    @property
    def pkey(self):
        return PATH
