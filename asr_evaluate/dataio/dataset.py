import csv
from re import S
from torch.utils.data import Dataset
from asr_evaluate.CONFIG import *
from asr_evaluate.utils import *
import pandas as pd

class WavDataset(Dataset):
    def __init__(self, container, *args, **kwargs):
        super(WavDataset, self).__init__(*args, **kwargs)
        self.container = container

    def __len__(self):
        return len(self.container)

    def __getitem__(self, idx):
        return self.container[idx]

    def __add__(self, other):
        return self.__class__(self.container + other.container)
    
    def __sub__(self, other):
        df = pd.DataFrame(self.container)
        other_df = pd.DataFrame(other.container)
        try:
            result_df = df[~df[UTTERANCE_ID].isin(other_df[UTTERANCE_ID])]
        except:
            result_df = df[~df[PATH].isin(other_df[PATH])]
        
        container = result_df.to_dict(orient='records')

        return self.__class__(container)
    

    @classmethod
    def from_aal_format_old(cls, wav_meta_path, speaker_meta_path=None, utterance_meta_path=None):
        speakers = {}
        utterances = {}
        container = []
        if speaker_meta_path is not None:
            with open(speaker_meta_path, 'r', encoding='utf-8') as f:
                rows = csv.DictReader(f, delimiter=',', quotechar='"')
                for row in rows:
                    spkid = row[AAL_CONST[SPEAKER_ID]]
                    speakers[spkid] = row
        if utterance_meta_path is not None:
            with open(utterance_meta_path, 'r',  encoding='utf-8') as f:
                rows = csv.DictReader(f, delimiter=',', quotechar='"')
                for row in rows:
                    utt_id = row[AAL_CONST[UTTERANCE_ID]]
                    utterances[utt_id] = row

        with open(wav_meta_path, 'r', encoding='utf-8') as f:
            rows = csv.DictReader(f, delimiter=',', quotechar='"')
            for row in rows:
                if speaker_meta_path is not None:
                    spkid = row[AAL_CONST[SPEAKER_ID]]
                    print(row)
                    print(spkid)
                    row.update(speakers[spkid])
                if utterance_meta_path is not None:
                    utt_id = row[AAL_CONST[UTTERANCE_ID]]
                    row.update(utterances[utt_id])

                row = change_keys(row, reverse_dict(AAL_CONST))
                row[TEXT] = row[TEXT].upper()
                container.append(row)
        return cls(container)

    @classmethod
    def from_aal_format(cls, wav_meta_path, speaker_meta_path=None, utterance_meta_path=None):
        container = pd.read_csv(wav_meta_path)

        if speaker_meta_path is not None:
            spk = pd.read_csv(speaker_meta_path)
            container = container.join(spk.set_index(AAL_CONST[SPEAKER_ID]), on = AAL_CONST[SPEAKER_ID])
        
        if utterance_meta_path is not None:
            utt = pd.read_csv(utterance_meta_path)
            container = container.join(utt.set_index(AAL_CONST[UTTERANCE_ID]), on = AAL_CONST[UTTERANCE_ID])

        try:
            container[AAL_CONST[TEXT]] = container[AAL_CONST[TEXT]].fillna('')
        # For unlabeled
        except:
            pass 
        container = container.to_dict(orient='records')
        container = [change_keys(row, reverse_dict(AAL_CONST), allow_mismatch=True) for row in container]
        
        return cls(container)
    
    @classmethod 
    def from_kaldi_format(cls, wav_scp_path, utt2spk_path, text_path):
        # TODO: Handle `sox`
        container = {}
        with open(wav_scp_path, 'r', encoding='utf-8') as wav_f, \
            open(utt2spk_path, 'r', encoding='utf-8') as utt2spk_f, \
            open(text_path, 'r', encoding='utf-8') as text_f:
            
            wav_reader = wav_f.readlines()
            for line in wav_reader:
                line = line.strip().split(' ', 1)
                utt_id = line[KALDI_CONST[UTTERANCE_ID]]
                path_info = line[KALDI_CONST[PATH]]
                # Handle sox
                # e.g: sox "A117016535#10028#9005550453037837.wav" -t wav - trim 0.89 =1.47|
                if path_info.endswith('|'):
                    # NOTE (Duy): Do not support path that contains space
                    path_ele = path_info.split(' ')
                    path = path_ele[1]
                    start = float(path_ele[-2])
                    end = float(path_ele[-1].replace('|', '').replace('=', ''))
                    container[utt_id] = {UTTERANCE_ID: utt_id, PATH: path, START: start, END: end}
                else:
                    path = path_info
                    container[utt_id] = {UTTERANCE_ID: utt_id, PATH: path}

            utt2spk_reader = utt2spk_f.readlines()
            for line in utt2spk_reader:
                line = line.strip().split(' ', 1)
                utt_id = line[KALDI_CONST[UTTERANCE_ID]]
                spkid = line[KALDI_CONST[SPEAKER_ID]]
                container[utt_id][SPEAKER_ID] = spkid

            text_reader = text_f.readlines()
            for line in text_reader:
                line = line.strip().split(' ', 1)
                utt_id = line[KALDI_CONST[UTTERANCE_ID]]
                try:
                    text = line[KALDI_CONST[TEXT]]
                # Empty utterance
                except IndexError:
                    text = '' 
                container[utt_id][TEXT] = text
            
        container = [container[k] for k in container]
 
        return cls(container)

    @classmethod
    def from_dir(cls, audio_dir):
        container = []
        for filepath in get_all_file(audio_dir, 'wav'):
            filepath = os.path.relpath(filepath, audio_dir)
            container.append({PATH: filepath})
        return cls(container)

if __name__ == '__main__':
    # ds = WavDataset.from_dir('dummy_aal')
    # ds = WavDataset.from_kaldi_format(
    #     '/mnt/d/duy/data/350-test/kaldi/wav.scp',
    #     '/mnt/d/duy/data/350-test/kaldi/utt2spk',
    #     '/mnt/d/duy/data/350-test/kaldi/text'
    # )
    ds = WavDataset.from_aal_format(
        'dummy_aal/wav.csv',
        'dummy_aal/speaker.csv',
        'dummy_aal/utt.csv',
    )
    print(ds[0])
    