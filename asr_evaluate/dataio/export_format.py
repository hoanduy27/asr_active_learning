import csv
import os
from asr_evaluate.CONFIG import *
from asr_evaluate.utils import *

class Exporter:
    @staticmethod
    def to_aal_format(dataset, format_dir):
        os.makedirs(format_dir, exist_ok=True)
        wav_filepath = os.path.join(format_dir, 'wav.csv')
        spk_filepath = os.path.join(format_dir, 'speaker.csv')
        utt_filepath = os.path.join(format_dir, 'utt.csv')
        
        WAV_COLUMNS = [UTTERANCE_ID, TEXT, PATH, SPEAKER_ID, START, END]
        SPK_COLUMNS = [SPEAKER_ID, NAME, GENDER, AGE, REGION]
        UTT_COLUMNS = [UTTERANCE_ID, DEVICE, CONDITION]

        explored_speaker = set()

        for filepath, columns in zip([wav_filepath, spk_filepath, utt_filepath], [WAV_COLUMNS, SPK_COLUMNS, UTT_COLUMNS]):
            with open(filepath, 'w', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[AAL_CONST[k] for k in columns])
                writer.writeheader()
                for data in dataset:
                    row = {AAL_CONST[k]: get_val_or_none(data, k) for k in columns}

                    if columns == SPK_COLUMNS:
                        speaker_id = data[SPEAKER_ID]
                        if speaker_id not in explored_speaker:
                            explored_speaker.add(speaker_id)
                            writer.writerow(row)
                        else:
                            pass
                    else:
                        writer.writerow(row)


    @staticmethod
    def to_kaldi_format(dataset, format_dir, pipe_sox=False):
        """
        Format dataset to kaldi 
        - format_dir: Target directory
        - pipe_sox: If True, sox command will be piped in wav.scp whenever `start_s` and `end_s` is not None, 
        Pipe: sox "$1" -t wav - trim <start> <end-start> |
        """
        os.makedirs(format_dir, exist_ok=True)
        wav_filepath = os.path.join(format_dir, 'wav.scp')
        utt2spk_filepath = os.path.join(format_dir, 'utt2spk')
        spk2utt_filepath = os.path.join(format_dir, 'spk2utt')
        text_filepath = os.path.join(format_dir, 'text')
        
        dataset = sorted(dataset, key=lambda x: x[UTTERANCE_ID])
        with open(wav_filepath, 'w', encoding='utf-8') as f:
            for data in dataset:
                utterance_id = data[UTTERANCE_ID]
                path = data[PATH]
                start_s = get_val_or_none(data, START)
                end_s = get_val_or_none(data, END)

                if pipe_sox and start_s is not None and end_s is not None:
                    f.write('%s sox "%s" -t wav - trim %s =%s|\n'%(utterance_id, path, str(start_s), str(end_s)))
                else:
                    f.write('%s %s\n'%(utterance_id, path))

        with open(utt2spk_filepath, 'w', encoding='utf-8') as f:
            for data in dataset:
                f.write('%s %s\n'%(data[UTTERANCE_ID], data[SPEAKER_ID]))

        with open(text_filepath, 'w', encoding='utf-8') as f:
            for data in dataset:
                f.write('%s %s\n'%(data[UTTERANCE_ID], data[TEXT]))

        # utt2spk to spk2utt
        spk_list = {}
        for data in dataset:
            speaker_id = data[SPEAKER_ID]
            if speaker_id not in spk_list:
                spk_list[speaker_id] = []
            spk_list[speaker_id].append(data[UTTERANCE_ID])
        with open(spk2utt_filepath, 'w', encoding='utf-8') as f:
            for speaker_id, utt_list in spk_list.items():
                f.write('%s %s\n'%(speaker_id, ' '.join(utt_list)))

if __name__ == '__main__':
    from asr_evaluate.dataio.dataset import WavDataset

    ds = WavDataset.from_aal_format(
        'dummy/aal/wav.csv'
    )
    # print(ds[0])
    Exporter.to_kaldi_format(ds, 'dummy/aal2kaldi', True)
