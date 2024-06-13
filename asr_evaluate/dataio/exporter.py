import argparse
from ast import arg
from asr_evaluate.dataio.dataset import *
from asr_evaluate.dataio.export_format import *

from_format_mapper = {
 #   'backend': WavDataset.from_backend_format,
    'aal': WavDataset.from_aal_format,
    'kaldi': WavDataset.from_kaldi_format
}

to_format_mapper = {
    'aal': Exporter.to_aal_format,
    'kaldi': Exporter.to_kaldi_format
}

def main(args):
    # digest format
    try:
        from_format_func = from_format_mapper[args.from_format]
    except:
        raise RuntimeError('Source format not supported')
    if args.from_format == 'backend':
        dataset = from_format_func(args.meta_dir)
    elif args.from_format == 'aal':
        wav_fp = os.path.join(args.meta_dir, 'wav.csv')
        spk_fp = os.path.join(args.meta_dir, 'speaker.csv')
        utt_fp = os.path.join(args.meta_dir, 'utt.csv')
        try:
            dataset = from_format_func(wav_fp, spk_fp, utt_fp)
        except:
            dataset = from_format_func(wav_fp)
    elif args.from_format == 'kaldi':
        wav_scp = os.path.join(args.meta_dir, 'wav.scp')
        utt2spk = os.path.join(args.meta_dir, 'utt2spk')
        text = os.path.join(args.meta_dir, 'text')
        dataset = from_format_func(wav_scp, utt2spk, text)
    # export format
    try:
        to_format_func = to_format_mapper[args.to_format]
    except:
        raise RuntimeError('Destination format not supported')

    to_format_func(dataset, args.format_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--meta_dir',
        '-src',
        required=True,
        help='Path to metadata'
    )
    parser.add_argument(
        '--from_format', 
        '-from',
        required=True,
        choices=from_format_mapper.keys(),
        help='Source format'
    )
    parser.add_argument(
        '--to_format',
        '-to',
        required=True,
        choices=to_format_mapper.keys(),
        help='Destionation format'
    )
    parser.add_argument(
        '--format_dir',
        '-dst',
        required=True,
        help='Path to save new format'
    )
    args = parser.parse_args()
    main(args)
    

    
