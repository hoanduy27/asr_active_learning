import os
import sys

import numpy as np
import pandas as pd 

from asr_evaluate.dataio.dataset import WavDataset
from asr_evaluate.dataio.export_format import Exporter

def random_split(orig_path, size, split_dir, dataset_dir):
    df = pd.read_csv(orig_path)
    shuffle_idx = np.random.permutation(df.index)

    segments = [shuffle_idx[i:i+size] for i in range(0, len(df), size)]

    dataset_segments = [
        WavDataset(df.loc[segment].to_dict(orient='records')) 
        for segment in segments
    ]

    dataset_segments_accum = []

    for ds in dataset_segments:
        if len(dataset_segments_accum) == 0:
            dataset_segments_accum.append(ds)
        else:
            dataset_segments_accum.append(dataset_segments_accum[-1] + ds)
    
    # Save
    for i,ds in enumerate(dataset_segments):
        dataset_name = f"round_{i+1}"
        Exporter.to_aal_format(ds, os.path.join(split_dir, dataset_name))
    
    for i,ds in enumerate(dataset_segments_accum):
        dataset_name = f"train_nodev_round_{i+1}"
        Exporter.to_kaldi_format(ds, os.path.join(dataset_dir, dataset_name))
    
        
if __name__ == "__main__":
    try:
        orig_path, size, split_dir, dataset_dir = sys.argv[1:5]
    except:
        print(f"usage: {sys.argv[0]} <orig_path> <size> <split_dir> <dataset_dir>")
        exit()

    random_split(orig_path, int(size), split_dir, dataset_dir)
