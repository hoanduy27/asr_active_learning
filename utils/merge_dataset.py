import os
import sys
import functools as ft
from espnet2.active_learning.dataio.export_format import Exporter as exp
from espnet2.active_learning.dataio.dataset import WavDataset as wd 
import glob

def merge_dataset(sources, dest):
    sources = ft.reduce(
        lambda cur, next: cur + glob.glob(next), 
        sources,
        []
    )

    merged_ds = ft.reduce(
        lambda ds, source: ds + wd.from_aal_format(source), 
        sources,
        wd([])
    )

    exp.to_kaldi_format(merged_ds, dest)


if __name__ == "__main__":
    try:
        dest = sys.argv[1]
        src = sys.argv[2:]
    except:
        print(f'<usage>: {sys.argv[0]} <dest> <src 1> <src 2> ... <src N>')
        exit()
    
    merge_dataset(src, dest)