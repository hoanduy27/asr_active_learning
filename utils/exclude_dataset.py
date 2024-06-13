from espnet2.active_learning.dataio.dataset import WavDataset as wd
from espnet2.active_learning.dataio.export_format import Exporter as exp 
import sys

def exclude_dataset(orig_set_path, exclude_set_path, result_dir):
    orig_set = wd.from_aal_format(orig_set_path)
    exclude_set = wd.from_aal_format(exclude_set_path)

    result_set = orig_set - exclude_set 

    exp.to_aal_format(result_set, result_dir)

if __name__ == "__main__":
    try:
        orig_set_path = sys.argv[1]
        exclude_set_path = sys.argv[2]
        result_dir = sys.argv[3]
    except:
        print(f"usage: {sys.argv[0]} <orig_set_path> <exclude_set_path> <result_dir>")
        exit(1)
    
    exclude_dataset(orig_set_path, exclude_set_path, result_dir)

