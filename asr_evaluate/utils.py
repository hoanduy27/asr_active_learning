import os

import pandas as pd


def reverse_dict(dct):
    return {v:k for k,v in dct.items()}

def change_keys(dct, mapper, allow_mismatch=False):
    if allow_mismatch:
        return {mapper.get(k, k): v for k,v in dct.items()}
    else:
        return {mapper[k]: v for k,v in dct.items() if k in mapper}

def get_val_or_none(dct, key):
    val = dct.get(key, None)
    return None if pd.isna(val) else val

def get_all_file(dir, ext='wav'):
    all_files = []
    file_list = os.listdir(dir)
    for entry in file_list:
        full_path = os.path.join(dir, entry)
        if os.path.isdir(full_path):
            all_files += get_all_file(full_path, ext)
        elif full_path.endswith(f'.{ext}'):
            all_files.append(full_path)
    return all_files

def split_name_ext(filepath, return_parent=True):
    name = os.path.basename(filepath)
    name = name.split('.')
    name, ext = '.'.join(name[:-1]), name[-1]
    if return_parent:
        return name, ext, os.path.dirname(filepath)
    return name, ext    

def add_postfix(filepath, postfix, return_path=True):
    name, ext, par = split_name_ext(filepath, return_parent=True)
    basename = '%s%s.%s'%(name, postfix, ext)
    if return_path:
        return os.path.join(par, basename)
    else:
        return basename