import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from tslearn import utils as tsu
from tslearn.clustering import silhouette_score

tqdm.pandas()
df = pd.read_csv('/home/duy/github/asr_model_tesing/tmp/OAD2208/OAD2208-conformer_OAD350v2.0_OAD2204v0.9_lb_loss_grad_decompose-configID-Apr_04_2023.csv')

def get_grad_embedding(df, sample_size: int = None):
    def extract_grad(path):
        grad_info = torch.load(path)

        # return grad_info['real_grad']

        return pd.Series(dict(
            real_grad = grad_info['real_grad'].numpy(),
            pseudo_grad = grad_info['pseudo_grad'].numpy()
        ))

    df_sub = df.copy()
    if sample_size is not None:
        df_sub = df.head(sample_size)
    
    # df_sub.loc[:, ('real_grad','pseudo_grad')] = df_sub.margin_grad_info_path.progress_apply(extract_grad)
    df_sub[['real_grad', 'pseudo_grad']] = df_sub.margin_grad_info_path.progress_apply(extract_grad)

    return df_sub 

def compute_mean_std(x: pd.Series, sample_size=None):
    # Compute global mean and std (get from a subset of df)
    sub_x = x
    if sample_size is not None:
        sub_x = x.sample(x)
    
    long_x = np.concatenate(sub_x, axis=0)

    mean = np.nanmean(np.log(long_x + 1e-7), axis=0)
    std = np.nanstd(np.log(long_x + 1e-7), axis=0)

    return mean, std

def z_normalize(arr, mean, std):
    # arr: List[Arr(T, V)]
    return list(map(lambda x: (np.log(x + 1e-7)-mean)/std, arr))

df_sub = get_grad_embedding(df, 2000)

real_grad = df_sub.real_grad
pseudo_grad = df_sub.pseudo_grad

print("Normalize data")
real_grad_mean, real_grad_std = compute_mean_std(real_grad, None)
pseudo_grad_mean, pseudo_grad_std = compute_mean_std(pseudo_grad, None)

real_grad_normalize = z_normalize(real_grad, real_grad_mean, real_grad_std)
pseudo_grad_normalize = z_normalize(pseudo_grad, pseudo_grad_mean, pseudo_grad_std)

X_train = tsu.to_time_series_dataset(real_grad_normalize)
pseudo_X_train = tsu.to_time_series_dataset(pseudo_grad_normalize)

# print("Load model")
# with open('/home/duy/github/asr_model_tesing/tmp/OAD2208/kmean_model.pkl', 'rb') as f:
#     model=pickle.load(f)

# with open('/home/duy/github/asr_model_tesing/tmp/OAD2208/pseudo_kmean_model.pkl', 'rb') as f:
#     pseudo_model=pickle.load(f)

# print("Compute ss")
# real_ss = silhouette_score(X_train, model['kmean'].labels_, verbose=1, n_jobs=4)
# pseudo_ss = silhouette_score(pseudo_X_train, pseudo_model['kmean'].labels_, verbose=1, n_jobs=4)

# print(real_ss)
# print(pseudo_ss)

from tslearn.metrics import cdist_dtw 
import numpy as np
pseudo_cdist_dtw = cdist_dtw(pseudo_X_train[:250], n_jobs=4)
np.save('/home/duy/github/asr_model_tesing/tmp/OAD2208/kmeans/pseudo_cdist_dtw_250.npy')