import pickle

import numpy as np
import pandas as pd
import torch
from time import time
from tqdm import tqdm
import pickle

from tslearn import utils as tsu
from tslearn.metrics import cdist_dtw
# from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from sklearn_extra.cluster import CLARA

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

    mean = np.nanmean(long_x, axis=0)
    std = np.nanstd(long_x, axis=0)

    return mean, std

def z_normalize(arr, mean, std):
    # arr: List[Arr(T, V)]
    return list(map(lambda x: (x - mean)/std, arr))

def cdist_dtw_wrapper(feat_dim, X, Y=None, **kwds):
    N, T_F = X.shape()
    T = T_F//feat_dim


def get_model() :
    model = CLARA(
        n_clusters=30,
        metric=cdist_dtw,
        init='k-medoids++',
        max_iter=300,
        random_state=7,
        n_sampling = 1000,
        n_sampling_iter=5,
    )

    return model

def main():
    tqdm.pandas()
    df = pd.read_csv('/home/duy/github/asr_model_tesing/tmp/OAD2208/OAD2208-conformer_OAD350v2.0_OAD2204v0.9_lb_loss_grad_decompose-configID-Apr_04_2023.csv')
    CHECKPOINT = '/home/duy/github/asr_model_tesing/tmp/OAD2208/kmeans/exp/clara_target-pseudo_scale-none_5000.pkl'
    

    df_sub = get_grad_embedding(df, 5000)

    # real_grad = df_sub.real_grad
    pseudo_grad = df_sub.pseudo_grad

    print("Normalize data")
    # real_grad_mean, real_grad_std = compute_mean_std(real_grad, None)
    pseudo_grad_mean, pseudo_grad_std = compute_mean_std(pseudo_grad, None)

    # real_grad_normalize = z_normalize(real_grad, real_grad_mean, real_grad_std)
    pseudo_grad_normalize = z_normalize(pseudo_grad, pseudo_grad_mean, pseudo_grad_std)

    # X_train = tsu.to_time_series_dataset(real_grad_normalize)
    pseudo_X_train = tsu.to_time_series_dataset(pseudo_grad_normalize)

    print("Create model")
    model = get_model()

    print("Model fitting")

    start_t = time()
    model.fit(pseudo_X_train)
    elapse_t = time() - start_t 

    print("Fit time: ", elapse_t)

    with open(CHECKPOINT, 'wb') as f:
        pickle.dump(model, f)

if __name__=='__main__':
    main()