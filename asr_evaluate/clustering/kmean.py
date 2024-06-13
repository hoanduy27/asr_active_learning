import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import pickle

from tslearn import utils as tsu
from tslearn.clustering import TimeSeriesKMeans, silhouette_score

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

def kmeans() :
    model = TimeSeriesKMeans(
        n_clusters=30,
        metric="dtw",
        max_iter=10,
        max_iter_barycenter=30,
        dtw_inertia=True,
        verbose=1,
        n_jobs=4
    )
    return model


def main():
    tqdm.pandas()
    df = pd.read_csv('/home/duy/github/asr_model_tesing/tmp/OAD2208/OAD2208-conformer_OAD350v2.0_OAD2204v0.9_lb_loss_grad_decompose-configID-Apr_04_2023.csv')
    CHECKPOINT = '/home/duy/github/asr_model_tesing/tmp/OAD2208/kmeans/exp/kmean_target-pseudo_scale-none_2000.pkl'
    
    df_sub = get_grad_embedding(df, 2000)

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

    model = kmeans()

    model.fit(pseudo_X_train)

    with open(CHECKPOINT, 'wb') as f:
        pickle.dump(model, f)

if __name__=='__main__':
    main()