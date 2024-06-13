import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import pickle

# from tslearn import utils as tsu
# from tslearn.clustering import TimeSeriesKMeans, silhouette_score

from dtaidistance import dtw_ndim

# from sklearn_extra.cluster import KMedoids
import kmedoids

import argparse
import os

def get_grad_embedding(df, sample_size: int = None, seed=None):
    def extract_grad(path):
        grad_info = np.load(path, allow_pickle=True).tolist()

        # return grad_info['real_grad']

        return pd.Series(dict(
            real_grad = grad_info['real_grad'],
            pseudo_grad = grad_info['pseudo_grad']
        ))

    df_sub = df.copy()
    if sample_size is not None:
        df_sub = df.sample(sample_size, random_state=seed).reset_index()
    
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

def kmedoids_model(k) :
    model = kmedoids.KMedoids(
        n_clusters=k,
        metric='precomputed', 
        method='fasterpam',
        init='build', 
        max_iter=300, 
        random_state=7,
    )
    return model

def validate_dist_path(dist_path):
    return os.path.exists(dist_path)


def train(
    dataset, save_feat_path, checkpoint_path, k, n=None, seed=None
):
    tqdm.pandas()
    
    df = pd.read_csv(dataset)
    
    os.makedirs(save_feat_path, exist_ok=True)
    dist_path = os.path.join(save_feat_path, 'dist.npy')
    df_path = os.path.join(save_feat_path, 'sample.csv')
    centroid_path = os.path.join(save_feat_path, 'centroid_sample.csv')

    df_sub = get_grad_embedding(df, sample_size=n, seed=seed)
    df_sub.drop(columns=['real_grad', 'pseudo_grad']).to_csv(df_path, index=False)

    # real_grad = df_sub.real_grad
    pseudo_grad = df_sub.pseudo_grad

    print("Compute mu, sigma")
    # real_grad_mean, real_grad_std = compute_mean_std(real_grad, None)
    pseudo_grad_mean, pseudo_grad_std = compute_mean_std(pseudo_grad, None)

    print('normalize')
    # real_grad_normalize = z_normalize(real_grad, real_grad_mean, real_grad_std)
    pseudo_grad_normalize = z_normalize(pseudo_grad, pseudo_grad_mean, pseudo_grad_std)
    pseudo_grad_normalize = list(map(lambda x: np.asarray(x, np.float64), pseudo_grad_normalize))

    # X_train = tsu.to_time_series_dataset(real_grad_normalize)
    # pseudo_X_train = tsu.to_time_series_dataset(pseudo_grad_normalize)

    print("Compute dist")
    if not validate_dist_path(dist_path):
        X_dist = dtw_ndim.distance_matrix_fast(pseudo_grad_normalize).astype(np.float32)
        np.save(dist_path, X_dist)
    else:
        X_dist = np.load(dist_path)

    print(X_dist)


    print("Create model")
    model = kmedoids_model(k)

    print("Fit model")
    model.fit(X_dist)

    with open(checkpoint_path, 'wb') as f:
        pickle.dump(model, f)

    
    centroid_sample = df_sub.iloc[model.medoid_indices_]

    centroid_sample.drop(columns=['real_grad', 'pseudo_grad']).to_csv(centroid_path, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        required=True,
        help='Path to dataset'
    )
    parser.add_argument(
        '--feat',
        required=True,
        help='Path to save train feat'
    )
    parser.add_argument(
        '--ckpt',
        required=True,
        help='Path to save checkpoint'
    )
    parser.add_argument(
        '--k',
        required=True,
        type=int,
        help='Num of cluster'
    )
    parser.add_argument(
        '--n',
        required=False,
        default=None,
        type=int,
        help='Number of sample of dataset to fit'
    )
    parser.add_argument(
        '--seed',
        required=False,
        default=None,
        type=int,
        help='Random seed'
    )

    args = parser.parse_args()

    train(dataset=args.dataset, save_feat_path=args.feat, checkpoint_path=args.ckpt, k=args.k, n=args.n)

if __name__=='__main__':
    main()