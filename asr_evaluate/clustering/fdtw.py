#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division

import numbers
import warnings
from collections import defaultdict

import numpy
import numpy as np
from fastdtw import fastdtw
from numba import njit, prange
from sklearn.metrics.pairwise import pairwise_distances
from tslearn.metrics import dtw_path
from tslearn.metrics.dtw_variants import compute_mask, njit_accumulated_matrix, _return_path
# from fastdtw import fastdtw
from tslearn.metrics.utils import _cdist_generic
from tslearn.utils import to_time_series

GLOBAL_CONSTRAINT_CODE = {None: 0, "": 0, "itakura": 1, "sakoe_chiba": 2}

def fast_dtw(s1, s2, radius=1):
    s1 = to_time_series(s1, remove_nans = True)
    s2 = to_time_series(s2, remove_nans = True)
    if len(s1) == 0 or len(s2) == 0:
        raise ValueError(
            "One of the input time series contains only nans or has zero length.")

    if s1.shape[1] != s2.shape[1]:
        raise ValueError("All input time series must have the same feature size.")


    return __fast_dtw(s1, s2, radius)

def __fast_dtw(s1, s2, radius):
        
    min_time_size = radius + 2

    if len(s1) < min_time_size or len(s2) < min_time_size:
        mask = compute_mask(
            s1, s2, GLOBAL_CONSTRAINT_CODE[None], sakoe_chiba_radius=None, itakura_max_slope=None
        )
        acc_cost_mat = njit_accumulated_matrix(s1, s2, mask=mask)
        path = _return_path(acc_cost_mat)
        return np.sqrt(acc_cost_mat[-1, -1]), path

    x_shrinked = __reduce_by_half(s1)
    y_shrinked = __reduce_by_half(s2)
    distance, path = \
        __fast_dtw(x_shrinked, y_shrinked, radius=radius)
    window = __expand_window(path, len(x), len(y), radius)
    return __dtw(x, y, window, dist=dist)

def fastdtw(x, y, radius=1, dist=None):
    ''' return the approximate distance between 2 time series with O(N)
        time and memory complexity

        Parameters
        ----------
        x : array_like
            input array 1
        y : array_like
            input array 2
        radius : int
            size of neighborhood when expanding the path. A higher value will
            increase the accuracy of the calculation but also increase time
            and memory consumption. A radius equal to the size of x and y will
            yield an exact dynamic time warping calculation.
        dist : function or int
            The method for calculating the distance between x[i] and y[j]. If
            dist is an int of value p > 0, then the p-norm will be used. If
            dist is a function then dist(x[i], y[j]) will be used. If dist is
            None then abs(x[i] - y[j]) will be used.

        Returns
        -------
        distance : float
            the approximate distance between the 2 time series
        path : list
            list of indexes for the inputs x and y

        Examples
        --------
        >>> import numpy as np
        >>> import fastdtw
        >>> x = np.array([1, 2, 3, 4, 5], dtype='float')
        >>> y = np.array([2, 3, 4], dtype='float')
        >>> fastdtw.fastdtw(x, y)
        (2.0, [(0, 0), (1, 0), (2, 1), (3, 2), (4, 2)])
    '''
    x, y, dist = __prep_inputs(x, y, dist)
    return __fastdtw(x, y, radius, dist)

def __difference(a, b):
    return abs(a - b)

def __norm(p):
    @njit
    def norm_impl(a,b):
        return np.sqrt(np.sum(np.power(np.abs(a-b), p)))
    return norm_impl

def __fastdtw(x, y, radius, dist):
    min_time_size = radius + 2

    if len(x) < min_time_size or len(y) < min_time_size:
        return dtw(x, y, dist=dist)

    x_shrinked = __reduce_by_half(x)
    y_shrinked = __reduce_by_half(y)
    distance, path = \
        __fastdtw(x_shrinked, y_shrinked, radius=radius, dist=dist)
    window = __expand_window(path, len(x), len(y), radius)
    return __dtw(x, y, window, dist=dist)

def __prep_inputs(x, y, dist):
    x = np.asanyarray(x, dtype='float')
    y = np.asanyarray(y, dtype='float')

    if x.ndim == y.ndim > 1 and x.shape[1] != y.shape[1]:
        raise ValueError('second dimension of x and y must be the same')
    if isinstance(dist, numbers.Number) and dist <= 0:
        raise ValueError('dist cannot be a negative integer')

    if dist is None:
        if x.ndim == 1:
            dist = __difference
        else: 
            dist = __norm(p=1)
    elif isinstance(dist, numbers.Number):
        dist = __norm(p=dist)

    return x, y, dist

def dtw(x, y, dist=None):
    ''' return the distance between 2 time series without approximation

        Parameters
        ----------
        x : array_like
            input array 1
        y : array_like
            input array 2
        dist : function or int
            The method for calculating the distance between x[i] and y[j]. If
            dist is an int of value p > 0, then the p-norm will be used. If
            dist is a function then dist(x[i], y[j]) will be used. If dist is
            None then abs(x[i] - y[j]) will be used.

        Returns
        -------
        distance : float
            the approximate distance between the 2 time series
        path : list
            list of indexes for the inputs x and y

        Examples
        --------
        >>> import numpy as np
        >>> import fastdtw
        >>> x = np.array([1, 2, 3, 4, 5], dtype='float')
        >>> y = np.array([2, 3, 4], dtype='float')
        >>> fastdtw.dtw(x, y)
        (2.0, [(0, 0), (1, 0), (2, 1), (3, 2), (4, 2)])
    '''
    x, y, dist = __prep_inputs(x, y, dist)
    return __dtw(x, y, None, dist)

def __dtw(x, y, window, dist):
    len_x, len_y = len(x), len(y)
    if window is None:
        window = [(i, j) for i in range(len_x) for j in range(len_y)]
    window = [(i + 1, j + 1) for i, j in window]
    # D = defaultdict(lambda: (float('inf'),))

    D = np.full((len_x + 1, len_y + 1, 3), np.inf)

    D[0, 0] = 0.
    for i, j in window:
        dt = dist(x[i-1], y[j-1])
        D[i,j] = min(D[i-1, j] + dt, D[i, j-1] + dt, D[i-1, j-1] + dt) 

        D[i, j] = min((D[i-1, j][0] + dt, i-1, j), (D[i, j-1][0] + dt, i, j-1),
                      (D[i-1, j-1][0] + dt, i-1, j-1), key=lambda a: a[0])
  
    path = []
    i, j = len_x, len_y

    while not (i == j == 0):
        path.append((i-1, j-1))
        print("i: ", i)
        i, j = D[i, j][1], D[i, j][2]
    path.reverse()
    return (D[len_x, len_y][0], path)

@njit()
def __reduce_by_half(x):
    return [(x[i] + x[1+i]) / 2 for i in range(0, len(x) - len(x) % 2, 2)]

@njit()
def __expand_window(path, len_x, len_y, radius):
    path_ = set(path)
    for i, j in path:
        for a, b in [(i + a, j + b)
                     for a in range(-radius, radius+1)
                     for b in range(-radius, radius+1)]:
            path_.add((a, b))

    window_ = set()
    for i, j in path_:
        for a, b in ((i * 2, j * 2), (i * 2, j * 2 + 1),
                     (i * 2 + 1, j * 2), (i * 2 + 1, j * 2 + 1)):
            window_.add((a, b))

    window = []
    start_j = 0
    for i in range(0, len_x):
        new_start_j = None
        for j in range(start_j, len_y):
            if (i, j) in window_:
                window.append((i, j))
                if new_start_j is None:
                    new_start_j = j
            elif new_start_j is not None:
                break
        start_j = new_start_j

    return window

from pyts.metrics import dtw as pyts_dtw
from tslearn.metrics import cdist_dtw
from tslearn.metrics.utils import _cdist_generic


def cdist_fast_dtw(dataset1, dataset2=None, n_jobs=None, verbose=0, dist=2, radius=1, *args, **kwargs):
    return _cdist_generic(
        fastdtw, 
        dataset1=dataset1, dataset2=dataset2, 
        n_jobs=n_jobs, verbose=verbose, compute_diagonal=False, 
        dist=dist, radius=radius
    )

if __name__ == '__main__':
    S = np.random.rand(10, 273, 97)
    dist = cdist_fast_dtw(S, dist=2, radius=10, verbose=1, n_jobs=4)
    print(dist)
    # s1 = np.random.rand(100, 97)
    # s2 = np.random.rand(50, 97)
    