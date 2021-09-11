"""
FEATURE_ANALYSIS

@author: Soufiane Mourragui

Main module for analyzing kernel ridge regression.


Notes
-------
	-

References
-------
"""

import numpy as np
import pandas as pd
import gc
import scipy
from itertools import combinations_with_replacement
from joblib import Parallel, delayed
from functools import reduce


def combination_to_idx(idx, p):
    return np.array([np.sum(np.array(idx) == i) for i in range(p)])

def basis(x, k, gamma):
    if k == 0:
        return np.ones(x.shape[0])
    
    product = x
    for i in range(1,k):
        product = x.multiply(product)
    coef = np.power(2*gamma,k/2) / np.sqrt(scipy.math.factorial(k))

    return coef * product


def combinatorial_product(x, idx, gamma):
    prod = [
        basis(x[:,i], k, gamma)
        for i,k in enumerate(combination_to_idx(idx, x.shape[1])) if k > 0
    ]
    if len(prod) == 0:
        return 1
    return reduce(scipy.sparse.csc_matrix.multiply, prod)

def interaction_name(gene_combi):
    combin_name = [
        '%s^%s' % (g, r)
        for g, r in zip(*np.unique(gene_combi, return_counts=True))
    ]
    return '*'.join(combin_name) if len(combin_name) > 0 else '1'

def higher_order_interaction_wrapper(data, x, gamma, gene_names):
    return [
        combinatorial_product(data, x, gamma),
        interaction_name(gene_names, combination_to_idx(x, data.shape[1]))
    ]


def higher_order_contribution(
        d: int,
        data: np.array,
        sample_offset: np.array,
        gene_names: list,
        gamma: float,
        n_jobs=1
):
    sparse_data = scipy.sparse.csc_matrix(data)
    print('\t START FEATURES', flush=True)
    combinations_features = Parallel(n_jobs=n_jobs, verbose=1, max_nbytes=1e6, pre_dispatch=int(1.5*n_jobs))(
        delayed(combinatorial_product)(sparse_data, x, gamma)
        for x in combinations_with_replacement(np.arange(sparse_data.shape[1]), r=d)
    )
    gc.collect()

    print('\t START CONCATENATION', flush=True)
    print('\t\t START STACKING', flush=True)
    combinations_features = scipy.sparse.hstack(combinations_features, format='csc')
    print('\t\t START PRODUCT', flush=True)
    combinations_features = scipy.sparse.diags(sample_offset).dot(combinations_features)
    print('\t\t DENSIFY', flush=True)
    gc.collect()
    # combinations_features = combinations_features.todense()

    print('\t\t FIND NAMES', flush=True)
    combinations_names = Parallel(n_jobs=min(5,n_jobs), verbose=1, max_nbytes=1e4, pre_dispatch=int(1.5*min(5,n_jobs)))(
        delayed(interaction_name)(x)
        for x in combinations_with_replacement(gene_names, r=d)
    )

    # return pd.DataFrame(
    #     data=combinations_features,
    #     columns=combinations_names
    # )
    return pd.DataFrame.sparse.from_spmatrix(
        data=combinations_features,
        columns=combinations_names
    )
    

def _compute_offset(data, gamma):
    sample_offset = np.linalg.norm(data, axis=1)
    return np.exp(-gamma * np.power(sample_offset, 2))
