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
import scipy
from itertools import combinations_with_replacement
from joblib import Parallel, delayed


def combination_to_idx(idx, p):
    return np.array([np.sum(np.array(idx) == i) for i in range(p)])


def basis(x, k, gamma):
    if k == 0:
        return np.ones(x.shape[0])
    product = x
    for i in range(1,k):
        product = np.multiply(x,product)
    coef = np.power(2*gamma,k/2) / np.sqrt(scipy.math.factorial(k))

    return coef * product


def combinatorial_product(x, idx, gamma):
    prod = np.prod([
        basis(x[:,i], k, gamma)
        for i,k in enumerate(combination_to_idx(idx, x.shape[1]))
    ], axis=0)
    return prod


def interaction_name(gene_names, combi):
    combin_name = [
        '%s^%s' % (x, c)
        for x, c in zip(gene_names, combi) if c > 0
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
    print('START %s' % (d), flush=True)
    combin_results = Parallel(n_jobs=n_jobs, verbose=1, max_nbytes=1e6)(
        delayed(higher_order_interaction_wrapper)(data, x, gamma, gene_names)
        for x in combinations_with_replacement(np.arange(data.shape[1]), r=d)
    )
    combinations_features = np.array([c[0] for c in combin_results]).T
    combinations_features = np.diag(sample_offset).dot(combinations_features)

    return pd.DataFrame(
        data=combinations_features,
        columns=[c[1] for c in combin_results]
    )

# def higher_order_contribution(
#         d: int,
#         data: np.array,
#         sample_offset: np.array,
#         gene_names: list,
#         gamma: float,
#         n_jobs=1
# ):
#     print('START %s' % (d), flush=True)
#     combinations = []
#     combination_names = []
#     for x in combinations_with_replacement(np.arange(data.shape[1]), r=d):
#         combinations.append(x)
#         # combinations.append(combinatorial_product(data, x, gamma))
#         combination_names.append(interaction_name(gene_names, combination_to_idx(x, data.shape[1])))
#     combinations = Parallel(n_jobs=n_jobs, verbose=1)(
#         delayed(combinatorial_product)(data, x, gamma) for x in combinations
#     )
#
#     combinations_features = np.array(combinations).T
#     combinations_features = np.diag(sample_offset).dot(combinations_features)
#
#     return pd.DataFrame(
#         data=combinations_features,
#         columns=combination_names
#     )

def _compute_offset(data, gamma):
    sample_offset = np.linalg.norm(data, axis=1)
    return np.exp(-gamma * np.power(sample_offset, 2))
