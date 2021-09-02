"""
GENERATE ARTIFICIAL SAMPLE

@author: Soufiane Mourragui

Use the generative nature of VAE to generate new samples.

"""

import torch
import scvi
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def generate_samples(
        sample_size: int,
        batch_names,
        covariates_values,
        lib_size,
        model,
        return_dist: bool=False
):
    """

    Parameters
        ----------
        sample_size
            Number of samples to generate.
        batch_names
            List of batches for each samples to generate.
        lib_size
            Dictionary of mean library size per batch.
        model
            model to train
    """
    # Retrieve batch name from scVI one-hot-encoding
    if batch_names is not None:
        batch_name_ids = [
            np.where(model.scvi_setup_dict_['categorical_mappings']['_scvi_batch']['mapping'] == str(n))[0][0]
            for n in batch_names
        ]
        # Recover log library size (exponential)
        lib_size_samples = np.array([
            np.random.choice(lib_size[n], 1)[0]
            for n in batch_names
        ])
        lib_size_samples = np.log(lib_size_samples)
    else:
        batch_name_ids = []
        lib_size_samples = [lib_size] * int(sample_size)

    # Process covariates
    if covariates_values is None:
        cont_covs = None
    elif type(covariates_values) is pd.DataFrame:
        cont_covs = torch.Tensor(covariates_values.values.astype(float))
    elif type(covariates_values) is np.array:
        cont_covs = torch.Tensor(covariates_values)

    # Generate random noise
    z = torch.Tensor(np.random.normal(size=(int(sample_size), model.init_params_['non_kwargs']['n_latent'])))
    dist_param_samples = model.module.generative(
        z=z,
        library=torch.Tensor(np.array(lib_size_samples).reshape(-1, 1)),
        batch_index=torch.Tensor(np.array(batch_name_ids).reshape(-1, 1)),
        cont_covs=cont_covs
    )

    # Sample from distribution
    if model.module.gene_likelihood == 'zinb':
        samples = scvi.distributions.ZeroInflatedNegativeBinomial(
            mu=dist_param_samples['px_rate'],
            theta=dist_param_samples['px_r'],
            zi_logits=dist_param_samples['px_dropout']
        ).sample()
    elif model.module.gene_likelihood == 'nb':
        samples = scvi.distributions.NegativeBinomial(
            mu=dist_param_samples['px_rate'],
            theta=dist_param_samples['px_r']
        ).sample()
    elif model.module.gene_likelihood == "poisson":
        samples = torch.distributions.Poisson(
            dist_param_samples['px_rate']
        ).sample()
    else:
        raise ValueError(
            "{} reconstruction error not handled right now".format(
                model.module.gene_likelihood
            )
        )

    if return_dist:
        return dist_param_samples, samples
    else:
        return samples.cpu()


def parallel_generate_samples(sample_size,
                              batch_names,
                              covariates_values,
                              lib_size,
                              model,
                              return_dist=False,
                              batch_size=10**3,
                              n_jobs=1):
    results = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(generate_samples)(batch_size, batch_names[i:i+batch_size], covariates_values[i:i+batch_size], lib_size, model, return_dist)
            for i in range(0,sample_size,batch_size)
    )

    if return_dist:
        return results
    return torch.cat(results)