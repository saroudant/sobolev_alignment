"""
SOBOLEV ALIGNMENT

@author: Soufiane Mourragui

Main module for the Sobolev Alignment framework.


Notes
-------
	-

References
-------
"""

import numpy as np
from pickle import load, dump
import scipy
from joblib import Parallel, delayed
import torch
from anndata import AnnData
import scvi

from .generate_artificial_sample import parallel_generate_samples


class SobolevAlignment:
    """

    """

    default_scvi_params = {
        'model': {},
        'plan': {},
        'train': {}
    }

    def __init__(
            self,
            source_scvi_params: dict = None,
            target_scvi_params: dict = None,
            source_krr_params: dict = {},
            target_krr_params: dict = {},
            n_jobs=1
    ):
        """
        Parameters
        ----------
        source_scvi_params
            Dictionary with scvi params for the source dataset. Must have three keys, each assigned to a dictionary
            of params: model, plan and train.
        target_scvi_params
            Dictionary with scvi params for the target dataset. Must have three keys, each assigned to a dictionary
            of params: model, plan and train.
        """
        # scVI params
        self.scvi_params = {
            'source': source_scvi_params if source_scvi_params is not None else self.default_scvi_params,
            'target': target_scvi_params if target_scvi_params is not None else self.default_scvi_params
        }

        # KRR params
        self.source_krr_params = source_krr_params
        self.target_krr_params = target_krr_params

        # Create scVI models
        self.n_jobs = 1


    def fit(
            self,
            X_source: AnnData,
            X_target: AnnData,
            source_batch_name: str = None,
            target_batch_name: str = None,
            n_artificial_samples: int = 10e5
    ):
        """
        Parameters
        ----------
        X_source
            Source data.
        X_target
            Target data.
        """

        # Train VAE
        self.training_data = {
            'source': X_source,
            'target': X_target
        }
        self.batch_name = {
            'source': source_batch_name,
            'target': target_batch_name
        }
        self._train_scvi_modules()

        # Approximation by kernel machines
        self.artificial_samples_ = self._generate_artificial_samples(n_artificial_samples)
        self._approximate_encoders()

        # Comparison and alignment
        self._compare_approximated_encoders()
        self._compute_principal_vectors()

        return self


    def _train_scvi_modules(self):
        """
        Train the scVI models based on data given and specifications.
        """
        self.scvi_models = {}

        for x in ['source', 'target']:
            self.training_data[x].layers["counts"] = self.training_data[x].X.copy()
            scvi.data.setup_anndata(
                self.training_data[x],
                layer='counts',
                batch_key=self.batch_name[x]
            )

            self.scvi_models[x] = scvi.model.SCVI(
                self.training_data[x],
                **self.scvi_params[x]['model']
            )
            self.scvi_models[x].train(
                plan_kwargs=self.scvi_params[x]['plan'],
                **self.scvi_params[x]['train'])

        return True


    def _generate_artificial_samples(
            self,
            n_artificial_samples: int
    ):
        """
        Sample from the normal distribution associated to the latent space (for either source or target VAE model),
        generate some new data and process to recompute a new latent.

        Parameters
        ----------
        n_artificial_samples
            Number of artificial samples to produce for source and for target.

        Returns
        ----------
        artificial_data: dict
            Dictionary containing the generated data for both source and target
        """

        lib_size = self._compute_batch_library_size()
        return {
            x: parallel_generate_samples(
                sample_size=n_artificial_samples,
                batch_names=self._sample_batches(n_artificial_samples=n_artificial_samples, data=x),
                lib_size=lib_size[x],
                model=self.scvi_models[x],
                return_dist=False,
                batch_size=min(10**3, n_artificial_samples),
                n_jobs=self.n_jobs
            )
            for x in ['source', 'target']
        }


    def _compute_batch_library_size(self):
        if self.batch_name['source'] is None or self.batch_name['target'] is None:
            return {
                x : float(np.mean(np.sum(self.training_data[x].X, axis=1)))
                for x in self.training_data
            }

        unique_batches = {
            x: np.unique(self.training_data[x].obs[self.batch_name[x]])
            for x in self.training_data
        }

        return {
            x: {
                str(b): float(np.mean(np.sum(self.training_data[x][self.training_data[x].obs[self.batch_name[x]] == b].X, axis=1)))
                for b in unique_batches[x]
            }
            for x in self.training_data
        }

    def _sample_batches(self, n_artificial_samples, data):
        """
        Sample batches for either source or target.
        """

        if self.batch_name[data] is None:
            return None

        return np.random.choice(
            self.training_data[data].obs[self.batch_name[data]].values,
            size=int(n_artificial_samples)
        )


    def _approximate_encoders(self):
        pass


    def _compare_approximated_encoders(self):
        pass


    def _compute_principal_vectors(self):
        pass