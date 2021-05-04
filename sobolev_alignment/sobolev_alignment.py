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

import os, sys
import numpy as np
import pandas as pd
from pickle import load, dump
import scipy
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import torch
from anndata import AnnData
import scvi

from .generate_artificial_sample import parallel_generate_samples
from .krr_approx import KRRApprox


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
            source_krr_params: dict = None,
            target_krr_params: dict = None,
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
        self.krr_params = {
            'source': source_krr_params if source_krr_params is not None else {'method': 'falkon'},
            'target': target_krr_params if target_krr_params is not None else {'method': 'falkon'}
        }
        self._check_same_kernel() # Check whether source and target have the same kernel

        # Create scVI models
        self.n_jobs = n_jobs

    def _check_same_kernel(self):
        """
        Same kernel has to be used for source and kernel KRR.
        """
        if 'kernel' in self.krr_params['source'] or 'kernel' in self.krr_params['target']:
            assert self.krr_params['source']['kernel'] == self.krr_params['target']['kernel']
        if 'kernel_params' in  self.krr_params['source'] or 'kernel_params' in  self.krr_params['target']:
            assert self.krr_params['source']['kernel_params'] == self.krr_params['target']['kernel_params']

    def fit(
            self,
            X_source: AnnData,
            X_target: AnnData,
            source_batch_name: str = None,
            target_batch_name: str = None,
            n_artificial_samples: int = int(10e5)
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
        self.artificial_samples_, self.artificial_batches_ = self._generate_artificial_samples(n_artificial_samples)
        self.artificial_embeddings_ = {
            x: self._embed_artificial_samples(x)
            for x in ['source', 'target']
        }
        for x in self.krr_params:
            if self.krr_params[x]['method'] == 'falkon':
                self.artificial_samples_[x] = torch.Tensor(self.artificial_samples_[x])
                self.artificial_embeddings_[x] = torch.Tensor(self.artificial_embeddings_[x])
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
        artificial_batches = {
            x: self._sample_batches(n_artificial_samples=n_artificial_samples, data=x)
            for x in ['source', 'target']
        }
        artificial_samples = {
            x: parallel_generate_samples(
                sample_size=n_artificial_samples,
                batch_names=artificial_batches[x],
                lib_size=lib_size[x],
                model=self.scvi_models[x],
                return_dist=False,
                batch_size=min(10**4, n_artificial_samples),
                n_jobs=self.n_jobs
            )
            for x in ['source', 'target']
        }
        return artificial_samples, artificial_batches

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

    def _embed_artificial_samples(self, data: str):
        # Format artificial samples to be fed into scVI.
        x_train = self.artificial_samples_[data]
        train_obs = pd.DataFrame(
            np.array(self.artificial_batches_[data]),
            columns=[self.batch_name[data]]
        )
        if self.
        x_train_an = AnnData(x_train.cpu().detach().numpy(),
                             obs=train_obs)
        x_train_an.layers['counts'] = x_train_an.X.copy()

        # Forward these formatted samples
        return self.scvi_models[data].get_latent_representation(x_train_an)

    def _approximate_encoders(self):
        """
        Approximate the encoder by a KRR regression
        """
        self.approximate_krr_regressions_ = {
            x: KRRApprox(**self.krr_params[x])
            for x in ['source', 'target']
        }

        for x in ['source', 'target']:
            self.approximate_krr_regressions_[x].fit(
                self.artificial_samples_[x],
                self.artificial_embeddings_[x]
            )

        return True

    def _compare_approximated_encoders(self):
        self.M_X = self._compute_cosine_sim_intra_dataset('source')
        self.M_Y = self._compute_cosine_sim_intra_dataset('target')
        self.M_XY = self._compute_cross_cosine_sim()

        sqrt_inv_M_X = scipy.linalg.sqrtm(np.linalg.pinv(self.M_X))
        sqrt_inv_M_Y = scipy.linalg.sqrtm(np.linalg.pinv(self.M_Y))
        self.cosine_sim = sqrt_inv_M_X.dot(self.M_XY).dot(sqrt_inv_M_Y)

    def _compute_cosine_sim_intra_dataset(
            self,
            data : str
    ):
        """
        Compute M_X if data='source', or M_Y if data='target'.

        :param data:
        :return:
        """
        krr_clf = self.approximate_krr_regressions_[data]
        K = krr_clf.kernel_(
            krr_clf.training_data_[krr_clf.ridge_samples_idx_],
            krr_clf.training_data_[krr_clf.ridge_samples_idx_]
        )
        K = torch.Tensor(K)
        return krr_clf.sample_weights_.T.matmul(K).matmul(krr_clf.sample_weights_)

    def _compute_cross_cosine_sim(self):
        K_XY = self.approximate_krr_regressions_['target'].kernel_(
            self.approximate_krr_regressions_['source'].training_data_[self.approximate_krr_regressions_['source'].ridge_samples_idx_],
            self.approximate_krr_regressions_['target'].training_data_[self.approximate_krr_regressions_['target'].ridge_samples_idx_]
        )
        K_XY = torch.Tensor(K_XY)
        return self.approximate_krr_regressions_['source'].sample_weights_.T.matmul(K_XY).matmul(self.approximate_krr_regressions_['target'].sample_weights_)

    def _compute_principal_vectors(self):
        cosine_svd = np.linalg.svd(self.cosine_sim, full_matrices=False)
        self.principal_angles = cosine_svd[1]
        self.untransformed_rotations_ = {
            'source': cosine_svd[0],
            'target': cosine_svd[2].T
        }

    def save(
            self,
            folder: str = '.'
    ):
        if not os.path.exists(folder) and not os.path.isdir(folder):
            os.mkdir(folder)

        # Dump scVI models
        for x in self.scvi_models:
            dump(
                self.scvi_models[x],
                open('%s/scvi_model_%s.pkl'%(folder, x), 'wb')
            )

        # Dump the KRR:
        for x in self.approximate_krr_regressions_:
            self.approximate_krr_regressions_[x].save('%s/krr_approx_%s'%(folder, x))

        # Save params
        pd.DataFrame(self.krr_params).to_csv('%s/krr_params.csv'%(folder))
        dump(self.krr_params, open('%s/krr_params.pkl'%(folder), 'wb'))

        for param_t in ['model', 'plan', 'train']:
            df = pd.DataFrame([self.scvi_params[x][param_t] for x in ['source', 'target']])
            df.to_csv('%s/scvi_params_%s.csv'%(folder, param_t))
        dump(self.scvi_params, open('%s/scvi_params.pkl'%(folder), 'wb'))

        # Save results
        torch.save(self.M_X, open('%s/alignment_M_X.pt'%(folder), 'wb'))
        torch.save(self.M_Y, open('%s/alignment_M_Y.pt'%(folder), 'wb'))
        torch.save(self.M_XY, open('%s/alignment_M_XY.pt'%(folder), 'wb'))
        np.save(open('%s/alignment_cosine_sim.npy'%(folder), 'wb'), self.cosine_sim)
        pd.DataFrame(self.cosine_sim).to_csv('%s/alignment_cosine_sim.csv'%(folder))
        np.save(open('%s/alignment_principal_angles.npy'%(folder), 'wb'), self.principal_angles)
        pd.DataFrame(self.principal_angles).to_csv('%s/alignment_principal_angles.csv'%(folder))


    def plot_training_metrics(self, folder: str='.'):
        """
        Plot the different training metric for the source and target scVI modules.


        """

        if not os.path.exists(folder) and not os.path.isdir(folder):
            os.mkdir(folder)

        for x in self.scvi_models:
            for metric in self.scvi_models[x].history:
                plt.figure(figsize=(6, 4))
                plt.plot(self.scvi_models[x].history[metric])
                plt.xlabel('Epoch', fontsize=20, color='black')
                plt.ylabel(metric, fontsize=20, color='black')
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.tight_layout()
                plt.savefig('%s/%s_model_train_%s.png'%(folder, x, metric), dpi=300)
                plt.show()