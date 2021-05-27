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
import seaborn as sns
from pickle import load, dump
import gc
import scipy
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import torch
from anndata import AnnData
import scvi

from .generate_artificial_sample import parallel_generate_samples
from .krr_approx import KRRApprox
from .kernel_operations import mat_inv_sqrt
from .feature_analysis import higher_order_contribution, _compute_offset
from .multi_krr_approx import MultiKRRApprox


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
            n_artificial_samples: int = int(10e5),
            fit_vae: bool = True,
            krr_approx: bool=True,
            sample_artificial: bool=True,
            n_samples_per_sample_batch: int=10**6,
            frac_save_artificial: float = 0.1,
            save_mmap: str = None,
            log_input: bool=False,
            n_krr_clfs: int=1
    ):
        """
        Parameters
        ----------
        X_source
            Source data.
        X_target
            Target data.
        """

        self.training_data = {
            'source': X_source,
            'target': X_target
        }
        self.batch_name = {
            'source': source_batch_name,
            'target': target_batch_name
        }

        # Train VAE
        if fit_vae:
            self._train_scvi_modules()

        if krr_approx:
            self.lib_size = self._compute_batch_library_size()

            self.approximate_krr_regressions_ = {}
            if sample_artificial:
                self.artificial_samples_ = {}
                self.artificial_embeddings_ = {}
            for data_source in ['source', 'target']:
                self._train_krr(
                    data_source=data_source,
                    n_artificial_samples=n_artificial_samples,
                    sample_artificial=sample_artificial,
                    save_mmap=save_mmap,
                    log_input=log_input,
                    n_samples_per_sample_batch=n_samples_per_sample_batch,
                    frac_save_artificial=frac_save_artificial,
                    n_krr_clfs=n_krr_clfs
                )

            # Comparison and alignment
            self._compare_approximated_encoders()
            self._compute_principal_vectors()

        return self

    def _train_krr(
            self,
            data_source:str,
            n_artificial_samples: int,
            sample_artificial: bool=True,
            save_mmap: str = None,
            log_input: bool = True,
            n_samples_per_sample_batch:int = 10**5,
            frac_save_artificial: float=0.1,
            n_krr_clfs: int = 1
    ):

        if n_krr_clfs == 1:
            self.approximate_krr_regressions_[data_source] = self._train_one_krr(
                data_source=data_source,
                n_artificial_samples=n_artificial_samples,
                sample_artificial=sample_artificial,
                save_mmap=save_mmap,
                log_input=log_input,
                n_samples_per_sample_batch=n_samples_per_sample_batch,
                frac_save_artificial=frac_save_artificial
            )
            return True

        elif n_krr_clfs > 1:
            self.approximate_krr_regressions_[data_source] = MultiKRRApprox()
            for idx in range(n_krr_clfs):
                krr_approx = self._train_one_krr(
                    data_source=data_source,
                    n_artificial_samples=n_artificial_samples,
                    sample_artificial=sample_artificial,
                    save_mmap=save_mmap,
                    log_input=log_input,
                    n_samples_per_sample_batch=n_samples_per_sample_batch,
                    frac_save_artificial=frac_save_artificial
                )
                self.approximate_krr_regressions_[data_source].add_clf(krr_approx)

            self.approximate_krr_regressions_[data_source].process_clfs()
            return True

    def _train_one_krr(
            self,
            data_source: str,
            n_artificial_samples: int,
            sample_artificial: bool = True,
            save_mmap: str = None,
            log_input: bool = True,
            n_samples_per_sample_batch: int = 10**5,
            frac_save_artificial: float = 0.1
    ):
        # Generate samples (decoder)
        if sample_artificial:
            artificial_samples, artificial_batches = self._generate_artificial_samples(
                data_source=data_source,
                n_artificial_samples=n_artificial_samples,
                large_batch_size=n_samples_per_sample_batch,
                save_mmap=save_mmap
            )

            # Compute embeddings (encoder)
            artificial_embeddings = self._embed_artificial_samples(
                artificial_samples=artificial_samples,
                artificial_batches=artificial_batches,
                data_source=data_source,
                large_batch_size=n_samples_per_sample_batch
            )
            del artificial_batches
            gc.collect()

            # Store in memmap
            print('START MEMMAPING', flush=True)
            artificial_samples = self._memmap_log_processing(
                data_source=data_source,
                artificial_samples=artificial_samples,
                artificial_embeddings=artificial_embeddings,
                save_mmap=save_mmap,
                log_input=log_input
            )
            print('END MEMMAPING', flush=True)
        else:
            artificial_samples = self.artificial_samples_[data_source]
            artificial_embeddings = self.artificial_embeddings_[data_source]

        # KRR approx
        krr_approx = self._approximate_encoders(
            data_source=data_source,
            artificial_samples=artificial_samples,
            artificial_embeddings=artificial_embeddings
        )

        # Subsample the artificial sample saved
        if sample_artificial:
            n_save = int(frac_save_artificial * n_artificial_samples)
            subsampled_idx = np.random.choice(a=np.arange(n_artificial_samples), size=n_save, replace=False)
            self.artificial_samples_[data_source] = artificial_samples[subsampled_idx]
            del artificial_samples
            gc.collect()
            self.artificial_embeddings_[data_source] = artificial_embeddings[subsampled_idx]
            del artificial_embeddings
            gc.collect()

        return krr_approx


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
            data_source: str,
            n_artificial_samples: int,
            large_batch_size: int=10**5,
            save_mmap: str=None
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

        batch_sizes = [large_batch_size] * (n_artificial_samples // large_batch_size) + [n_artificial_samples % large_batch_size]
        batch_sizes = [x for x in batch_sizes if x > 0]
        _generated_data = [
            self._generate_artificial_samples_batch(
                batch,
                data_source
            )
            for batch in batch_sizes
        ]
        _generated_data = list(zip(*_generated_data))
        artificial_samples = np.concatenate(_generated_data[0])
        artificial_batches_ = np.concatenate(_generated_data[1])
        del _generated_data
        gc.collect()

        if save_mmap is not None and type(save_mmap) == str:
            np.save(
                open('%s/%s_artificial_input.npy'%(save_mmap, data_source), 'wb'),
                artificial_samples
            )
            artificial_samples = np.load(
                '%s/%s_artificial_input.npy'%(save_mmap, data_source),
                mmap_mode='r'
            )
            gc.collect()

        return artificial_samples, artificial_batches_

    def _generate_artificial_samples_batch(
            self,
            n_artificial_samples: int,
            data_source: str
    ):
        artificial_batches = self._sample_batches(n_artificial_samples=n_artificial_samples, data=data_source)
        artificial_samples = parallel_generate_samples(
                sample_size=n_artificial_samples,
                batch_names=artificial_batches,
                lib_size=self.lib_size[data_source],
                model=self.scvi_models[data_source],
                return_dist=False,
                batch_size=min(10 ** 4, n_artificial_samples),
                n_jobs=self.n_jobs
        )
        non_zero_samples = torch.where(torch.sum(artificial_samples, axis=1) > 0)
        artificial_samples = artificial_samples[non_zero_samples]
        artificial_batches = artificial_batches[non_zero_samples]
        gc.collect()

        return artificial_samples, artificial_batches

    def _compute_batch_library_size(self):
        if self.batch_name['source'] is None or self.batch_name['target'] is None:
            return {
                x :np.sum(self.training_data[x].X, axis=1).astype(float)
                for x in self.training_data
            }

        unique_batches = {
            x: np.unique(self.training_data[x].obs[self.batch_name[x]])
            for x in self.training_data
        }

        return {
            x: {
                str(b): np.sum(self.training_data[x][self.training_data[x].obs[self.batch_name[x]] == b].X, axis=1).astype(float)
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

    def _embed_artificial_samples(
            self,
            artificial_samples,
            artificial_batches,
            data_source: str,
            large_batch_size=10**5
    ):
        # Divide in batches
        n_artificial_samples = artificial_samples.shape[0]
        batch_sizes = [large_batch_size] * (n_artificial_samples // large_batch_size) + [n_artificial_samples % large_batch_size]
        batch_sizes = [0] + list(np.cumsum([x for x in batch_sizes if x > 0]))
        batch_start = batch_sizes[:-1]
        batch_end = batch_sizes[1:]

        # Format artificial samples to be fed into scVI.
        embedding = []
        for start, end in zip(batch_start, batch_end):
            x_train = artificial_samples[start:end]
            train_obs = pd.DataFrame(
                np.array(artificial_batches[start:end]),
                columns=[self.batch_name[data_source]]
            )
            x_train_an = AnnData(x_train,
                                 obs=train_obs)
            x_train_an.layers['counts'] = x_train_an.X.copy()
            embedding.append(self.scvi_models[data_source].get_latent_representation(x_train_an))

        # Forward these formatted samples
        return np.concatenate(embedding)

    def _memmap_log_processing(
            self,
            data_source: str,
            artificial_samples,
            artificial_embeddings,
            save_mmap: str=None,
            log_input: bool=False
    ):
        # If no save_mmap, then no log
        if save_mmap is not None and type(save_mmap) == str:
            self._save_mmap = save_mmap
            self._memmap_embedding(data_source=data_source, artificial_embeddings=artificial_embeddings, save_mmap=save_mmap)

        self.krr_log_input_ = log_input
        if log_input:
            if save_mmap is not None and type(save_mmap) == str:
                artificial_samples = np.log10(artificial_samples + 1)

                #Re-save
                np.save(
                    open('%s/%s_artificial_input.npy' % (save_mmap, data_source), 'wb'),
                    artificial_samples
                )
                artificial_samples = np.load(
                    '%s/%s_artificial_input.npy' % (save_mmap, data_source),
                    mmap_mode='r'
                )
                gc.collect()

            else:
                artificial_samples = np.log10(artificial_samples + 1)

        return artificial_samples


    def _memmap_embedding(self, data_source, artificial_embeddings, save_mmap):
        np.save(
            open('%s/%s_artificial_embedding.npy' % (save_mmap, data_source), 'wb'),
            artificial_embeddings
        )
        artificial_embeddings = np.load(
            '%s/%s_artificial_embedding.npy' % (save_mmap, data_source),
            mmap_mode='r'
        )
        gc.collect()

        return artificial_embeddings

    def _approximate_encoders(self, data_source:str, artificial_samples, artificial_embeddings):
        """
        Approximate the encoder by a KRR regression
        """
        # self.approximate_krr_regressions_[data_source] = KRRApprox(**self.krr_params[data_source])
        #
        # self.approximate_krr_regressions_[data_source].fit(
        #     torch.from_numpy(artificial_samples),
        #     torch.from_numpy(artificial_embeddings)
        # )
        krr_approx = KRRApprox(**self.krr_params[data_source])

        krr_approx.fit(
            torch.from_numpy(artificial_samples),
            torch.from_numpy(artificial_embeddings)
        )

        return krr_approx

    def _compare_approximated_encoders(self):
        self.M_X = self._compute_cosine_sim_intra_dataset('source')
        self.M_Y = self._compute_cosine_sim_intra_dataset('target')
        self.M_XY = self._compute_cross_cosine_sim()

        sqrt_inv_M_X = mat_inv_sqrt(self.M_X)
        sqrt_inv_M_Y = mat_inv_sqrt(self.M_Y)
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
            krr_clf.anchors(),
            krr_clf.anchors()
        )
        K = torch.Tensor(K)
        return krr_clf.sample_weights_.T.matmul(K).matmul(krr_clf.sample_weights_)

    def _compute_cross_cosine_sim(self):
        K_XY = self.approximate_krr_regressions_['target'].kernel_(
            self.approximate_krr_regressions_['source'].anchors(),
            self.approximate_krr_regressions_['target'].anchors()
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


    def plot_cosine_similarity(
            self,
            folder: str='.',
            absolute_cos: bool=False
    ):
        if absolute_cos:
            sns.heatmap(np.abs(self.cosine_sim), cmap='seismic_r', center=0)
        else:
            sns.heatmap(self.cosine_sim, cmap='seismic_r', center=0)
        plt.xticks(fontsize=12, color='black')
        plt.yticks(fontsize=12, color='black')
        plt.xlabel('Tumor', fontsize=25, color='black')
        plt.ylabel('Cell lines', fontsize=25)
        plt.tight_layout()
        plt.savefig(
            '%s/%scosine_similarity.png' % (folder, 'abs_' if absolute_cos else ''),
            dpi=300
        )
        plt.show()

    def compute_error(self):
        """
        Compute error of the KRR approximation on the input (data used for VAE training) and used for KRR.
        :return:
        """
        return {
            'source': self._compute_error_one_type('source'),
            'target': self._compute_error_one_type('target')
        }

    def _compute_error_one_type(self, data_type):
        # KRR error of input data
        latent = self.scvi_models[data_type].get_latent_representation()
        if self.krr_log_input_:
            input_krr_pred =  self.approximate_krr_regressions_[data_type].transform(torch.Tensor(np.log10(self.training_data[data_type].X+1)))
        else:
            input_krr_pred =  self.approximate_krr_regressions_[data_type].transform(torch.Tensor(self.training_data[data_type].X))
        input_krr_diff = input_krr_pred - latent
        input_mean_square = torch.square(input_krr_diff)
        input_factor_mean_square = torch.mean(input_mean_square, axis=0)
        input_latent_mean_square = torch.mean(input_mean_square)
        input_factor_reconstruction_error = np.linalg.norm(input_krr_diff, axis=0) / np.linalg.norm(latent, axis=0)
        input_latent_reconstruction_error = np.linalg.norm(input_krr_diff) / np.linalg.norm(latent)
        del input_krr_pred, input_mean_square, input_krr_diff
        gc.collect()

        # KRR error of artificial data
        training_krr_diff = self.approximate_krr_regressions_[data_type].transform(torch.Tensor(self.artificial_samples_[data_type]))
        training_krr_diff = training_krr_diff - self.artificial_embeddings_[data_type]
        training_krr_factor_reconstruction_error = np.linalg.norm(training_krr_diff, axis=0) / np.linalg.norm(self.artificial_embeddings_[data_type], axis=0)
        training_krr_latent_reconstruction_error = np.linalg.norm(training_krr_diff) / np.linalg.norm(self.artificial_embeddings_[data_type])

        return {
            'factor':{
                'MSE': {
                    'input': input_factor_mean_square.detach().numpy(),
                    'artificial': torch.mean(torch.square(training_krr_diff), axis=0).detach().numpy()
                },
                'reconstruction_error': {
                    'input': input_factor_reconstruction_error,
                    'artificial': training_krr_factor_reconstruction_error
                },
            },
            'latent':{
                'MSE': {
                    'input': input_latent_mean_square.detach().numpy(),
                    'artificial': torch.mean(torch.square(training_krr_diff)).detach().numpy()
                },
                'reconstruction_error': {
                    'input': input_latent_reconstruction_error,
                    'artificial': training_krr_latent_reconstruction_error
                },
            }
        }

    def feature_analysis(self,
                         max_order: int=1,
                         gene_names:list=None):

        # Make parameters
        if 'gamma' in self.krr_params['source']['kernel_params'] and 'gamma' in self.krr_params['target']['kernel_params']:
            gamma_s = self.krr_params['source']['kernel_params']['gamma']
            gamma_t = self.krr_params['target']['kernel_params']['gamma']
        elif 'sigma' in self.krr_params['source']['kernel_params'] and 'sigma' in self.krr_params['target']['kernel_params']:
            gamma_s = 1 / (2 * self.krr_params['source']['kernel_params']['sigma'] ** 2)
            gamma_t = 1 / (2 * self.krr_params['target']['kernel_params']['sigma'] ** 2)
        assert gamma_s == gamma_t
        self.gamma = gamma_s

        self.sample_offset = {
            x:_compute_offset(self.approximate_krr_regressions_[x].anchors(), self.gamma)
            for x in self.training_data
        }

        if gene_names is None:
            self.gene_names = self.training_data['source'].columns
        else:
            self.gene_names = gene_names

        self.basis_feature_weights_df = {
            x: higher_order_contribution(
                d=max_order,
                data=self.approximate_krr_regressions_[x].anchors().detach().numpy(),
                sample_offset=self.sample_offset[x],
                gene_names=self.gene_names,
                gamma=self.gamma,
                n_jobs=self.n_jobs
            )
            for x in self.training_data
        }

        self.factor_level_feature_weights_df = {
            x: pd.DataFrame(
                self.approximate_krr_regressions_[x].sample_weights_.T.detach().numpy(),
                index=np.arange(self.approximate_krr_regressions_[x].sample_weights_.T.shape[0]),
                columns=self.basis_feature_weights_df[x].index
            )
            for x in self.training_data
        }

        self.factor_level_feature_weights_df = {
            x: self.factor_level_feature_weights_df[x].dot(self.basis_feature_weights_df[x])
            for x in self.training_data
        }

        self.pv_level_feature_weights_df = {
            x: pd.DataFrame(
                self.untransformed_rotations_[x].T.dot(self.factor_level_feature_weights_df[x]),
                index=['PV %s'%(i) for i in range(self.untransformed_rotations_[x].shape[1])],
                columns=self.factor_level_feature_weights_df[x].columns
            )
            for x in self.training_data
        }


