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
from copy import deepcopy
import scipy
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
from anndata import AnnData
import scvi

from .generate_artificial_sample import parallel_generate_samples
from .krr_approx import KRRApprox
from .kernel_operations import mat_inv_sqrt
from .feature_analysis import higher_order_contribution, _compute_offset
from .multi_krr_approx import MultiKRRApprox


# Default library size used when re-scaling artificial data
DEFAULT_LIB_SIZE = 10**3

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
        self.scaler_ = {}

        # Create scVI models
        self.n_jobs = n_jobs

        # Initialize some values
        self._frob_norm_param = None

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
            continuous_covariate_names: list = None,
            n_artificial_samples: int = int(10e5),
            fit_vae: bool = True,
            krr_approx: bool=True,
            sample_artificial: bool=True,
            n_samples_per_sample_batch: int=10**6,
            frac_save_artificial: float = 0.1,
            save_mmap: str = None,
            log_input: bool=False,
            n_krr_clfs: int=1,
            no_posterior_collapse=False,
            mean_center: bool=False,
            unit_std: bool=False,
            frob_norm_source: bool=False,
            lib_size_norm: bool=False
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

        self.continuous_covariate_names = {
            'source': continuous_covariate_names,
            'target': continuous_covariate_names
        }

        self._fit_params = {
            'sample_artificial': sample_artificial,
            'n_samples_per_sample_batch': n_samples_per_sample_batch,
            'frac_save_artificial': frac_save_artificial,
            'save_mmap': save_mmap,
            'log_input': log_input,
            'n_krr_clfs': n_krr_clfs,
            'no_posterior_collapse': no_posterior_collapse,
            'mean_center': mean_center,
            'unit_std': unit_std,
            'frob_norm_source': frob_norm_source,
            'lib_size_norm': lib_size_norm
        }

        # Train VAE
        if fit_vae:
            self._train_scvi_modules(no_posterior_collapse=no_posterior_collapse)

        if krr_approx:
            self.lib_size = self._compute_batch_library_size()

            self.approximate_krr_regressions_ = {}
            if sample_artificial:
                self.mean_center = mean_center
                self.unit_std = unit_std
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
                    n_krr_clfs=n_krr_clfs,
                    mean_center=self.mean_center,
                    unit_std=self.unit_std,
                    frob_norm_source=frob_norm_source,
                    lib_size_norm=lib_size_norm
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
            n_krr_clfs: int = 1,
            mean_center: bool=False,
            unit_std: bool=False,
            frob_norm_source: bool=False,
            lib_size_norm: bool=False
    ):

        if n_krr_clfs == 1:
            self.approximate_krr_regressions_[data_source] = self._train_one_krr(
                data_source=data_source,
                n_artificial_samples=n_artificial_samples,
                sample_artificial=sample_artificial,
                save_mmap=save_mmap,
                log_input=log_input,
                n_samples_per_sample_batch=n_samples_per_sample_batch,
                frac_save_artificial=frac_save_artificial,
                mean_center=mean_center,
                unit_std=unit_std,
                frob_norm_source=frob_norm_source,
                lib_size_norm=lib_size_norm
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
                    frac_save_artificial=frac_save_artificial,
                    frob_norm_source=frob_norm_source
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
            frac_save_artificial: float = 0.1,
            mean_center: bool=False,
            unit_std: bool=False,
            frob_norm_source: bool=False,
            lib_size_norm: bool=False
    ):
        # Generate samples (decoder)
        if sample_artificial:
            artificial_samples, artificial_batches, artificial_covariates = self._generate_artificial_samples(
                data_source=data_source,
                n_artificial_samples=n_artificial_samples,
                large_batch_size=n_samples_per_sample_batch,
                save_mmap=save_mmap
            )

            # Compute embeddings (encoder)
            artificial_embeddings = self._embed_artificial_samples(
                artificial_samples=artificial_samples,
                artificial_batches=artificial_batches,
                artificial_covariates=artificial_covariates,
                data_source=data_source,
                large_batch_size=n_samples_per_sample_batch
            )
            gc.collect()

            # If artificial samples must be normalized for library size
            if lib_size_norm:
                artificial_samples = self._correct_artificial_samples_lib_size(
                    artificial_samples=artificial_samples,
                    artificial_batches=artificial_batches,
                    artificial_covariates=artificial_covariates,
                    data_source=data_source,
                    large_batch_size=n_samples_per_sample_batch
                )
            del artificial_batches, artificial_covariates
            gc.collect()

            # Store in memmap
            artificial_samples = self._memmap_log_processing(
                data_source=data_source,
                artificial_samples=artificial_samples,
                artificial_embeddings=artificial_embeddings,
                save_mmap=save_mmap,
                log_input=log_input,
                mean_center=mean_center,
                unit_std=unit_std,
                frob_norm_source=frob_norm_source
            )
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
            # Remove data in memmap
            if save_mmap is not None:
                os.remove('%s/%s_artificial_input.npy'%(save_mmap, data_source))
            self.artificial_embeddings_[data_source] = artificial_embeddings[subsampled_idx]
            # Remove data in memmap
            if save_mmap is not None:
                os.remove('%s/%s_artificial_embedding.npy'%(save_mmap, data_source))
            del artificial_embeddings
            gc.collect()
            torch.cuda.empty_cache()

        return krr_approx


    def _train_scvi_modules(self, no_posterior_collapse=False):
        """
        Train the scVI models based on data given and specifications.
        """
        self.scvi_models = {}

        for x in ['source', 'target']:
            self.training_data[x].layers["counts"] = self.training_data[x].X.copy()
            scvi.data.setup_anndata(
                self.training_data[x],
                layer='counts',
                batch_key=self.batch_name[x],
                continuous_covariate_keys=self.continuous_covariate_names[x]
            )

            # Change covariates to float
            if self.continuous_covariate_names[x] is not None:
                for cov in self.continuous_covariate_names[x]:
                    self.training_data[x].obs[cov] = self.training_data[x].obs[cov].astype(np.float64)

            latent_variable_variance = np.zeros(1)
            save_iter = 0
            while np.any(latent_variable_variance<0.2):
                print('START TRAINING %s model number %s'%(x, save_iter), flush=True)
                try:
                    self.scvi_models[x] = scvi.model.SCVI(
                        self.training_data[x],
                        **self.scvi_params[x]['model']
                    )
                    self.scvi_models[x].train(
                        plan_kwargs=self.scvi_params[x]['plan'],
                        **self.scvi_params[x]['train'])
                except Exception as err:
                    print('\n SCVI TRAINING ERROR: \n %s \n\n\n\n'%(err))
                    latent_variable_variance = np.zeros(1)
                    continue

                if not no_posterior_collapse:
                    break
                else:
                    embedding = self.scvi_models[x].get_latent_representation()
                    latent_variable_variance = np.var(embedding, axis=0)
                    save_iter += 1

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
        artificial_covariates_ = pd.concat(_generated_data[2]) if _generated_data[2][0] is not None else None
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

        return artificial_samples, artificial_batches_, artificial_covariates_


    def _generate_artificial_samples_batch(
            self,
            n_artificial_samples: int,
            data_source: str
    ):
        artificial_batches = self._sample_batches(n_artificial_samples=n_artificial_samples, data=data_source)
        artificial_covariates = self._sample_covariates(n_artificial_samples=n_artificial_samples, data=data_source)
        artificial_samples = parallel_generate_samples(
                sample_size=n_artificial_samples,
                batch_names=artificial_batches,
                covariates_values=artificial_covariates,
                lib_size=self.lib_size[data_source],
                model=self.scvi_models[data_source],
                return_dist=False,
                batch_size=min(10 ** 4, n_artificial_samples),
                n_jobs=self.n_jobs
        )

        non_zero_samples = torch.where(torch.sum(artificial_samples, axis=1) > 0)
        artificial_samples = artificial_samples[non_zero_samples]
        if artificial_covariates is not None:
            artificial_covariates = artificial_covariates.iloc[non_zero_samples]
        if artificial_batches is not None:
            artificial_batches = artificial_batches[non_zero_samples]
        gc.collect()

        return artificial_samples, artificial_batches, artificial_covariates

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


    def _sample_covariates(self, n_artificial_samples, data):
        """
        Sample batches for either source or target.
        """


        if self.continuous_covariate_names[data] is None:
            return None

        return self.training_data[data].obs[self.continuous_covariate_names[data]].sample(
            n_artificial_samples,
            replace=True
        )


    def _embed_artificial_samples(
            self,
            artificial_samples,
            artificial_batches,
            artificial_covariates,
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
                columns=[self.batch_name[data_source]],
                index=np.arange(end-start)
            )
            if artificial_covariates is not None:
                train_obs = pd.concat(
                    [train_obs, artificial_covariates.iloc[start:end].reset_index(drop=True)],
                    ignore_index=True,
                    axis=1
                )
                train_obs.columns = [self.batch_name[data_source], *self.continuous_covariate_names[data_source]]

            x_train_an = AnnData(x_train,
                                 obs=train_obs)
            x_train_an.layers['counts'] = x_train_an.X.copy()
            embedding.append(self.scvi_models[data_source].get_latent_representation(x_train_an))

        # Forward these formatted samples
        return np.concatenate(embedding)


    def _correct_artificial_samples_lib_size(
            self,
            artificial_samples,
            artificial_batches,
            artificial_covariates,
            data_source: str,
            large_batch_size=10**5
    ):
        """
            Correct for library size the artificial samples.
        """
        # Divide in batches
        n_artificial_samples = artificial_samples.shape[0]
        batch_sizes = [large_batch_size] * (n_artificial_samples // large_batch_size) + [n_artificial_samples % large_batch_size]
        batch_sizes = [0] + list(np.cumsum([x for x in batch_sizes if x > 0]))
        batch_start = batch_sizes[:-1]
        batch_end = batch_sizes[1:]

        # Format artificial samples to be fed into scVI.
        artificial_samples = [artificial_samples[start:end] for start, end in zip(batch_start, batch_end)]
        for idx, (x_train, start, end) in enumerate(zip(artificial_samples, batch_start, batch_end)):
            train_obs = pd.DataFrame(
                np.array(artificial_batches[start:end]),
                columns=[self.batch_name[data_source]],
                index=np.arange(end-start)
            )
            if artificial_covariates is not None:
                train_obs = pd.concat(
                    [train_obs, artificial_covariates.iloc[start:end].reset_index(drop=True)],
                    ignore_index=True,
                    axis=1
                )
                train_obs.columns = [self.batch_name[data_source], *self.continuous_covariate_names[data_source]]

            x_train_an = AnnData(x_train,
                                 obs=train_obs)
            x_train_an.layers['counts'] = x_train_an.X.copy()
            artificial_samples[idx] = self.scvi_models[data_source].get_normalized_expression(
                x_train_an,
                return_numpy=True,
                library_size=DEFAULT_LIB_SIZE
            )

        artificial_samples = np.concatenate(artificial_samples)
        


        return artificial_samples


    def _memmap_log_processing(
            self,
            data_source: str,
            artificial_samples,
            artificial_embeddings,
            save_mmap: str=None,
            log_input: bool=False,
            mean_center: bool=False,
            unit_std: bool=False,
            frob_norm_source: bool=False
    ):

        # Save embedding
        if save_mmap is not None and type(save_mmap) == str:
            self._save_mmap = save_mmap
            self._memmap_embedding(data_source=data_source, artificial_embeddings=artificial_embeddings, save_mmap=save_mmap)

        self.krr_log_input_ = log_input
        if log_input:
            artificial_samples = np.log10(artificial_samples + 1)

            # Standard Scaler
            scaler_ = StandardScaler(with_mean=mean_center, with_std=unit_std)
            artificial_samples = scaler_.fit_transform(np.array(artificial_samples))

            # Frobenius norm scaling
            artificial_samples = self._frobenius_normalisation(data_source, artificial_samples, frob_norm_source)

            if save_mmap is not None and type(save_mmap) == str:
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
                pass

        else:
            # Frobenius norm scaling
            artificial_samples = self._frobenius_normalisation(data_source, artificial_samples, frob_norm_source)

        return artificial_samples


    def _frobenius_normalisation(self, data_source, artificial_samples, frob_norm_source):
        # Normalise to same Frobenius norm per sample
        if frob_norm_source:
            if data_source == 'source':
                self._frob_norm_param = np.mean(np.linalg.norm(artificial_samples, axis=1))
            else:
                frob_norm = np.mean(np.linalg.norm(artificial_samples, axis=1))
                artificial_samples = artificial_samples * self._frob_norm_param / frob_norm
        else:
            pass
        
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


    def _approximate_encoders(
        self, 
        data_source:str, 
        artificial_samples, 
        artificial_embeddings
    ):
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

        self.sqrt_inv_M_X_ = mat_inv_sqrt(self.M_X)
        self.sqrt_inv_M_Y_ = mat_inv_sqrt(self.M_Y)
        self.sqrt_inv_matrices_ = {
            'source': self.sqrt_inv_M_X_,
            'target': self.sqrt_inv_M_Y_
        }
        self.cosine_sim = self.sqrt_inv_M_X_.dot(self.M_XY).dot(self.sqrt_inv_M_Y_)


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
        self.principal_vectors_coef_ = {
            x: self.untransformed_rotations_[x].T.dot(self.sqrt_inv_matrices_[x]).dot(self.approximate_krr_regressions_[x].sample_weights_.T.detach().numpy())
            for x in self.untransformed_rotations_
        }


    def save(
            self,
            folder: str = '.',
            with_krr: bool=True,
            with_model: bool=True
    ):
        if not os.path.exists(folder) and not os.path.isdir(folder):
            os.mkdir(folder)

        # Dump scVI models
        if with_model:
            for x in self.scvi_models:
                dump(
                    self.scvi_models[x],
                    open('%s/scvi_model_%s.pkl'%(folder, x), 'wb')
                )
                self.scvi_models[x].save(
                    '%s/scvi_model_%s'%(folder, x),
                    save_anndata=True
                )

        # Dump the KRR:
        if not with_krr:
            return True

        for x in self.approximate_krr_regressions_:
            self.approximate_krr_regressions_[x].save('%s/krr_approx_%s'%(folder, x))

        # Save params
        pd.DataFrame(self.krr_params).to_csv('%s/krr_params.csv'%(folder))
        dump(self.krr_params, open('%s/krr_params.pkl'%(folder), 'wb'))

        for param_t in ['model', 'plan', 'train']:
            df = pd.DataFrame([self.scvi_params[x][param_t] for x in ['source', 'target']])
            df.to_csv('%s/scvi_params_%s.csv'%(folder, param_t))
        dump(self.scvi_params, open('%s/scvi_params.pkl'%(folder), 'wb'))

        pd.DataFrame(self._fit_params, index=['params']).to_csv('%s/fit_params.csv'%(folder))
        dump(self._fit_params, open('%s/fit_params.pkl'%(folder), 'wb'))

        # Save results
        results_elements = {
            'alignment_M_X': self.M_X,
            'alignment_M_Y': self.M_Y,
            'alignment_M_XY': self.M_XY,
            'alignment_cosine_sim': self.cosine_sim,
            'alignment_principal_angles': self.principal_angles
        }
        for idx, element in results_elements.items():
            if type(element) is np.ndarray:
                np.savetxt('%s/%s.csv'%(folder, idx), element)
                np.save(open('%s/%s.npy'%(folder, idx), 'wb'), element)
            elif type(element) is torch.Tensor:
                np.savetxt('%s/%s.csv'%(folder, idx), element.detach().numpy())
                torch.save(element, open('%s/%s.pt'%(folder, idx), 'wb'))

        if self._frob_norm_param is not None:
            np.savetxt('%s/frob_norm_param.csv'%(folder), self._frob_norm_param)


    def load(
            folder: str = '.',
            with_krr: bool=True,
            with_model: bool=True
    ):
        clf = SobolevAlignment()

        if with_model:
            clf.scvi_models = {}
            for x in ['source', 'target']:
                clf.scvi_models[x] = scvi.model.SCVI.load(
                    '%s/scvi_model_%s'%(folder, x)
                )
        
        if with_krr:
            clf.approximate_krr_regressions_ = {}
            for x in ['source', 'target']:
                clf.approximate_krr_regressions_[x] = KRRApprox.load('%s/krr_approx_%s/'%(folder, x))

            # Load params
            clf.krr_params = load(open('%s/krr_params.pkl'%(folder), 'rb'))
            clf.scvi_params = load(open('%s/scvi_params.pkl'%(folder), 'rb'))
            if 'fit_params.pkl' in os.listdir(folder):
                clf._fit_params = load(open('%s/fit_params.pkl'%(folder), 'rb'))

            # Load results
            if 'alignment_M_X.npy' in os.listdir(folder):
                clf.M_X = np.load('%s/alignment_M_X.npy'%(folder))
            elif 'alignment_M_X.pt' in os.listdir(folder):
                clf.M_X = torch.load(open('%s/alignment_M_X.pt'%(folder), 'rb'))

            if 'alignment_M_Y.npy' in os.listdir(folder):
                clf.M_Y = np.load('%s/alignment_M_Y.npy'%(folder))
            elif 'alignment_M_Y.pt' in os.listdir(folder):
                clf.M_Y = torch.load(open('%s/alignment_M_Y.pt'%(folder), 'rb'))
            
            if 'alignment_M_XY.npy' in os.listdir(folder):
                clf.M_XY = np.load('%s/alignment_M_XY.npy'%(folder))
            elif 'alignment_M_XY.pt' in os.listdir(folder):
                clf.M_XY = torch.load(open('%s/alignment_M_XY.pt'%(folder), 'rb'))
            
            if 'alignment_cosine_sim.npy' in os.listdir(folder):
                clf.cosine_sim = np.load('%s/alignment_cosine_sim.npy'%(folder))
            elif 'alignment_cosine_sim.pt' in os.listdir(folder):
                clf.cosine_sim = torch.load(open('%s/alignment_cosine_sim.pt'%(folder), 'rb'))
            
            if 'alignment_principal_angles.npy' in os.listdir(folder):
                clf.principal_angles = np.load('%s/alignment_principal_angles.npy'%(folder))
            elif 'alignment_principal_angles.pt' in os.listdir(folder):
                clf.principal_angles = torch.load(open('%s/alignment_principal_angles.pt'%(folder), 'rb'))

            clf.sqrt_inv_M_X_ = mat_inv_sqrt(clf.M_X)
            clf.sqrt_inv_M_Y_ = mat_inv_sqrt(clf.M_Y)
            clf.sqrt_inv_matrices_ = {
                'source': clf.sqrt_inv_M_X_,
                'target': clf.sqrt_inv_M_Y_
            }
            clf._compute_principal_vectors()

        if 'frob_norm_param.csv' in os.listdir(folder):
            clf._frob_norm_param = np.loadtxt(open('%s/frob_norm_param.csv'%(folder), 'r'))

        return clf


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


    def compute_error(self, size=-1):
        """
        Compute error of the KRR approximation on the input (data used for VAE training) and used for KRR.
        :return:
        """
        return {
            'source': self._compute_error_one_type('source', size=size),
            'target': self._compute_error_one_type('target', size=size)
        }


    def _compute_error_one_type(self, data_type, size=-1):
        # KRR error of input data
        latent = self.scvi_models[data_type].get_latent_representation()
        if self._fit_params['lib_size_norm']:
            input_krr_pred = self.scvi_models[data_type].get_normalized_expression(
                return_numpy=True,
                library_size=DEFAULT_LIB_SIZE
            )
        else:
            input_krr_pred = self.training_data[data_type].X
        if self.krr_log_input_:
            input_krr_pred = np.log10(input_krr_pred+1)

        input_krr_pred = StandardScaler(with_mean=self.mean_center, with_std=self.unit_std).fit_transform(input_krr_pred)
        input_krr_pred =  self.approximate_krr_regressions_[data_type].transform(torch.Tensor(input_krr_pred))
        input_spearman_corr = np.array([scipy.stats.spearmanr(x,y)[0] for x,y in zip(input_krr_pred.T, latent.T)])
        input_krr_diff = input_krr_pred - latent
        input_mean_square = torch.square(input_krr_diff)
        input_factor_mean_square = torch.mean(input_mean_square, axis=0)
        input_latent_mean_square = torch.mean(input_mean_square)
        input_factor_reconstruction_error = np.linalg.norm(input_krr_diff, axis=0) / np.linalg.norm(latent, axis=0)
        input_latent_reconstruction_error = np.linalg.norm(input_krr_diff) / np.linalg.norm(latent)
        del input_krr_pred, input_mean_square, input_krr_diff
        gc.collect()

        # KRR error of artificial data
        if size > 1:
            subsamples = np.random.choice(np.arange(self.artificial_samples_[data_type].shape[0]), size, replace=False)
        elif size <= 0:
            return {
            'factor':{
                    'MSE': {
                        'input': input_factor_mean_square.detach().numpy()
                    },
                    'reconstruction_error': {
                        'input': input_factor_reconstruction_error
                    },
                    'spearmanr': {
                        'input': np.array(input_spearman_corr)
                    },
                },
                'latent':{
                    'MSE': {
                        'input': input_latent_mean_square.detach().numpy()
                    },
                    'reconstruction_error': {
                        'input': input_latent_reconstruction_error
                    },
                    'spearmanr': {
                        'input': np.mean(input_spearman_corr)
                    },
                }
            }
        else:
            subsamples = np.arange(self.artificial_samples_[data_type].shape[0])
        training_krr_diff = self.approximate_krr_regressions_[data_type].transform(torch.Tensor(self.artificial_samples_[data_type][subsamples]))
        training_spearman_corr = np.array([scipy.stats.spearmanr(x,y)[0] for x,y in zip(training_krr_diff.T, self.artificial_embeddings_[data_type][subsamples].T)])
        training_krr_diff = training_krr_diff - self.artificial_embeddings_[data_type][subsamples]
        training_krr_factor_reconstruction_error = np.linalg.norm(training_krr_diff, axis=0) / np.linalg.norm(self.artificial_embeddings_[data_type][subsamples], axis=0)
        training_krr_latent_reconstruction_error = np.linalg.norm(training_krr_diff) / np.linalg.norm(self.artificial_embeddings_[data_type][subsamples])

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
                'spearmanr': {
                    'input': np.array(input_spearman_corr),
                    'artificial': np.array(training_spearman_corr)
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
                'spearmanr': {
                    'input': np.mean(input_spearman_corr),
                    'artificial': np.mean(training_spearman_corr)
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

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.factor_level_feature_weights_df = {}
        for x in self.training_data:
            basis_feature_weights_df = higher_order_contribution(
                d=max_order,
                data=self.approximate_krr_regressions_[x].anchors().cpu().detach().numpy(),
                sample_offset=self.sample_offset[x],
                gene_names=self.gene_names,
                gamma=self.gamma,
                n_jobs=self.n_jobs
            )
            index = np.arange(self.approximate_krr_regressions_[x].sample_weights_.T.shape[0])
            columns = basis_feature_weights_df.columns
            values = self.approximate_krr_regressions_[x].sample_weights_.T.to(device)
            values = values.matmul(torch.Tensor(basis_feature_weights_df.values).to(device))
            self.factor_level_feature_weights_df[x] = pd.DataFrame(
                values.cpu().detach().numpy(), index=index, columns=columns
            )
            del basis_feature_weights_df
            gc.collect()

        self.pv_level_feature_weights_df = {
            x: pd.DataFrame(
                self.untransformed_rotations_[x].T.dot(self.sqrt_inv_matrices_[x]).dot(self.factor_level_feature_weights_df[x]),
                index=['PV %s'%(i) for i in range(self.untransformed_rotations_[x].shape[1])],
                columns=self.factor_level_feature_weights_df[x].columns
            )
            for x in self.training_data
        }


    def compute_gradients_factors(self, n_samples=10**4):
        gradient_expectation_ = {}
        for data_type in self.krr_params:
            gradient_expectation_[data_type] = self._gradient_expectation(data_type, n_samples)
        return gradient_expectation_


    def _gradient_expectation(self, data_type, n_samples, cuda_device=None):
        # Generate some samples
        artificial_samples_, artificial_batches_, artificial_covariates_ = self._generate_artificial_samples(
            data_source=data_type,
            n_artificial_samples=n_samples,
            large_batch_size=10**5,
            save_mmap=None
        )

        if cuda_device is None:
            device = "cuda" if torch.cuda.is_available() else 'cpu'
        else:
            device = cuda_device
        artificial_samples_ = torch.Tensor(artificial_samples_).to(device)
        if artificial_covariates_ is not None:
            artificial_covariates_ = torch.Tensor(artificial_covariates_.values.astype(float)).to(device)
        if artificial_batches_ is not None:
            artificial_batches_ = np.array([
                np.where(self.scvi_models[data_type].scvi_setup_dict_['categorical_mappings']['_scvi_batch']['mapping'] == str(n))[0][0]
                for n in artificial_batches_
            ])
            artificial_batches_ = torch.Tensor(artificial_batches_.astype(int)).reshape(-1,1).to(device)

        # Compute embedding
        vae_input = {
            'x': artificial_samples_,
            'batch_index': artificial_batches_ if artificial_batches_ is not None else None,
            'cont_covs': artificial_covariates_.values if artificial_covariates_ is not None else None,
            'cat_covs': None
        }
        artificial_samples_.requires_grad = True
        module = deepcopy(self.scvi_models[data_type].module).to(device)
        embedding_values = module.inference(**vae_input)['qz_m']

        # Compute the gradients element by element
        gradients = []
        for pv_idx in range(embedding_values.shape[1]):
            for samples_idx in range(artificial_samples_.shape[0]):
                b = embedding_values[samples_idx,pv_idx]
                b.retain_grad()
                b.backward(retain_graph=True)
            gradients.append(torch.mean(artificial_samples_.grad, axis=0))
            artificial_samples_.grad = None
        
        return torch.stack(gradients)




