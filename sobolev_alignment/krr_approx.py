""" Encoder approximation by Kernel Ridge Regression

@author: Soufiane Mourragui

This modules train a Kernel Ridge Regression (KRR) on a pair
of sampled (x_hat) and embedding (z_hat) using two possible
versions:
    - scikit-learn implementation : deterministic, but limited in
    memory and time efficiency.
    - Falkon implementation : stochastic Nyström approximation, but
    faster both memory and time-wise.


Notes
-------
	-
	
References
-------
"""

import os, sys
import numpy as np
from pickle import load, dump
import torch

# Falkon import
from falkon import Falkon, kernels
from falkon.options import FalkonOptions
from falkon.kernels import GaussianKernel, MaternKernel, LaplacianKernel

# Scikit-learn import
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.gaussian_process.kernels import PairwiseKernel
from sklearn.kernel_ridge import KernelRidge


class KRRApprox:
    """
    Kernel Ridge Regression approximation.

    """

    sklearn_kernel = {
        'rbf': 'wrapper',
        'laplacian': 'wrapper',
        'matern': Matern,
    }
    falkon_kernel = {
        'rbf': GaussianKernel,
        'laplacian': LaplacianKernel,
        'matern': MaternKernel,
    }
    default_kernel_params = {
        'falkon': { 'rbf': {'sigma': 1}, 'laplacian': {'sigma': 1}, 'matern': {'sigma': 1, 'nu': .5}},
        'sklearn': { 'rbf': {}, 'laplacian': {}, 'matern': {}}
    }

    def __init__(
            self,
            method: str = 'sklearn',
            kernel: str = 'rbf',
            M: int = 100,
            kernel_params: dict = None,
            penalization: float = 10e-6,
            use_cpu: bool = None
    ):
        """
        Create a KRRApprox instance that aims at creating one KRR model per
        embedding based on artificial data.

        Parameters
        ----------
        method
            Method used for KRR approximation, either 'sklearn' (for scikit-learn)
            implementation, or 'falkon' (for FALKON).
        kernel
            Name of the kernel to use in the approximation. Can be choosen between
            'rbf' (Gaussian kernel), 'matern' (Matern kernel), 'laplace' (Laplace).
        M
            Number of anchors samples for the Nystrom approximation. Only when
            method = 'falkon'. Default to 100 ; can be set automatically, and prior
            to fitting.
        kernel_params
            Dictionary containing the kernel hyper-parameters.
        penalization
            Amount of penalization. The higher, the more penalization.
            Corresponds to alpha in sklearn.kernel_ridge.KernelRidge.
        use_cpu
            Whether CPU should be used. By default set to None. Only relevant for FALKON.
        """


        self.method = method

        # Set kernel
        self.kernel = kernel
        self.kernel_params = kernel_params if kernel_params is not None else self.default_kernel_params[self.method][self.kernel]
        self._make_kernel()

        # Set penalization parameters
        self.penalization = penalization
        self.M = M

        # Set hardware specifications
        self.use_cpu = use_cpu


    def _make_kernel(self):
        """
        Create kernel depending on parameters given.

        :return:
            Return True if kernel has been properly set up.
        """

        # scikit-learn initialization
        if self.method.lower() == 'sklearn':
            if self.sklearn_kernel[self.kernel.lower()] != 'wrapper':
                self.kernel_ = self.sklearn_kernel[self.kernel.lower()](**self.kernel_params)
            else:
                self.kernel_ = PairwiseKernel(metric=self.kernel.lower(), **self.kernel_params)

        # Falkon
        elif self.method.lower() == 'falkon':
            self.kernel_ = self.falkon_kernel[self.kernel.lower()](**self.kernel_params)

        # If not implemented
        else:
            raise NotImplementedError('%s not implemented. Choices: sklearn and falkon'%(self.method))

        return True

    def fit(
            self,
            X: torch.Tensor,
            y: torch.Tensor):
        """
        Approximate by a KRR the relationship between X and Y.

        Parameters
        ----------
        X
            Tensor containing the artificial input (x_hat).
        y
            Tensor containing the artificial embedding (z_hat). Called y for compliance with
            sklearn functions.
        """
        self._setup_clf()
        self.training_data_ = X

        if self.method == 'sklearn':
            self.ridge_clf_.fit(self.kernel_(self.training_data_), y)
        elif self.method == 'falkon':
            self.ridge_clf_.fit(self.training_data_, y)

        self._save_coefs()

        return self

    def _setup_clf(self):
        if self.method.lower() == 'sklearn':
            self._setup_sklearn_clf()
        elif self.method.lower() == 'falkon':
            self._setup_falkon_clf()


    def _setup_sklearn_clf(self):
        self.ridge_clf_ = KernelRidge(kernel='precomputed',alpha=self.penalization)
        return True

    def _setup_falkon_clf(self):
        self.ridge_clf_ = Falkon(
            kernel=self.kernel_,
            penalty=self.penalization,
            M=self.M,
            options=FalkonOptions(use_cpu=self.use_cpu)
        )
        return True

    def _save_coefs(self):
        if self.method.lower() == 'sklearn':
            self._process_coef_ridge_sklearn()
        elif self.method.lower() == 'falkon':
            self._process_coef_ridge_falkon()

    def _process_coef_ridge_sklearn(self):
        self.sample_weights_ = torch.Tensor(self.ridge_clf_.dual_coef_)
        self.ridge_samples_idx_ = np.arange(self.training_data_.shape[0])

    def _process_coef_ridge_falkon(self):
        self.sample_weights_ = self.ridge_clf_.alpha_

        # Finds training_idxs by matching product over rows
        mask = - torch.min(self.training_data_)
        mask = np.random.uniform(mask, mask+0.2, self.training_data_.shape[1])
        self._train_product = np.sum(np.log(self.training_data_.detach().numpy() + mask), axis=1)
        self._ny_product = np.sum(np.log(self.ridge_clf_.ny_points_.detach().numpy() + mask), axis=1)
        self.ridge_samples_idx_ = [np.where(self._train_product == x)[0] for x in self._ny_product]
        for x in self.ridge_samples_idx_:
            if x.shape[0] > 1:
                assert False
        self.ridge_samples_idx_ = [x[0] for x in self.ridge_samples_idx_]
        assert len(self.ridge_samples_idx_) == self.M

    def transform(
            self,
            X: torch.Tensor
    ):
        """
        Predict approximations ; out-of-sample extension

        Parameters
        ----------
        X
            Tensor containing the artificial input (x_hat).
        """

        if self.method == 'sklearn':
            return self.ridge_clf_.predict(self.kernel_(X, self.training_data_))
        elif self.method == 'falkon':
            return self.ridge_clf_.predict(X)
        else:
            raise NotImplementedError('%s not implemented. Choices: sklearn and falkon'%(self.method))

    def save(
            self,
            folder:str = '.'
    ):
        if not os.path.exists(folder) and not os.path.isdir(folder):
            os.mkdir(folder)

        # Save params
        params = {
            'method': self.method,
            'kernel': self.kernel_,
            'M': self.M,
            'penalization': self.penalization,
            'use_cpu': self.use_cpu
        }
        params.update(self.kernel_params)
        dump(params, open('%s/params.pkl'%(folder), 'wb'))

        # Save important material:
        #   - KRR weights
        #   - Samples used for prediction.
        dump(self.sample_weights_, open('%s/sample_weights.pkl'%(folder), 'wb'))
        dump(
            torch.Tensor(self.training_data_[self.ridge_samples_idx_]),
            open('%s/sample_weights.pkl'%(folder), 'wb')
        )



