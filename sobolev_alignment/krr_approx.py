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
import gc
from sklearn.preprocessing import StandardScaler

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
        'gaussian': 'wrapper',
        'laplacian': 'wrapper',
        'matern': Matern,
    }
    falkon_kernel = {
        'rbf': GaussianKernel,
        'gaussian': GaussianKernel,
        'laplacian': LaplacianKernel,
        'matern': MaternKernel,
    }
    default_kernel_params = {
        'falkon': { 'rbf': {'sigma': 1}, 'gaussian': {'sigma': 1}, 'laplacian': {'sigma': 1}, 'matern': {'sigma': 1, 'nu': .5}},
        'sklearn': { 'rbf': {}, 'gaussian': {}, 'laplacian': {}, 'matern': {}}
    }

    def __init__(
            self,
            method: str = 'sklearn',
            kernel: str = 'rbf',
            M: int = 100,
            kernel_params: dict = None,
            penalization: float = 10e-6,
            maxiter: int=20,
            falkon_options: dict = {},
            mean_center: bool=False,
            unit_std: bool=False
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
        """


        self.method = method

        # Set kernel
        self.kernel = kernel
        self.kernel_params = kernel_params if kernel_params is not None else self.default_kernel_params[self.method][self.kernel]
        self._make_kernel()

        # Set penalization parameters
        self.penalization = penalization
        self.M = M
        self.maxiter = maxiter

        # Set hardware specifications
        self.falkon_options = falkon_options

        # Preprocessing
        self.mean_center = mean_center
        self.unit_std = unit_std
        # self.pre_process_ = StandardScaler(with_mean=mean_center, with_std=unit_std, copy=False)


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
        # self.pre_process_.fit(X)
        # self.training_data_ = torch.Tensor(self.pre_process_.transform(torch.Tensor(X)))
        self.training_data_ = X

        if self.method == 'sklearn':
            self.ridge_clf_.fit(self.kernel_(self.training_data_), y)
        elif self.method == 'falkon':
            print(self.training_data_)
            self.ridge_clf_.fit(self.training_data_, y)

        self._save_coefs()

        if self.method == 'falkon':
            self.training_data_ = self.ridge_clf_.ny_points_
        gc.collect()

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
            maxiter=self.maxiter,
            options=FalkonOptions(**self.falkon_options)
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

    def anchors(self):
        return self.training_data_

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

        # X_t = torch.Tensor(self.pre_process_.transform(X))

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
            'mean_center': self.mean_center,
            'unit_std': self.unit_std
        }
        params.update(self.kernel_params)
        dump(params, open('%s/params.pkl'%(folder), 'wb'))

        # Save important material:
        #   - KRR weights
        #   - Samples used for prediction.
        torch.save(
            torch.Tensor(self.anchors()), 
            open('%s/sample_anchors.pt'%(folder), 'wb')
        )
        torch.save(
            torch.Tensor(self.sample_weights_), 
            open('%s/sample_weights.pt'%(folder), 'wb')
        )
        
        np.savetxt('%s/sample_weights.csv'%(folder), self.sample_weights_.detach().numpy())
        np.savetxt('%s/sample_anchors.csv'%(folder), self.anchors().detach().numpy())


    def load(folder:str = '.'):
        params = load(open('%s/params.pkl'%(folder), 'rb'))
        krr_params = {e:f for e,f in params.items() if e in ['method', 'M', 'penalization', 'mean_center', 'unit_std']}
        # krr_params['kernel'] = krr_params['kernel'].kernel_name
        krr_approx_clf = KRRApprox(**krr_params)
        krr_approx_clf.kernel_ = params['kernel']

        krr_approx_clf.sample_weights_ = torch.load(open('%s/sample_weights.pt'%(folder), 'rb'))
        krr_approx_clf.training_data_ = torch.load(open('%s/sample_anchors.pt'%(folder), 'rb'))

        krr_approx_clf._setup_clf()
        krr_approx_clf.ridge_clf_.ny_points_ = krr_approx_clf.training_data_
        krr_approx_clf.ridge_clf_.alpha_ = krr_approx_clf.sample_weights_

        return krr_approx_clf



