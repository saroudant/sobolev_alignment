
import numpy as np
import torch

# Falkon import
from falkon import Falkon, kernels
from falkon.kernels import GaussianKernel, MaternKernel, LaplacianKernel


class MultiKRRApprox:
    def __init__(self):
        self.krr_regressors = []

    def predict(
            self,
            X: torch.Tensor
    ):
        prediction = [
            clf.transform(torch.Tensor(X)).detach().numpy()
            for clf in self.krr_regressors
        ]
        prediction = torch.Tensor(prediction)
        prediction = torch.mean(prediction, axis=0)

        return prediction

    def transform(
            self,
            X: torch.Tensor
    ):
        return self.predict(X)

    def anchors(self):
        return self.anchors

    def process_clfs(self):
        self.anchors = torch.cat([clf.anchors() for clf in self.krr_regressors])
        self.sample_weights_ = torch.cat([clf.sample_weights_ for clf in self.krr_regressors])
        self.sample_weights_ = 1 / len(self.krr_regressors) * self.sample_weights_

    def add_clf(self, clf):
        self.krr_regressors.append(clf)

