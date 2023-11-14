# v 2022.02.09
# one stage Saab transformation
# modified from https://github.com/ChengyaoWang/PixelHop-_c-wSaab/blob/master/saab.py

import numpy as np
import numba
from sklearn.decomposition import PCA, IncrementalPCA


@numba.jit(nopython = True, parallel = False)
def pca_cal(X: np.ndarray):
    cov = np.cov(X,rowvar=False)#X.transpose() @ X
    eva, eve = np.linalg.eigh(cov)
    inds = eva.argsort()[::-1]
    eva = eva[inds]
    kernels = eve.transpose()[inds]
    return kernels, eva / (X.shape[0] - 1)

# @numba.jit(nopython = True, parallel = True)
def pca_cov(X: np.ndarray):
    return np.cov(X,rowvar=False)

@numba.jit(forceobj = True, parallel = False)
def remove_mean(X: np.ndarray, feature_mean: np.ndarray):
    return X - feature_mean

@numba.jit(nopython = True, parallel = False)
def feat_transform(X: np.ndarray, kernel: np.ndarray):
    return X @ kernel.transpose()


class SimpSaab():
    def __init__(self, num_kernels=-1):
        self.num_kernels = num_kernels
        self.Kernels = []
        self.Mean0 = [] # feature mean of AC
        self.Energy = [] # kernel energy list
        self.trained = False

    def remove_constant_patch(self, X, thrs=1e-5):
        diff = np.sum(X, axis = 1)
        idx = np.argwhere(diff<thrs).squeeze()
        # X = np.delete(X, idx, axis=0)
        return X[diff>thrs]

    def fit(self, X): 
        assert (len(X.shape) == 2), "Input must be a 2D array!"
        X = X.astype('float32')

        # remove DC, get AC components
        dc = np.mean(X, axis = 1, keepdims = True)
        X = remove_mean(X, dc)

        # remove feature mean --> self.Mean0
        self.Mean0 = np.mean(X, axis = 0, keepdims = True)
        X = remove_mean(X, self.Mean0)

        if self.num_kernels == -1:
            self.num_kernels = X.shape[-1]
        
        # Rewritten PCA Using Numpy
        # kernels, eva, cov = pca_cal(X)
        pca = PCA(n_components=self.num_kernels,svd_solver='full').fit(X)
        #pca = PCA(self.num_kernels).fit(X)
        kernels = pca.components_
        eva = pca.explained_variance_
        self.cov = pca_cov(X)

        # Concatenate with DC kernel
        dc_kernel = 1 / np.sqrt(X.shape[-1]) * np.ones((1, X.shape[-1]))# / np.sqrt(largest_ev)
        kernels = np.concatenate((dc_kernel, kernels[:-1]), axis = 0)
        
        # Concatenate with DC energy
        largest_ev = np.var(dc * np.sqrt(X.shape[-1]))  
        energy = np.concatenate((np.array([largest_ev]), eva[:-1]), axis = 0)
        energy = energy / np.sum(energy)
        
        # store
        self.Kernels, self.Energy = kernels.astype('float32'), energy
        self.trained = True


    def transform(self, X):
        assert (self.trained == True), "Must call fit first!"
        X = X.astype('float32')

        # remove feature mean of AC
        X = remove_mean(X, self.Mean0)
        
        # convolve with DC and AC filters
        X = feat_transform(X, self.Kernels)
        
        return X
    
 
