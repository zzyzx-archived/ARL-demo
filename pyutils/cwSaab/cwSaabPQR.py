# v2020.02.06 avoid depth overflow
# a generialzed version of channel wise PCA
#
# Shrink: to support different type of data (image, pointcloud)
# Output_Concat: how to concatenate features from different hop, especially when spatial shape is different
# 
# train: <bool> 
# par: <dict>, parameters
# depth: <int>, depth of tree
# energtTH: <float>, energy threshold for stopping spliting on nodes
# SaabArgs: <list>, ex: [{'needBias':False, 'useDC':True, 'batch':None}]
# shrinkArgs: <list>, ex: [{'dilate':[1], 'pad':'reflect'}]
# concatArgs: <list>, currently not used, left for future
#
# during testing, settings like depth, SaabArgs, shrinkArgs will be loaded from par

import numpy as np
from skimage.util.shape import view_as_windows
from skimage.measure import block_reduce
from sklearn.decomposition import PCA

from .saab import Saab
from .sparsepattern import sparse_sample


def window_process2(samples, kernel_size, stride, pattern=None, pqr=False):
    n, h, w, c = samples.shape
    output_h = (h - kernel_size) // stride + 1
    output_w = (w - kernel_size) // stride + 1
    patches = view_as_windows(samples, (1, kernel_size, kernel_size, c), step=(1, stride, stride, c))
    patches = patches.reshape(n, output_h, output_w, kernel_size, kernel_size, c)
    patches = np.moveaxis(patches, -1, -3)
    if pattern is not None:
        patches = sparse_sample(patches, pattern)
    else:
        if pqr:
            patches = patches.reshape(n, output_h, output_w, kernel_size * kernel_size, c)
        else:
            patches = patches.reshape(n, output_h, output_w, c * kernel_size * kernel_size)
    return patches


def Shrink_(X, shrinkArg, pqr=False):
    pad = shrinkArg['pad']
    ksize = shrinkArg['size']
    if pad == 'reflect':
        feature = np.pad(X.copy(), ((0, 0), (ksize // 2, ksize // 2), (ksize // 2, ksize // 2), (0, 0)), 'reflect')
    elif pad == 'zeros':
        feature = np.pad(X.copy(), ((0, 0), (ksize // 2, ksize // 2), (ksize // 2, ksize // 2), (0, 0)), 'constant',
                         constant_values=0)
    else:
        feature = X.copy()
    if 'pattern' in shrinkArg and shrinkArg['pattern'] is not None:
        return window_process2(feature, ksize, shrinkArg['stride'], shrinkArg['pattern'])
    else:
        return window_process2(feature, ksize, shrinkArg['stride'], pqr=pqr)


def Output_Concat(X):
    return np.concatenate(X, axis=-1)


def Transform(X, par, train, shrinkArg, SaabArg, pqr=False):
    S = X.shape
    X = X.reshape(-1, S[-1])
    transformed, par = Saab(None, num_kernels=SaabArg['num_AC_kernels'], useDC=SaabArg['useDC'], batch=SaabArg['batch'],
                            needBias=SaabArg['needBias']).Saab_transform(X, train=train, pca_params=par)
    S_new = (S[0], S[1], S[2], SaabArg['num_AC_kernels'])
    transformed = transformed.reshape(S_new)
    return par, transformed


def rgb2pqr(X, train, model):
    # X shape [N,h,w,neighborhood,3]
    N, h, w, n, _ = X.shape
    means = np.mean(X, axis=-2, keepdims=True)
    X -= means
    if train:
        pca = PCA(n_components=3).fit(X.reshape(-1, 3))
        model = dict()
        model['pca'] = pca
    else:
        pca = model['pca']
    pqr = pca.transform(X.reshape(-1, 3))
    pqr = pqr.reshape(X.shape)
    return model, pqr, means.reshape(N, h, w, 3)


def cwSaab_pqr_layer(X, energyTH, train, par_cur, SaabArg, shrinkArg):
    X = Shrink_(X, shrinkArg=shrinkArg, pqr=True)
    if train == True:
        par_cur = {'par_pqr': None, 'par_saab': list()}
    par_cur['par_pqr'], X, DC = rgb2pqr(X, train, par_cur['par_pqr'])

    output, eng_cur = [], []
    X = np.moveaxis(X, -1, 0)
    ct, split = -1, False
    if not train:
        pidx = 0
    for i in range(X.shape[0]):
        X_tmp = X[ct]
        split = True
        if train == True:
            par_tmp, out_tmp = Transform(X_tmp, par=None, train=train, shrinkArg=shrinkArg, SaabArg=SaabArg)
            par_tmp['Energy'] *= par_cur['par_pqr']['pca'].explained_variance_ratio_[i]
            eng_cur.append(par_tmp['Energy'])
            par_cur['par_saab'].append(par_tmp)
            output.append(out_tmp)
        else:
            par_tmp, out_tmp = Transform(X_tmp, par=par_cur['par_saab'][pidx], train=train, shrinkArg=shrinkArg,
                                         SaabArg=SaabArg)
            output.append(out_tmp)
            eng_cur.append(par_cur['par_saab'][pidx]['Energy'])
            pidx += 1
    if split == True:
        output = np.concatenate(output, axis=-1)
        eng_cur = np.concatenate(eng_cur, axis=0)
    return DC, output, par_cur, eng_cur, split


def cwSaab(X, train=True, par=None, depth=None, energyTH=None, SaabArgs=None, shrinkArgs=None, concatArgs=None,
           poolArgs=None, verbose=False):
    X = X.astype(float)
    output, eng = [], []
    if train == True:
        par = {'depth': depth, 'energyTH': energyTH, 'SaabArgs': SaabArgs, 'shrinkArgs': shrinkArgs,
               'concatArgs': concatArgs, 'poolArgs': poolArgs}
        kept_index_all = list()
        for i in range(0, depth):
            X, AC, par_tmp, eng_tmp, split = cwSaab_pqr_layer(X, energyTH=energyTH[i], train=train, par_cur=[],
                                                              SaabArg=SaabArgs[i], shrinkArg=shrinkArgs[i])

            if split == False:
                par['depth'], depth = i, i
                print("       <WARNING> Cannot futher split, actual depth: %s" % str(i))
                break
            feat = np.concatenate((X, AC[..., eng_tmp >= energyTH[i]]), axis=-1)
            output.append(feat)

            # add maxpooling
            if (poolArgs is not None and poolArgs[i]['maxpool']):
                X = block_reduce(X, (1, 2, 2, 1), np.max)

            eng.append(eng_tmp)
            par['Layer' + str(i)] = par_tmp
            if verbose:
                print(len(eng_tmp), np.sum(eng_tmp >= energyTH[i]))
            kept_index_all.append(eng_tmp >= energyTH[i])
        par['kept_index_all'] = kept_index_all
    else:
        depth, energyTH, shrinkArgs, SaabArgs, concatArgs, poolArgs = par['depth'], par['energyTH'], par['shrinkArgs'], \
        par['SaabArgs'], par['concatArgs'], par['poolArgs']

        for i in range(0, depth):
            X, AC, par_tmp, eng_tmp, split = cwSaab_pqr_layer(X, energyTH=energyTH[i], train=train,
                                                              par_cur=par['Layer' + str(i)], SaabArg=SaabArgs[i],
                                                              shrinkArg=shrinkArgs[i])
            feat = np.concatenate((X, AC[..., par['kept_index_all'][i]]), axis=-1)
            output.append(feat)
            ## add maxpooling
            if (poolArgs is not None and poolArgs[i]['maxpool']):
                X = block_reduce(X, (1, 2, 2, 1), np.max)

            eng.append(eng_tmp)

    return output, par


if __name__ == "__main__":
    X = np.random.rand(100, 32, 32, 3)
    print("Input shape: ", X.shape)
    SaabArgs = [{'num_AC_kernels': 6, 'needBias': False, 'useDC': True, 'batch': None},
                {'num_AC_kernels': 16, 'needBias': True, 'useDC': True, 'batch': None},
                {'num_AC_kernels': -1, 'needBias': True, 'useDC': True, 'batch': None},
                {'num_AC_kernels': -1, 'needBias': True, 'useDC': True, 'batch': None},
                {'num_AC_kernels': -1, 'needBias': True, 'useDC': True, 'batch': None},
                {'num_AC_kernels': -1, 'needBias': True, 'useDC': True, 'batch': None}]
    shrinkArgs_ = [{'size': 5, 'stride': 1, 'pad': 'zeros'},
                   {'size': 5, 'stride': 1, 'pad': 'zeros'},
                   {'size': 5, 'stride': 1, 'pad': 'zeros'},
                   {'size': 1, 'stride': 1, 'pad': 'zeros'},
                   {'size': 1, 'stride': 1, 'pad': 'zeros'},
                   {'size': 1, 'stride': 1, 'pad': 'zeros'}]
    poolArgs = [{'maxpool': True},
                {'maxpool': True},
                {'maxpool': False},
                {'maxpool': False},
                {'maxpool': False},
                {'maxpool': False}]
    output, par = cwSaab(X, train=True, par=None, depth=len(SaabArgs), energyTH=0.5,
                         SaabArgs=SaabArgs, shrinkArgs=shrinkArgs_, concatArgs=None, poolArgs=poolArgs)
