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

from .saab import Saab
from .sparsepattern import sparse_sample


def window_process2(samples, kernel_size, stride, pattern=None):
    n, h, w, c = samples.shape
    output_h = (h - kernel_size) // stride + 1
    output_w = (w - kernel_size) // stride + 1
    patches = view_as_windows(samples, (1, kernel_size, kernel_size, c), step=(1, stride, stride, c))
    patches = patches.reshape(n, output_h, output_w, kernel_size, kernel_size, c)
    patches = np.moveaxis(patches, -1, -3)
    if pattern is not None:
        patches = sparse_sample(patches, pattern)
    else:
        patches = patches.reshape(n, output_h, output_w, c * kernel_size * kernel_size)
    return patches


def Shrink_(X, shrinkArg):
    pad = shrinkArg['pad']
    ksize = shrinkArg['size']
    if pad == 'reflect':
        feature = np.pad(X.copy(), ((0, 0), (ksize // 2, ksize // 2), (ksize // 2, ksize // 2), (0, 0)), 'reflect')
    elif pad == 'zeros':
        feature = np.pad(X.copy(), ((0, 0), (ksize // 2, ksize // 2), (ksize // 2, ksize // 2), (0, 0)), 'constant',
                         constant_values=0)
    else:
        feature = X.copy()
    if 'pattern' in shrinkArg:
        return window_process2(feature, ksize, shrinkArg['stride'], shrinkArg['pattern'])
    else:
        return window_process2(feature, ksize, shrinkArg['stride'])


def Output_Concat(X):
    return np.concatenate(X, axis=-1)


def Transform(X, par, train, shrinkArg, SaabArg):
    X = Shrink_(X, shrinkArg=shrinkArg)
    S = X.shape
    X = X.reshape(-1, S[-1])
    transformed, par = Saab(None, num_kernels=SaabArg['num_AC_kernels'], useDC=SaabArg['useDC'], batch=SaabArg['batch'],
                            needBias=SaabArg['needBias']).Saab_transform(X, train=train, pca_params=par)
    S_new = (S[0], S[1], S[2], SaabArg['num_AC_kernels'])
    transformed = transformed.reshape(S_new)
    return par, transformed


def cwSaab_1_layer(X, train, par_cur, SaabArg, shrinkArg):
    par, transformed = Transform(X, par=par_cur, train=train, shrinkArg=shrinkArg, SaabArg=SaabArg)
    return transformed, [par], par['Energy']


def cwSaab_1_layer_(X, train, par_cur, SaabArg, shrinkArg):
    transformed, eng_cur = [], []
    # channel wise saab
    S = list(X.shape)
    S[-1] = 1  # [N, spatial, 1]
    # move all channels to dim0
    X = np.moveaxis(X, -1, 0)
    for i in range(X.shape[0]):  # for each spectral channel

        X_tmp = X[i].reshape(S)  # reshape to [N, spatial, 1]
        if train == True:
            par_tmp, tmp_transformed = Transform(X_tmp, par=None, train=train, shrinkArg=shrinkArg, SaabArg=SaabArg)
            par_cur.append(par_tmp)
            eng_cur.append(par_tmp['raw_energy'])
        else:
            if len(par_cur) == i:
                break
            _, tmp_transformed = Transform(X_tmp, par=par_cur[i], train=train, shrinkArg=shrinkArg, SaabArg=SaabArg)
        transformed.append(tmp_transformed)
    transformed = np.concatenate(transformed, axis=-1)

    if train == True:
        eng_cur = np.concatenate(eng_cur, axis=0)
        eng_cur = eng_cur / np.sum(eng_cur)
        print(eng_cur)

    return transformed, par_cur, eng_cur


def cwSaab_n_layer(X, energyTH, train, par_prev, par_cur, SaabArg, shrinkArg):
    output, eng_cur = [], []
    S = list(X.shape)
    S[-1] = 1
    X = np.moveaxis(X, -1, 0)
    ct, split = -1, False
    if train == True:
        par_cur = []
    else:
        pidx = 0
    for i in range(len(par_prev)):
        for j in range(par_prev[i]['Energy'].shape[0]):
            ct += 1
            if par_prev[i]['Energy'][j] < energyTH:
                continue
            X_tmp = X[ct].reshape(S)
            split = True
            if train == True:
                par_tmp, out_tmp = Transform(X_tmp, par=None, train=train, shrinkArg=shrinkArg, SaabArg=SaabArg)
                par_tmp['Energy'] *= par_prev[i]['Energy'][j]
                eng_cur.append(par_tmp['Energy'])
                par_cur.append(par_tmp)
                output.append(out_tmp)
            else:
                par_tmp, out_tmp = Transform(X_tmp, par=par_cur[pidx], train=train, shrinkArg=shrinkArg,
                                             SaabArg=SaabArg)
                output.append(out_tmp)
                eng_cur.append(par_cur[pidx]['Energy'])
                pidx += 1
    if split == True:
        output = np.concatenate(output, axis=-1)
        eng_cur = np.concatenate(eng_cur, axis=0)
    return output, par_cur, eng_cur, split


def cwSaab(X, train=True, par=None, depth=None, energyTH=None, SaabArgs=None, shrinkArgs=None, concatArgs=None,
           poolArgs=None, verbose=False):
    output, eng = [], []
    if train == True:
        par = {'depth': depth, 'energyTH': energyTH, 'SaabArgs': SaabArgs, 'shrinkArgs': shrinkArgs,
               'concatArgs': concatArgs, 'poolArgs': poolArgs}
        X, par_tmp, eng_tmp = cwSaab_1_layer(X, train=train, par_cur=[], SaabArg=SaabArgs[0], shrinkArg=shrinkArgs[0])
        output.append(X)
        ## add maxpooling
        if (poolArgs is not None and poolArgs[0]['maxpool']):
            X = block_reduce(X, (1, 2, 2, 1), np.max)

        kept_index_all = list()
        kept_index_all.append(eng_tmp >= energyTH[0])
        if verbose:
            print(len(eng_tmp), np.sum(eng_tmp >= energyTH[0]))
        eng.append(eng_tmp)
        par['Layer0'] = par_tmp

        for i in range(1, depth):
            X, par_tmp, eng_tmp, split = cwSaab_n_layer(X, energyTH=energyTH[i - 1], train=train, par_prev=par_tmp,
                                                        par_cur=[], SaabArg=SaabArgs[i], shrinkArg=shrinkArgs[i])

            if split == False:
                par['depth'], depth = i, i
                print("       <WARNING> Cannot futher split, actual depth: %s" % str(i))
                break

            output.append(X)

            ## add maxpooling
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
        X, par_tmp, eng_tmp = cwSaab_1_layer(X, train=train, par_cur=par['Layer0'][0], SaabArg=SaabArgs[0],
                                             shrinkArg=shrinkArgs[0])
        output.append(X)
        ## add maxpooling
        if (poolArgs is not None and poolArgs[0]['maxpool']):
            X = block_reduce(X, (1, 2, 2, 1), np.max)

        eng.append(eng_tmp)
        for i in range(1, depth):
            X, par_tmp, eng_tmp, split = cwSaab_n_layer(X, energyTH=energyTH[i - 1], train=train,
                                                        par_prev=par['Layer' + str(i - 1)],
                                                        par_cur=par['Layer' + str(i)], SaabArg=SaabArgs[i],
                                                        shrinkArg=shrinkArgs[i])
            output.append(X)
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
