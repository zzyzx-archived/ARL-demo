import numpy as np
import cv2
from skimage.measure import block_reduce

#from cwSaab.cwSaab import cwSaab, window_process2
from .cwSaab.cwSaabPQR import cwSaab, window_process2
from .saab_dft.lib_global_saab import get_joint_gs_feat

def get_par(train_set,numKernel,sizeKernel,stride,maxpool=False,pattern=None,energyTh=None,num_layer=1):
    patch_size = 32
    train_set = np.expand_dims(train_set, axis=0)
    train_set = window_process2(train_set, kernel_size=patch_size, stride=patch_size//5)
    tmp = train_set.shape
    train_set = train_set.reshape((tmp[0]*tmp[1]*tmp[2], patch_size, patch_size, -1))

    SaabArgs = [{'num_AC_kernels': numKernel, 'needBias':False, 'useDC':True, 'batch':None}]
    shrinkArgs_ = [{'size':sizeKernel, 'stride':stride, 'pad':'reflect', 'pattern':pattern}]
    poolArgs = [{'maxpool':maxpool}]
    if energyTh is None:
        energyTh = [1e-2,1e-3,1e-3,1e-3]

    out, parSaab = cwSaab(train_set, train=True, par=None, depth=len(SaabArgs[:num_layer]), energyTH=energyTh[:num_layer],  
                      SaabArgs=SaabArgs, shrinkArgs=shrinkArgs_, concatArgs=None, poolArgs=poolArgs)
    return parSaab

def get_saab_feat(patch,parSaab,maxpool=False):
    feat,_ = cwSaab(np.expand_dims(patch,axis=0), train=False, par=parSaab)
    feat = feat[0]
    if maxpool:
        feat = block_reduce(feat, (1,2,2,1), np.max)
    return feat.squeeze()

def get_par_joint(train_set,patch_size,numKernel,sizeKernel,stride,maxpool=False,pattern=None,energyTh=None):
    #patch_size = 10
    train_set = np.expand_dims(train_set, axis=0)
    train_set = window_process2(train_set, kernel_size=patch_size, stride=patch_size//5)
    tmp = train_set.shape
    #print('train_set shape:', tmp)
    train_set = train_set.reshape((tmp[0]*tmp[1]*tmp[2], patch_size, patch_size, -1))

    SaabArgs = list()
    shrinkArgs_ = list()
    poolArgs = list()
    for i in range(len(numKernel)):
        SaabArgs.append({'num_AC_kernels': numKernel[i], 'needBias':False, 'useDC':True, 'batch':None})
        shrinkArgs_.append({'size':sizeKernel, 'stride':stride, 'pad':'', 'pattern':pattern})
        poolArgs.append({'maxpool':maxpool})
    if energyTh is None:
        energyTh = [5e-2,1e-3,1e-3,1e-3]

    out, parSaab = cwSaab(train_set, train=True, par=None, depth=len(SaabArgs), energyTH=energyTh,  
                      SaabArgs=SaabArgs, shrinkArgs=shrinkArgs_, concatArgs=None, poolArgs=poolArgs)
    #print('out_spectral shape:', out[0].shape)
    gs_model, gs_feat = get_joint_gs_feat(out[0],abs=1,mode='train',gs_model=None)
    #print('out_joint shape:', gs_feat.shape)
    return {'spectral':parSaab, 'spatial':gs_model}

def get_saab_joint_feat(imgs, parJointSaab, dft_idx=None, verbose=False):
    feat,_ = cwSaab(imgs, train=False, par=parJointSaab['spectral'])
    #print([v.shape for v in feat])
    #feat = feat[0]
    #feat = feat.reshape(feat.shape[0], -1)
    if len(feat)>0:
        feat = [v.reshape(v.shape[0],-1) for v in feat]
        feat = np.concatenate(feat,axis=-1)

    
    if dft_idx is not None:
        feat = feat[:, dft_idx]
    return feat

    N = feat.shape[0]
    gs_feat = get_joint_gs_feat(feat,mode='test',gs_model=parJointSaab['spatial'])
    if verbose:
        print(f'feat shape {feat.shape}, gs_feat shape {gs_feat.shape}')
    feat = feat.reshape(N, -1)
    gs_feat = gs_feat.reshape(N, -1)
    joint_feat = np.concatenate((feat,gs_feat), axis=-1)
    if dft_idx is not None:
        joint_feat = joint_feat[:, dft_idx]
    return joint_feat

from pyutils.hkmeans import H_KMeans
import warnings
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
def get_pqr(imgs,mode='train',flag_clst=False,model=None,dim_kept=3,verbose=False):
    #print('mode=',mode,model)
    imgs = imgs.astype(float)
    means = np.mean(imgs,axis=(1,2))
    if flag_clst:
        if mode=='train':
            hk = H_KMeans(max_clusters=100,min_n_samples=10,th_purity=300)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                hk.fit(means)
            centroids = hk.centroids
            model = {'centroids':centroids}
            print('number of centroids:',len(centroids))
        else:
            centroids = model['centroids']
            
        dist = euclidean_distances(means,centroids)
        idx = np.argmin(dist,axis=1)
        if verbose:
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(projection='3d') #projection='3d'
            for i in range(len(centroids)):
                tmp = means[idx==i]
                print(len(tmp))
                ax.scatter(tmp[0],tmp[1],tmp[2],label=str(i))
            plt.legend()
            plt.show()
        centroids = centroids[idx]
        imgs -= centroids[:,np.newaxis,np.newaxis,:]
    else:
        if model is None:
            model = dict()
        imgs -= means[:,np.newaxis,np.newaxis,:]

    if mode=='train':
        pca = PCA(n_components=3).fit(imgs.reshape(-1,3))
        print('pca explained_variance_ratio_:',pca.explained_variance_ratio_)
        model['pca'] = pca
        model['dim_kept'] = dim_kept
    else:
        pca = model['pca']
    pqr = pca.transform(imgs.reshape(-1,3))
    pqr = pqr.reshape(imgs.shape)
    return model, pqr[...,:model['dim_kept']], means
def get_par_joint_pqr(train_set,numKernel,sizeKernel,stride,maxpool=False,pattern=None,energyTh=None):
    num_channel = train_set.shape[-1]
    parSaab_pqr = list()
    feat = list()
    for i in range(num_channel):
        SaabArgs = [{'num_AC_kernels': numKernel, 'needBias':False, 'useDC':True, 'batch':None}]
        shrinkArgs_ = [{'size':sizeKernel, 'stride':stride, 'pad':''}]
        poolArgs = [{'maxpool':maxpool}]
        if energyTh is None:
            energyTh = [1e-2,1e-3,1e-3,1e-3]

        out, parSaab = cwSaab(train_set[...,i][...,np.newaxis], train=True, par=None, depth=1, energyTH=energyTh[:1],  
                          SaabArgs=SaabArgs, shrinkArgs=shrinkArgs_, concatArgs=None, poolArgs=poolArgs)
        feat.append(out[0])
        parSaab_pqr.append(parSaab)
    feat = np.concatenate(feat,axis=-1)
    print('out_spectral shape:', feat.shape)
    gs_model, gs_feat = get_joint_gs_feat(feat,abs=1,mode='train',gs_model=None)
    print('out_joint shape:', gs_feat.shape)
    return {'spectral':parSaab_pqr, 'spatial':gs_model}

def get_saab_joint_feat_pqr(imgs, parJointSaab, dft_idx=None, verbose=False):
    _,imgs,img_means = get_pqr(imgs,mode='test',model=parJointSaab['pqr'])
    feat = list()
    for i in range(imgs.shape[-1]):
        out,_ = cwSaab(imgs[...,i][...,np.newaxis], train=False, par=parJointSaab['spectral'][i])
        feat.append(out[0])
    feat = np.concatenate(feat,axis=-1)

    N = feat.shape[0]

    # no gs feat
    feat = feat.reshape(N, -1)
    joint_feat = np.concatenate((feat,img_means), axis=-1)
    
    # with gs feat
    #gs_feat = get_joint_gs_feat(feat,mode='test',gs_model=parJointSaab['spatial'])
    #if verbose:
    #    print(f'feat shape {feat.shape}, gs_feat shape {gs_feat.shape}')
    #feat = feat.reshape(N, -1)
    #gs_feat = gs_feat.reshape(N, -1)
    #joint_feat = np.concatenate((feat,gs_feat,img_means), axis=-1)

    if dft_idx is not None:
        joint_feat = joint_feat[:, dft_idx]
    return joint_feat

