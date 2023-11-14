import cv2
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import time

from pyutils.hc_features.features import TableFeature

##################################################33

def floodprint(mat,pos,old,new,connection_type):
    m,n = mat.shape[:2]
    res = 0
    stack = list()
    i,j = pos
    if mat[i][j]==old:
        mat[i][j] = new
        res += 1
        stack.append((i,j))
    while len(stack)>0:
        i,j = stack.pop()
        if connection_type==4:
            neighbors = ((i-1,j),(i,j-1),(i+1,j),(i,j+1))
        else:
            neighbors = ((i-1,j-1),(i-1,j),(i,j-1),(i+1,j+1),(i+1,j),(i,j+1),(i-1,j+1),(i+1,j-1))
        for ii,jj in neighbors:
            if ii<0 or ii>=m or jj<0 or jj>=n:
                continue
            if mat[ii][jj]==old:
                mat[ii][jj] = new
                stack.append((ii,jj))
                res += 1
    return res
def findmaxisland(mat,connection_type=8,val=1,return_ratio=False):
    # val: value for island pixels
    mat = mat.copy()
    num = 0
    maxisland_id = None
    maxisland_area = 0
    total_area = 0
    m,n = mat.shape[:2]
    for i in range(m):
        for j in range(n):
            if mat[i][j]==val:
                area = floodprint(mat,(i,j),val,num+2,connection_type=connection_type)
                if area>maxisland_area:
                    maxisland_area = area
                    maxisland_id = num+2
                num += 1
                total_area += area

    if maxisland_id is not None:
        mat[mat!=maxisland_id] = 0
        mat[mat==maxisland_id] = 1
    if return_ratio:
        return mat,num,maxisland_area/(total_area+1e-16)
    return mat,num
def showisland(mat,min_size=10,connection_type=8,val=1):
    # val: value for island pixels
    mat = mat.copy()
    island_label = 2
    num = 0
    maxisland_id = None
    maxisland_area = 0
    m,n = mat.shape[:2]
    for i in range(m):
        for j in range(n):
            if mat[i][j]==val:
                island_id = num+10
                area = floodprint(mat,(i,j),val,island_id,connection_type=connection_type)
                #print('area,min size:',area,min_size)
                if area<min_size:
                    floodprint(mat,(i,j),island_id,0,connection_type=connection_type)
                num += 1

    mat[mat>0] = 1
    return mat,num
def segment_clip(mat,bbox,val=0):
    # set content out of bb as val
    out = val*np.ones_like(mat)
    m,n = mat.shape[:2]
    xmin,xmax = int(bbox[0]), int(bbox[0]+bbox[2]-1)
    xmin,xmax = max(0,xmin), min(n-1,xmax)
    ymin,ymax = int(bbox[1]), int(bbox[1]+bbox[3]-1)
    ymin,ymax = max(0,ymin), min(m-1,ymax)
    out[ymin:ymax+1, xmin:xmax+1] = mat[ymin:ymax+1, xmin:xmax+1]
    return out
def seg2box(segments,bbox):
    m,n = segments.shape[:2]
    try:
        x_int = np.sum(segments,axis=0)
        xidx = np.arange(n)
        xidx = xidx[x_int>0]
        xmin,xmax = xidx[0],xidx[-1]

        y_int = np.sum(segments,axis=1)
        yidx = np.arange(m)
        yidx = yidx[y_int>0]
        ymin,ymax = yidx[0],yidx[-1]
    except Exception as e:
        #print(e)
        return bbox
    return np.array([xmin,ymin,xmax-xmin+1,ymax-ymin+1])
def calc_rect_int(A, B):
    if len(A)<1 or len(B)<1:
        return [0]
    if(len(np.array(A).shape)==1):
        tmp = list()
        tmp.append(A)
        A = tmp
    if(len(np.array(B).shape)==1):
        tmp = list()
        tmp.append(B)
        B = tmp
    leftA = [a[0] for a in A]
    bottomA = [a[1] for a in A]
    rightA = [leftA[i] + A[i][2] - 1 for i in range(len(A))]
    topA = [bottomA[i] + A[i][3] - 1 for i in range(len(A))]

    leftB = [b[0] for b in B]
    bottomB = [b[1] for b in B]
    rightB = [leftB[i] + B[i][2] - 1 for i in range(len(B))]
    topB = [bottomB[i] + B[i][3] - 1 for i in range(len(B))]

    overlap = []
    length = min(len(leftA), len(leftB))
    for i in range(length):
        tmp = (max(0, min(rightA[i], rightB[i]) - max(leftA[i], leftB[i])+1)
            * max(0, min(topA[i], topB[i]) - max(bottomA[i], bottomB[i])+1))
        areaA = A[i][2] * A[i][3]
        areaB = B[i][2] * B[i][3]
        overlap.append(tmp/float(areaA+areaB-tmp))

    return overlap
#####################################################
def getScoreInBox(mat,bbox):
    if len(bbox)<1:
        return 0
    m,n = mat.shape[:2]
    xmin,xmax = int(bbox[0]), int(bbox[0]+bbox[2]-1)
    xmin,xmax = max(0,xmin), min(n-1,xmax)
    ymin,ymax = int(bbox[1]), int(bbox[1]+bbox[3]-1)
    ymin,ymax = max(0,ymin), min(m-1,ymax)
    tmp = mat[ymin:ymax+1, xmin:xmax+1].astype(np.float)
    if len(tmp)>0:
        score = np.mean(tmp)
    else:
        score = 0
    return score
def isInBox(ptRc,bbox):
    # Return whether the point falls in the bbox
    # bbox: topleft-based, wrt patch
    # ptRc: wrt patch
    xmin = bbox[0]
    xmax = bbox[0]+bbox[2]-1
    ymin = bbox[1]
    ymax = bbox[1]+bbox[3]-1
    #print(ptRc,bbox, (ptRc[0]>=ymin and ptRc[0]<=ymax) and (ptRc[1]>=xmin and ptRc[1]<=xmax))
    return (ptRc[0]>=ymin and ptRc[0]<=ymax) and (ptRc[1]>=xmin and ptRc[1]<=xmax)
def patch2grid(patch,ksize,stride):
    if len(patch.shape)<3:
        patch = patch[...,np.newaxis]
    c = patch.shape[-1]
    grids = view_as_windows(patch, (ksize, ksize, c), step=(stride, stride, c))
    grids = np.squeeze(grids)
    #print(grids.shape)
    return grids
def getScaledBox(prev_bb,gt_bb,scale,patch_shape):
    # prev_bb: last prediction of bb wrt frame, topleft-based
    # gt_bb: [x,y,w,h] left-top based
    # return: topleft-base bb wrt patch
    cx,cy = convert_bbox_format(prev_bb, 'center')[:2]
    cx_patch,cy_patch = [get_center(patch_shape[1]), get_center(patch_shape[0])]
    xmin = scale*(gt_bb[0]-cx)+cx_patch
    xmax = scale*(gt_bb[0]+gt_bb[2]-1-cx)+cx_patch
    ymin = scale*(gt_bb[1]-cy)+cy_patch
    ymax = scale*(gt_bb[1]+gt_bb[3]-1-cy)+cy_patch
    tmp = [xmin,ymin,xmax-xmin+1,ymax-ymin+1]
    return tmp
def show_track(img,track_bb,color=(0,0,255),LineWidth=2):
    if track_bb is None or len(track_bb)<1:
        return img
    plt_bb=np.array(track_bb).astype(np.int32)
    #print(plt_bb)
    img_t=cv2.rectangle(img.copy(),(plt_bb[0],plt_bb[1]),(plt_bb[0]+plt_bb[2],plt_bb[1]+plt_bb[3]),color,LineWidth)
    return img_t
def get_config():
    config = dict()
    config['size_z'] = 32#48
    config['size_x'] = 48
    config['context_amount'] = 0#0.2
    config['num_sp'] = 5
    config['searchwin_size'] = 60
    config['scale'] = 100
    return config
def get_center(x):
    return (np.array(x) - 1.) / 2.
def convert_bbox_format(bbox, to):
    x, y, target_width, target_height = bbox[0], bbox[1], bbox[2], bbox[3]
    if to == 'topleft':
        x -= get_center(target_width)
        y -= get_center(target_height)
    elif to == 'center':
        y += get_center(target_height)
        x += get_center(target_width)
    else:
        raise ValueError("Bbox format: {} was not recognized".format(to))
    return [x, y, target_width, target_height]
def getRc_patch(h,w):
    # Return: [h*w,2] rows and cols wrt patch
    rc_patch = np.arange(0, h*w)
    rc_patch = np.unravel_index(rc_patch, (max(len(rc_patch),w),w))
    rc_patch = np.dstack(rc_patch)[0]
    return rc_patch
def getFrameBox(bb,scale,prev_bb,patch_shape):
    if len(bb)<1:
        return np.array([])
    cx,cy = convert_bbox_format(prev_bb, 'center')[:2]
    cx_patch = get_center(patch_shape[1])
    cy_patch = get_center(patch_shape[0])
    xmin = (bb[0]-cx_patch)/scale + cx
    xmax = (bb[0]+bb[2]-1-cx_patch)/scale + cx
    ymin = (bb[1]-cy_patch)/scale + cy
    ymax = (bb[1]+bb[3]-1-cy_patch)/scale + cy
    return np.array([xmin,ymin,xmax-xmin+1,ymax-ymin+1])
def getPatch(img,bbox,config,fill_color=None):
    patch,scale_x1,_ = get_crops(img,
                                    convert_bbox_format(bbox,'center'),
                                    config['size_z'], config['size_x'], config['context_amount'], fill_color=fill_color)

    return patch,scale_x1
def getPatch_matlab(imgpath,bbox,size_z,size_x,context=0,verbose=False):
    t0 = time.time()
    img = cv2.imread(imgpath)
    if verbose:
        print('img reading:',time.time()-t0)
    t0 = time.time()
    patch,scale_x1,_ = get_crops(img,
                                convert_bbox_format(bbox,'center'),
                                size_z, size_x, context)
    if verbose:
        print('img croping:',time.time()-t0)
    return cv2.cvtColor(patch, cv2.COLOR_BGR2RGB),scale_x1

def get_subwindow_avg(im, pos, model_sz, original_sz,fill_color=None):
    # avg_chans = np.mean(im, axis=(0, 1)) # This version is 3x slower
    if fill_color is None:
        avg_chans = [np.mean(im[:, :, 0]), np.mean(im[:, :, 1]), np.mean(im[:, :, 2])]
    else:
        avg_chans = np.array(fill_color,dtype=im.dtype)
    if not original_sz:
        original_sz = model_sz
    sz = original_sz
    im_sz = im.shape
    # make sure the size is not too small
    assert im_sz[0] > 2 and im_sz[1] > 2
    c = [get_center(s) for s in sz]

    # check out-of-bounds coordinates, and set them to avg_chans
    context_xmin = np.int(np.round(pos[1] - c[1]))
    context_xmax = np.int(context_xmin + sz[1] - 1)
    context_ymin = np.int(np.round(pos[0] - c[0]))
    context_ymax = np.int(context_ymin + sz[0] - 1)
    left_pad = np.int(np.maximum(0, -context_xmin))
    top_pad = np.int(np.maximum(0, -context_ymin))
    right_pad = np.int(np.maximum(0, context_xmax - im_sz[1] + 1))
    bottom_pad = np.int(np.maximum(0, context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad
    if top_pad > 0 or bottom_pad > 0 or left_pad > 0 or right_pad > 0:
        R = np.pad(im[:, :, 0], ((top_pad, bottom_pad), (left_pad, right_pad)), 'constant', constant_values=(avg_chans[0]))
        G = np.pad(im[:, :, 1], ((top_pad, bottom_pad), (left_pad, right_pad)), 'constant', constant_values=(avg_chans[1]))
        B = np.pad(im[:, :, 2], ((top_pad, bottom_pad), (left_pad, right_pad)), 'constant', constant_values=(avg_chans[2]))

        im = np.stack((R, G, B), axis=2)

    im_patch_original = im[context_ymin:context_ymax + 1, context_xmin:context_xmax + 1, :]
    if not (model_sz[0] == original_sz[0] and model_sz[1] == original_sz[1]):
        im_patch = cv2.resize(im_patch_original, tuple(model_sz), interpolation = cv2.INTER_LANCZOS4)
        #im_patch = cv2.resize(im_patch_original, tuple(model_sz), interpolation = cv2.INTER_LINEAR)
    else:
        im_patch = im_patch_original
    return im_patch, left_pad, top_pad, right_pad, bottom_pad
def get_crops(im, bbox, size_z, size_x, context_amount, factor=1,fill_color=None):
    """Obtain image sub-window, padding with avg channel if area goes outside of border
    Adapted from https://github.com/bertinetto/siamese-fc/blob/master/ILSVRC15-curation/save_crops.m#L46
    Args:
    im: Image ndarray
    bbox: Named tuple (x, y, width, height) x, y corresponds to the crops center
    size_z: Target + context size
    size_x: The resultant crop size
    context_amount: The amount of context
    factor: scaling factor
    Returns:
    image crop: Image ndarray
    """
    cy, cx, h, w = bbox[1], bbox[0], bbox[3], bbox[2]
    wc_z = w + context_amount * (w + h)
    hc_z = h + context_amount * (w + h)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = size_z / s_z

    d_search = (size_x - size_z) / 2
    pad = d_search / scale_z
    base_s_x = s_z + 2 * pad
    s_x = factor * base_s_x
    base_scale_x = size_x / base_s_x
    scale_x = base_scale_x / factor

    image_crop_x, _, _, _, _ = get_subwindow_avg(im, [cy, cx], [size_x, size_x], [np.round(s_x), np.round(s_x)], fill_color=fill_color)
    #print(wc_z,hc_z,s_z,s_x)
    objSize_new = [scale_z/factor*h, scale_z/factor*w]
    return image_crop_x, scale_x, objSize_new

##################################################
from collections import Counter
def get_hist(ary,bin_list,norm=True):
    if len(ary)<1:
        return np.zeros(len(bin_list))
    cnt = Counter(np.ravel(ary))
    hist = np.array([cnt[v] for v in bin_list])
    if norm:
        hist = hist/np.sum(hist)
    return hist
from sklearn.metrics.pairwise import euclidean_distances
def get_color_cov(patch,feat_hard,bin_list,norm=True):
    tmp = np.unique(feat_hard)
    ratio = np.array([np.sum(feat_hard==v) for v in tmp])
    ratio = ratio/np.sum(ratio)
    #print(ratio)
    centroids = [np.mean(patch[feat_hard==v],axis=0)/255.0 for v in tmp]
    dist = euclidean_distances(centroids)
    #print(dist)
    dist_exp = np.exp(dist**2)
    weights = np.array([np.sum(dist_exp[v,:]*ratio) for v in range(len(dist_exp))])
    weights = weights/np.sum(weights)
    wei_dict = dict()
    for i,v in enumerate(tmp):
        wei_dict[v] = weights[i]
    #print(wei_dict)
    return wei_dict
def get_main_color(patch,feat,bb,num_color=5,ct_dist=None,verbose=False):
    m,n = feat.shape[:2]
    x,y,w,h = bb
    xmin = int(min(max(0,x),n-1))
    xmax = int(max(0,min(n,x+w)))
    ymin = int(min(max(0,y),m-1))
    ymax = int(max(0,min(m,y+h)))
    
    idx = np.zeros(feat.shape[:2],dtype=np.bool_)
    idx[ymin:ymax,xmin:xmax] = True
    
    if len(feat.shape)>2:
        num_color = feat.shape[-1]
        cn_feat = np.argmax(feat,axis=-1)
    else:
        cn_feat = feat
    cn_in = cn_feat[idx]
    cn_out = cn_feat[~idx]
    hist_in  = get_hist(cn_in,np.arange(num_color))
    hist_out  = get_hist(cn_out,np.arange(num_color))
    hist_dif = hist_in-hist_out
    
    wei_dict = get_color_cov(patch,cn_feat,bin_list=np.arange(num_color),norm=True)
    hist_dif_wei = np.array([hist_dif[v]*(wei_dict.get(v,0)**2) for v in range(num_color)])
    
    salient_set = [v for v in range(len(hist_dif_wei)) if hist_dif_wei[v]>0.009 and hist_out[v]<0.2 or \
                   hist_dif_wei[v]>0.001 and hist_out[v]<0.05]
    if len(salient_set)<1:
        print('empty salient_set,',np.max(hist_dif_wei))
        #salient_set = [np.argmax(hist_dif_wei)]
    
    tmp = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            if cn_feat[i][j] in salient_set:
                tmp[i][j] = 1
    tmp,_ = findmaxisland(tmp)
    bb_ct = convert_bbox_format(bb,'center')
    mid_salient_score = getScoreInBox(tmp,convert_bbox_format([bb_ct[0],bb_ct[1],bb_ct[2]/2,bb_ct[3]/2],'topleft'))
    salient_bb_big = seg2box(tmp,[])
    salient_iou_big = calc_rect_int(bb,salient_bb_big)[0]
    tmp = segment_clip(tmp,bb)
    salient_bb = seg2box(tmp,[])
    salient_iou = calc_rect_int(bb,salient_bb)[0]
    good,bad = np.argmax(hist_dif_wei),np.argmin(hist_dif_wei)
    if ct_dist is not None:
        salient_dif = ct_dist[good][bad]
    else:
        salient_dif = 255
    
    if verbose:
        print('mid_salient_score:',mid_salient_score)
        print('salient_dif:',salient_dif)
        for i in range(num_color):
            tmp = np.zeros(feat.shape[:2],dtype=np.bool_)
            tmp[cn_feat==i] = 1
            plt.subplot(2,5,i+1)
            plt.imshow(tmp)
        plt.show()
        
        plt.figure(figsize=(18,3))
        #plt.subplot(151),plt.stem(get_hist(cn_feat,np.arange(num_color))),plt.gca().set_title('total')
        tmp = show_track(patch,bb)
        plt.subplot(161),plt.imshow(tmp[...,::-1])
        plt.subplot(162),plt.stem(hist_in),plt.gca().set_title('in')
        plt.subplot(163),plt.stem(hist_out),plt.gca().set_title('out')
        plt.subplot(164),plt.stem(hist_dif),plt.gca().set_title('dif')
        plt.subplot(165),plt.stem(hist_dif_wei),plt.gca().set_title('weighted hist')
        tmp = np.zeros(patch.shape,dtype=np.uint8)
        for i in range(m):
            for j in range(n):
                if cn_feat[i][j] in salient_set:
                    tmp[i][j] = patch[i][j]
        tmp = show_track(tmp,salient_bb_big,color=(0,255,0))
        plt.subplot(166),plt.imshow(show_track(tmp,salient_bb)[...,::-1])
        plt.gca().set_title('salient color (%d, %.2f-%.2f)'%(len(salient_set),salient_iou_big,salient_iou))
        plt.show()
    
    if salient_iou<0.5 or salient_iou_big<0.5 or mid_salient_score<0.3 or salient_dif<30:
        print('salient_iou,salient_iou_big,mid_salient_score,salient_dif:',salient_iou,salient_iou_big,mid_salient_score,salient_dif)
        salient_set = []
    return salient_set,hist_dif_wei
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
def get_color_keys(patch,cts=None,verbose=False):
    flag_success = True
    m,n = patch.shape[:2]
    X = patch.reshape((m*n,-1))

    if cts is None:
        gmm = GaussianMixture(n_components=5,random_state=0)
        gmm.fit(X)
        preds = gmm.predict(X)
        preds = preds.reshape((m,n))
        cts = gmm.means_
    else:
        cts = np.array(cts)

    while len(cts)>3:
        dist = euclidean_distances(cts)
        dist[dist<1] = 255
        idx = dist<30
        if np.sum(idx)<1:
            break
        a,b = np.where(dist==np.min(dist))
        new_cts = list()
        for i in range(len(cts)):
            if i!=a[0] and i!=b[0]:
                new_cts.append(cts[i])
        preds = extract_cn_feature_gmm(X,cts,is_patch=False)
        tmp = np.concatenate((X[preds==a[0]],X[preds==b[0]]),axis=0)
        tmp = np.mean(tmp,axis=0)
        new_cts.append(tmp)
        cts = new_cts
    #print(cts)
    cts = np.array(cts)
    dist = euclidean_distances(cts)
    if np.max(dist)<120:
        flag_success = False
    if verbose:
        print(dist)
        print(cts)
        preds = extract_cn_feature_gmm(patch,cts)
        for i in range(len(cts)):
            tmp = np.zeros((m,n),dtype=np.bool_)
            tmp[preds==i] = 1
            plt.subplot(4,5,i+1)
            plt.imshow(tmp)
            tmp = np.zeros_like(patch)
            tmp[preds==i] = np.uint8(cts[i])
            plt.subplot(4,5,i+11)
            plt.imshow(tmp[...,::-1])
        plt.show()
    return flag_success,cts,dist
def extract_cn_feature_gmm(X,cts,is_patch=True):
    if is_patch:
        m,n = X.shape[:2]
        X = X.reshape((m*n,-1))
    dist = euclidean_distances(X,cts)
    preds = np.argmin(dist,axis=-1)
    if is_patch:
        preds = preds.reshape((m,n))
    return preds

def extract_cn_feature(img,cell_size=1):
    cn = TableFeature(fname='cn', cell_size=cell_size, compressed_dim=11, table_name="CNnorm",
                      use_for_color=True)

    if np.all(img[:, :, 0] == img[:, :, 1]):
        img = img[:, :, :1]
    else:
        # # pyECO using RGB format
        img = img[:, :, ::-1]
    h,w=img.shape[:2]
    cn_feature = \
    cn.get_features(img, np.array(np.array([h/2,w/2]), dtype=np.int16), np.array([h,w]), 1, normalization=False)[
        0][:, :, :, 0]
    #print('cn_feature.shape:', cn_feature.shape)
    #print('cnfeature:',cn_feature.shape,cn_feature.min(),cn_feature.max())
    return cn_feature

def init_seg_params(img_path,bb,mode='pdf',cts=None,verbose=False):
    init_patch,scale = getPatch(cv2.imread(img_path),bb,get_config())
    init_bb = getScaledBox(bb,bb,scale,init_patch.shape[:2])

    #'''
    if mode=='pdf':
        flag_has_colorkey = True
        init_cn_feat = extract_cn_feature(init_patch)
        tmp = np.argmax(init_cn_feat,axis=-1)
        init_cts = np.array([np.mean(init_patch[tmp==v],axis=0) if np.sum(tmp==v)>0 else [0,0,0] for v in range(init_cn_feat.shape[-1])])
        print(init_cts.shape)
        num_color = init_cn_feat.shape[-1]
        salient_set,hist_dif_wei = get_main_color(init_patch,init_cn_feat,init_bb,
                                                  num_color=num_color,ct_dist=euclidean_distances(init_cts),verbose=verbose)
    else:
        flag_has_colorkey,init_cts,init_cts_dist = get_color_keys(init_patch,cts=cts,verbose=verbose)
        init_cn_feat = extract_cn_feature_gmm(init_patch,init_cts)
        num_color = len(init_cts)
        salient_set,hist_dif_wei = get_main_color(init_patch,init_cn_feat,init_bb,
                                                  num_color=num_color,ct_dist=init_cts_dist,verbose=verbose)
    #'''
    flag_seg = len(salient_set)>0 and flag_has_colorkey and (bb[2]>=50 and bb[3]>=50 or bb[2]<50 and bb[3]<50)
    return flag_seg,(init_cts,salient_set,hist_dif_wei),init_patch, init_cts[np.argmin(hist_dif_wei)]

def get_coords_prior(patch,bb,salient_set,hist,fnum,bnum,gmm_cts=None,mode='pdf',verbose=False):
    np.random.seed(32)
    minN = 10
    m,n = patch.shape[:2]
    
    if mode=='pdf':
        cn_feat = extract_cn_feature(patch)
        cn_feat = np.argmax(cn_feat,axis=-1)
    else:
        cn_feat = extract_cn_feature_gmm(patch,gmm_cts)
    salient_map = [v in salient_set for v in cn_feat.ravel()]
    salient_map = np.reshape(salient_map,(m,n))
    s_in = getScoreInBox(salient_map,bb)
    s_out = (np.sum(salient_map)-s_in*np.prod(bb[2:]))/(m*n-np.prod(bb[2:]))
    if verbose:
        print('cn_feat shape:',cn_feat.shape)
        print(np.unique(cn_feat))
        print(bb)
        print(Counter(cn_feat.ravel()))
        print('ratio of salient color points: in-%.3f, out-%.3f'%(s_in,s_out))
    
    xmin,ymin,xmax,ymax = bb[0],bb[1],bb[0]+bb[2]-1,bb[1]+bb[3]-1
    full = getRc_patch(m,n)
    fg_coords = np.array([v for v in full if \
                          xmin<v[1]<xmax and ymin<v[0]<ymax and cn_feat[v[0]][v[1]] in salient_set])
    bg_idx = np.argmin(hist)
    bg_coords = np.array([v for v in full if \
                not (xmin<=v[1]<=xmax and ymin<=v[0]<=ymax) and not (cn_feat[v[0]][v[1]] in salient_set) \
                or cn_feat[v[0]][v[1]]==bg_idx])
    #if len(fg_coords)<minN:
    #    fg_coords = np.array([v for v in full if xmin<=v[1]<=xmax and ymin<=v[0]<=ymax])
    #    bg_coords = np.array([v for v in full if not (xmin<=v[1]<=xmax and ymin<=v[0]<=ymax)])
    
    if len(fg_coords)>fnum:
        idx = np.random.choice(len(fg_coords),size=fnum,replace=False)
        fg_coords = fg_coords[idx]
    if len(bg_coords)>bnum:
        idx = np.random.choice(len(bg_coords),size=bnum,replace=False)
        bg_coords = bg_coords[idx]
    if len(fg_coords)<minN or len(bg_coords)<minN:
        return False,(np.array([]),np.array([])),(np.array([]),np.array([])),[]
    fg_coords = (fg_coords[:,0],fg_coords[:,1])
    bg_coords = (bg_coords[:,0],bg_coords[:,1])
    return True,fg_coords,bg_coords,[salient_map,s_in,s_out]
def get_box(seg,scale,bb,verbose=False):
    seg = np.array(seg)
    pred_scaled = getScaledBox(bb,bb,scale,seg.shape[:2])
    maxisland,_ = findmaxisland(seg)
    bb1 = seg2box(maxisland,[0,0,0,0])
    islands,_ = showisland(seg,min_size=25)
    bb2 = seg2box(islands,[0,0,0,0])
    if verbose:
        tmp = np.zeros((seg.shape[0],seg.shape[1],3))
        tmp[seg>0] = [255,255,255]
        tmp = show_track(tmp,bb1,color=(0,255,0))
        plt.imshow(show_track(tmp,bb2)[...,::-1])
        plt.show()
    if not (np.prod(bb1[2:])<12*12 and np.prod(bb2[2:])<12*12):
        if calc_rect_int(pred_scaled,bb2)[0]>calc_rect_int(pred_scaled,bb1)[0]:
            pred_scaled = bb2
        else:
            pred_scaled = bb1
    else:
        pred_scaled = [0,0,0,0]
    return pred_scaled,getFrameBox(pred_scaled,scale,bb,seg.shape[:2])
def process_frame(img_path,bb,info,mode='pdf',verbose=False):
    init_cts,salient_set,hist_dif_wei = info
    patch,scale = getPatch(cv2.imread(img_path),bb,get_config(), fill_color=init_cts[np.argmin(hist_dif_wei)])
    bb_scaled = getScaledBox(bb,bb,scale,patch.shape[:2])
    #flag_normal,fg_coords,bg_coords = get_coords(patch.shape[:2],bb_scaled,fnum=30,bnum=100)
    flag_normal,fg_coords,bg_coords,salient_scores = get_coords_prior(patch,bb_scaled,salient_set,hist_dif_wei,
                                                       fnum=100,bnum=100,gmm_cts=init_cts,mode=mode,verbose=verbose)
    if len(salient_scores)>1 and not (salient_scores[1]>0.1 and salient_scores[2]<0.5):
        flag_normal = False
    if verbose:
        print(flag_normal,'fg %d, bg %d'%(len(fg_coords[0]),len(bg_coords[0])))
    
    return flag_normal,patch,scale,fg_coords,bg_coords
def process_frame_fast(patch,scale,bb,info,mode='pdf',verbose=False):
    patch = np.array(patch,dtype=np.uint8)
    patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
    init_cts,salient_set,hist_dif_wei = info
    bb_scaled = getScaledBox(bb,bb,scale,patch.shape[:2])
    #flag_normal,fg_coords,bg_coords = get_coords(patch.shape[:2],bb_scaled,fnum=30,bnum=100)
    flag_normal,fg_coords,bg_coords,salient_scores = get_coords_prior(patch,bb_scaled,salient_set,hist_dif_wei,
                                                       fnum=100,bnum=100,gmm_cts=init_cts,mode=mode,verbose=verbose)
    if len(salient_scores)>1 and not (salient_scores[1]>0.1 and salient_scores[2]<0.5):
        flag_normal = False
    if verbose:
        print(flag_normal,'fg %d, bg %d'%(len(fg_coords[0]),len(bg_coords[0])))
    
    return flag_normal,patch,scale,fg_coords,bg_coords


from itertools import chain, combinations
def powerset(iterable):
    #"powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1)))
def box_refine_segments(patch,segments,box,boxes,verbose=False):
    patch = np.array(patch)
    segments = np.array(segments)
    cnt1 = Counter(segments.flatten())
    #print(segments.shape,cnt1)

    m,n = segments.shape[:2]
    xmin,xmax = int(box[0]), int(box[0]+box[2]-1)
    xmin,xmax = max(0,xmin), min(n-1,xmax)
    ymin,ymax = int(box[1]), int(box[1]+box[3]-1)
    ymin,ymax = max(0,ymin), min(m-1,ymax)
    cnt2 = Counter(segments[ymin:ymax+1, xmin:xmax+1].flatten())

    fg_keys = list()
    fg_ratio = dict()
    for v in cnt2.keys():
        fg_ratio[v] = cnt2[v]/cnt1[v]
        if fg_ratio[v]>0.5:
            fg_keys.append(v)
    if len(fg_keys)<1:
        fg_keys.append(cnt2.most_common()[0][0])
    psets = powerset(fg_keys)
    if verbose:
        print('keys in bag:',fg_keys)
        print(psets)
    
    best_box = list()
    best_iou = 0
    best_num = 0

    tmp_std = 0
    for pset in psets:
        binary_seg = np.zeros((m,n)).astype(np.bool_)
        for v in pset:
            binary_seg[segments==v] = 1
        pred_bb = seg2box(binary_seg,[])
        score = 0
        for i,bb in enumerate(boxes):
            score += calc_rect_int(bb,pred_bb)[0] * (1+0*(i>0))
        if score>best_iou:
            best_box = pred_bb
            best_iou = score
            best_num = len(pset)
            if len(pset)==1:
                tmp = patch[segments==pset[0]]
                tmp_std = max(np.std(tmp,axis=0))
            else:
                tmp_std = 0
    #if best_iou<0.2:
    #    best_box = box
    #sprint('tmp std:',tmp_std)
    return np.array(best_box), float(tmp_std), float(len(cnt1.keys())), best_num