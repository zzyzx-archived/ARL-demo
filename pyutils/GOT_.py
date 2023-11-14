import numpy as np
import cv2
from glob import glob
import re
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb
from skimage.morphology import binary_opening
from collections import Counter

from .segment import *
from .hc_feat import get_hc_feature, get_hc_feature_batch
from .sb_feat import *
from .classification import patch2sample, BinaryClassifier, pred2prob
from .saab_dft.feat_utils import feature_selection


def tryfloat(s):
    try:
        return float(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryfloat(c) for c in re.split('([0-9.]+)', s)]


def sort_nicely(l):
    """Sort the given list in the way that humans expect."""
    return sorted(l, key=alphanum_key)


def load_seq(img_dir, gt_path):
    frames = sort_nicely(glob(img_dir + '*.jpg'))
    text_lines = open(gt_path).readlines()
    gt_bbs = list()
    for u in text_lines:
        bb = u.strip().split(',')
        if (len(bb) < 4):
            bb = u.strip().split('\t')
        bb = [int(float(v)) for v in bb]
        gt_bbs.append([bb[0] - 1, bb[1] - 1, bb[2], bb[3]])
    gt_bbs = np.array(gt_bbs)
    return np.array(frames), gt_bbs


def show_cam_on_image(img, mask, coe=1.2, verbose=True):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + coe * np.float32(img) / 255
    cam = cam / np.max(cam)
    if verbose:
        plt.figure(figsize=(15, 3))
        plt.subplot(151)
        plt.imshow(img[:, :, ::-1]), plt.axis('off')
        plt.subplot(152)
        plt.imshow(np.uint8(255 * cam[:, :, ::-1])), plt.axis('off')
        plt.show()
    return np.uint8(255 * cam[:, :, ::-1])


def DFT(joint_feat, y_dft, num_feat=-1, verbose=False):
    selected_idx, dft_loss = feature_selection(joint_feat, y_dft, FStype='DFT_entropy', thrs=1.0, B=16)
    if verbose:
        print(joint_feat.shape, selected_idx.shape, dft_loss.shape)
        plt.plot(dft_loss[selected_idx])
        plt.title('DFT loss')
        plt.show()
    sel_feat_idx = selected_idx[:num_feat]
    return sel_feat_idx


def get_label_XGBoost(samples, labels, patch_shape, verbose=False):
    pos_idx = labels > 0.9
    neg_idx = labels < 0.1
    pos = samples[pos_idx]
    pos = pos.reshape(pos.shape[0], -1)
    neg = samples[neg_idx]
    neg = neg.reshape(neg.shape[0], -1)
    if verbose:
        print(f'# pos {len(pos)}, # neg {len(neg)}')
    y = np.zeros(len(pos) + len(neg))
    y[:len(pos)] = 1
    clf = BinaryClassifier(n_estimators=40, learning_rate=0.1, max_depth=4)
    clf.train(Xpos=pos, Xneg=neg, flag_balance='', verbose=False)
    pred = clf.predict(np.concatenate((pos, neg), axis=0), output_prob=False)
    if verbose:
        print('training acc:', np.sum(pred == y) / len(y))
    pred = clf.predict(samples.reshape(len(samples), -1), output_prob=True)
    if verbose:
        print(np.min(pred), np.mean(pred), np.max(pred))
    prob = pred2prob(pred=pred, h=patch_shape[0], w=patch_shape[1], sample_size=8, stride=1)
    if np.max(prob) > 1:
        prob = prob / (np.max(prob))

    pos_idx = np.logical_and(labels > 0.9, pred > 0.5)
    return pos_idx, prob


def init(frames, gt_bbs, numKernel, sizeKernel, stride, energyTh, verbose=False):
    sel_id = 0
    img_path = frames[sel_id]
    bb = gt_bbs[sel_id]
    patch, scale, _ = get_crops(cv2.imread(img_path), convert_bbox_format(bb, 'center'), 32, 60, 0)
    bb_scaled = getScaledBox(bb, bb, scale, patch.shape[:2])
    img = show_track(patch, bb_scaled)
    if verbose:
        plt.imshow(img[..., ::-1])
        plt.show()

    # patches
    samples, labels = patch2sample(patch=patch, sample_size=8, stride=1, box=bb_scaled, verbose=False)

    # hc feat
    hc_feat = get_hc_feature_batch(samples, cell_size=4)
    hc_feat = hc_feat.reshape(hc_feat.shape[0], -1)

    # pqr
    parJointSaab = get_par_joint(train_set=patch, patch_size=8,
                                 numKernel=numKernel, sizeKernel=sizeKernel, stride=stride,
                                 maxpool=True, energyTh=energyTh)
    joint_feat = get_saab_joint_feat(samples, parJointSaab)
    # print(joint_feat.shape)

    total_feat = np.concatenate((joint_feat, hc_feat), axis=-1)
    # print('total_feat shape:', total_feat.shape)
    # total_feat = joint_feat

    y_dft = np.zeros(len(total_feat))
    y_dft[labels > 0.9] = 1
    sel_feat_idx = DFT(total_feat, y_dft, num_feat=192, verbose=verbose)

    labels_r1, prob = get_label_XGBoost(total_feat[:, sel_feat_idx], labels, patch.shape[:2])
    # img_cam1 = show_cam_on_image(img, prob, verbose=False)
    labels_r2, prob = get_label_XGBoost(total_feat[:, sel_feat_idx], labels_r1, patch.shape[:2])
    # img_cam2 = show_cam_on_image(img, prob, verbose=False)
    y_dft = np.zeros(len(total_feat))
    y_dft[labels_r2] = 1
    sel_feat_idx = DFT(total_feat, y_dft, num_feat=50)

    return parJointSaab, sel_feat_idx, total_feat, y_dft


def get_trained_clf(pos, neg):
    clf = BinaryClassifier(n_estimators=40, learning_rate=0.1, max_depth=4)
    clf.train(Xpos=pos, Xneg=neg, flag_balance='', verbose=False)
    return clf


def get_clf_pred(clf, feat, patch_shape, sample_size, sample_stride):
    pred = clf.predict(feat, output_prob=True)
    prob = pred2prob(pred=pred, h=patch_shape[0], w=patch_shape[1], sample_size=sample_size, stride=sample_stride)
    if np.max(prob) > 1:
        prob_n = prob / (np.max(prob))
    else:
        prob_n = prob
    return pred, prob, prob_n


def superpixel_binaryseg_prob(patch, heatmap, th=0.1):
    # output: segments labeled with probs
    segments = felzenszwalb(patch, scale=100, sigma=0.6, min_size=50, multichannel=True)

    cnt1 = Counter(segments.flatten()).most_common()
    soft_seg = np.zeros(segments.shape).astype(float)
    cnt_prob = list()
    probs_dict = dict()
    for i in range(len(cnt1)):
        tmp = np.mean(heatmap[segments == cnt1[i][0]])
        soft_seg[segments == cnt1[i][0]] = tmp
        cnt_prob.append(tmp)
        probs_dict[cnt1[i][0]] = tmp
    cnt_prob = np.array(cnt_prob)
    cnt_prob = -np.sort(-cnt_prob)
    proposals = list()
    for v in cnt_prob:
        if v < th:
            break
        binary_seg = np.zeros(segments.shape)
        binary_seg[soft_seg >= v] = 1
        proposals.append(seg2box(binary_opening(binary_seg), np.array([])))
    return segments, soft_seg, proposals, probs_dict


def viz_boxes(img, boxes, LineWidth=5, draw=True):
    colors = [(255, 0, 0), (0, 165, 255), (0, 255, 0), (0, 0, 255)]
    for i, box in enumerate(boxes):
        if len(box) > 0:
            img = show_track(img, box, color=colors[min(i, len(colors) - 1)], LineWidth=LineWidth)
    if draw:
        plt.imshow(img[..., ::-1])
        plt.show()
    return img


def QA(pred, prob, bb, th_bin=0.7, verbose=False):
    # quality assessment of prob map
    def getScoreInBox(mat, bbox):
        if len(bbox) < 1:
            return 0
        m, n = mat.shape[:2]
        xmin, xmax = int(bbox[0]), int(bbox[0] + bbox[2] - 1)
        xmin, xmax = max(0, xmin), min(n - 1, xmax)
        ymin, ymax = int(bbox[1]), int(bbox[1] + bbox[3] - 1)
        ymin, ymax = max(0, ymin), min(m - 1, ymax)
        tmp = mat[ymin:ymax + 1, xmin:xmax + 1]
        if len(tmp) > 0:
            score = np.sum(tmp)
        else:
            score = 0
        return score

    vmin = np.min(prob)
    vmax = np.max(prob)
    if np.min(pred) >= 0.5 or np.max(pred) <= 0.5 or vmin >= 0.5 or vmax <= 0.5:
        return False, None, 0, 100
    prob = prob / vmax
    binary = np.zeros(prob.shape, dtype=np.int)
    binary[prob > th_bin] = 1
    max_blob, num_blob = findmaxisland(binary)
    conf = getScoreInBox(binary, bb) / np.sum(binary)

    isgood = conf > 0.8 or num_blob < 2

    return isgood, max_blob, conf, num_blob


def get_new_samples(feat, box_labels, pred_prob, th1=0.7, th2=0.15):
    pos_idx = np.logical_and(box_labels > 0.9, pred_prob > th1)
    neg_idx = np.logical_and(box_labels < 0.1, pred_prob < th2)
    pos = feat[pos_idx]
    neg = feat[neg_idx]
    viz_vec = 0.4 * np.ones((len(box_labels),))
    viz_vec[pos_idx] = 1
    viz_vec[neg_idx] = 0
    return pos, neg, viz_vec


def update(frame, bb, parJointSaab, sel_feat_idx, clf, ref_bb, pos_init, neg_init, flag_seg_adv=False, verbose=False):
    sample_size = 8
    sample_stride = 2
    patch, scale, _ = get_crops(frame, convert_bbox_format(bb, 'center'), 32, 60, 0)
    bb_scaled = getScaledBox(bb, bb, scale, patch.shape[:2])
    if max(bb_scaled[-2:]) > 0.8 * patch.shape[0]:
        patch, scale, _ = get_crops(frame, convert_bbox_format(bb, 'center'), 32, 80, 0)
        bb_scaled = getScaledBox(bb, bb, scale, patch.shape[:2])
    ref_bb_scaled = getScaledBox(bb, ref_bb, scale, patch.shape[:2])

    img = patch
    samples, labels = patch2sample(patch=patch, sample_size=sample_size, stride=sample_stride,
                                   box=bb_scaled, verbose=False)

    # hc feat
    hc_feat = get_hc_feature_batch(samples, cell_size=4)
    hc_feat = hc_feat.reshape(hc_feat.shape[0], -1)

    # extract joint feat
    joint_feat = get_saab_joint_feat(samples, parJointSaab, verbose=False)
    total_feat = np.concatenate((joint_feat, hc_feat), axis=-1)
    total_feat = total_feat[:, sel_feat_idx]

    pred, prob, prob_n = get_clf_pred(clf, total_feat, patch.shape, sample_size, sample_stride)
    img_cam = show_cam_on_image(img, prob_n, coe=1.2, verbose=False)

    segments, soft_seg_prob, seg_prob_bbs, probs_dict = superpixel_binaryseg_prob(patch, prob, th=0.1)

    isgood, max_blob, conf, num_blob = QA(pred, prob, bb_scaled, th_bin=0.5)
    if not isgood:
        isgood, max_blob, conf, num_blob = QA(pred, prob, bb_scaled, th_bin=0.7)

    # sample new data
    update_data = {
        'total_feat': total_feat,
        'labels': labels,
        'pred': pred,
        'patch': patch,
        'sample_size': sample_size,
        'sample_stride': sample_stride,
        'pos_init': pos_init,
        'neg_init': neg_init
    }
    clf_new = None

    # merge with baseline ref_bb
    if isgood:
        clf_bbt = seg2box(max_blob, [])
        if len(clf_bbt) > 0:
            img_cam = show_track(img_cam, clf_bbt, color=(0, 255, 0), LineWidth=1)
    else:
        clf_bbt = []
    if calc_rect_int(bb_scaled, ref_bb_scaled)[0] > 0.5:
        tmp_ref_bb = ref_bb_scaled
    else:
        tmp_ref_bb = bb_scaled
    weight = calc_rect_int(clf_bbt, tmp_ref_bb)[0] if flag_seg_adv else 1
    ious = [calc_rect_int(i, tmp_ref_bb)[0] + weight * calc_rect_int(i, clf_bbt)[0] for i in seg_prob_bbs]
    if len(ious) > 0:
        clf_bb = seg_prob_bbs[np.argmax(ious)]
        clf_bb_seg = clf_bb
        if verbose:
            print('iou(bb,ref_bb)=%.3f' % (calc_rect_int(bb_scaled, ref_bb_scaled)[0]))
        if calc_rect_int(bb_scaled, ref_bb_scaled)[0] > 0.5:
            tmp_better = calc_rect_int(clf_bbt, ref_bb_scaled)[0] > calc_rect_int(clf_bb, ref_bb_scaled)[0]
            if verbose:
                print('clf_vs_ref_bb %.3fm seg_vs_ref_bb %.3f' % (calc_rect_int(clf_bbt, ref_bb_scaled)[0],
                                                                  calc_rect_int(clf_bb, ref_bb_scaled)[0]))
        else:
            tmp_better = calc_rect_int(clf_bbt, bb_scaled)[0] > calc_rect_int(clf_bb, bb_scaled)[0]
            if verbose:
                print('clf_vs_bb %.3fm seg_vs_bb %.3f' % (calc_rect_int(clf_bbt, bb_scaled)[0],
                                                          calc_rect_int(clf_bb, bb_scaled)[0]))
        if not flag_seg_adv and \
                len(clf_bbt) > 0 and tmp_better:
            clf_bb = np.array(clf_bbt)
        clf_bb_frame = getFrameBox(clf_bb, scale, bb, patch.shape[:2])
    else:
        clf_bb_seg = []
        clf_bb = clf_bb_frame = []

    viz_data = [patch,
                img_cam,
                viz_boxes(patch, [seg_prob_bbs[np.argmax(ious)]], LineWidth=1, draw=False),
                scale,
                bb,
                conf,
                num_blob,
                seg_prob_bbs,
                seg_prob_bbs[np.argmax(ious)]]
    debug_data = [ref_bb_scaled, clf_bbt, clf_bb_seg]
    return pred, prob_n, bb_scaled, clf_new, clf_bb, clf_bb_frame, clf_bbt, viz_data, debug_data, update_data


def update_clf(update_data):
    patch = update_data['patch']
    total_feat = update_data['total_feat']
    labels = update_data['labels']
    pred = update_data['pred']
    sample_size = update_data['sample_size']
    sample_stride = update_data['sample_stride']
    pos_init = update_data['pos_init']
    neg_init = update_data['neg_init']

    pos, neg, _ = get_new_samples(total_feat, labels, pred)

    # update clf
    clf_new = get_trained_clf(pos, neg)
    pred_new, prob_new, prob_n_new = get_clf_pred(clf_new, total_feat, patch.shape, sample_size, sample_stride)
    pos, neg, _ = get_new_samples(total_feat, labels, pred_new, th1=0.8)
    clf_up = get_trained_clf(np.concatenate((pos_init, pos), axis=0), np.concatenate((neg_init, neg), axis=0))
    return clf_up


def clip_box_to_frame(bb, h, w):
    xmin, ymin, xmax, ymax = bb[0], bb[1], bb[0] + bb[2] - 1, bb[1] + bb[3] - 1
    xmin = min(w - 1, max(0, xmin))
    ymin = min(h - 1, max(0, ymin))
    xmax = max(0, min(w - 1, xmax))
    ymax = max(0, min(h - 1, ymax))
    return np.array([xmin, ymin, xmax - xmin + 1, ymax - ymin + 1])


def get_coords(shape, bb, fnum, bnum):
    np.random.seed(32)
    minN = 10
    m, n = shape[:2]
    xmin, ymin, xmax, ymax = bb[0], bb[1], bb[0] + bb[2] - 1, bb[1] + bb[3] - 1
    full = getRc_patch(m, n)
    fg_coords = np.array([v for v in full if xmin < v[1] < xmax and ymin < v[0] < ymax])
    bg_coords = np.array([v for v in full if not (xmin <= v[1] <= xmax and ymin <= v[0] <= ymax)])
    if len(fg_coords) < minN:
        fg_coords = np.array([v for v in full if xmin <= v[1] <= xmax and ymin <= v[0] <= ymax])
        bg_coords = np.array([v for v in full if not (xmin - 5 <= v[1] <= xmax + 5 and ymin - 5 <= v[0] <= ymax + 5)])

    if len(fg_coords) > fnum:
        idx = np.random.choice(len(fg_coords), size=fnum, replace=False)
        fg_coords = fg_coords[idx]
    if len(bg_coords) > bnum:
        idx = np.random.choice(len(bg_coords), size=bnum, replace=False)
        bg_coords = bg_coords[idx]
    if len(fg_coords) < minN or len(bg_coords) < minN:
        return False, (np.array([]), np.array([])), (np.array([]), np.array([]))
    fg_coords = (fg_coords[:, 0], fg_coords[:, 1])
    bg_coords = (bg_coords[:, 0], bg_coords[:, 1])
    return True, fg_coords, bg_coords


def get_coords_prior(shape, bb, fnum, bnum, prob):
    np.random.seed(32)
    minN = 10
    m, n = shape[:2]
    xmin, ymin, xmax, ymax = bb[0], bb[1], bb[0] + bb[2] - 1, bb[1] + bb[3] - 1
    full = getRc_patch(m, n)
    fg_coords = np.array([v for v in full if prob[v[0], v[1]] > 0.5 and xmin < v[1] < xmax and ymin < v[0] < ymax])
    bg_coords = np.array(
        [v for v in full if prob[v[0], v[1]] < 0.5 and not (xmin <= v[1] <= xmax and ymin <= v[0] <= ymax)])
    if len(fg_coords) < minN:
        fg_coords = np.array([v for v in full if xmin <= v[1] <= xmax and ymin <= v[0] <= ymax])
        bg_coords = np.array([v for v in full if not (xmin - 5 <= v[1] <= xmax + 5 and ymin - 5 <= v[0] <= ymax + 5)])

    if len(fg_coords) > fnum:
        idx = np.random.choice(len(fg_coords), size=fnum, replace=False)
        fg_coords = fg_coords[idx]
    if len(bg_coords) > bnum:
        idx = np.random.choice(len(bg_coords), size=bnum, replace=False)
        bg_coords = bg_coords[idx]
    if len(fg_coords) < minN or len(bg_coords) < minN:
        return False, (np.array([]), np.array([])), (np.array([]), np.array([]))
    fg_coords = (fg_coords[:, 0], fg_coords[:, 1])
    bg_coords = (bg_coords[:, 0], bg_coords[:, 1])
    return True, fg_coords, bg_coords


def reg_map(P, Q, lbda=1):
    res = (P + lbda ** 2 * Q) / (1 + lbda ** 2)
    return res


def fft2(x):
    return np.fft.fft(np.fft.fft(x, axis=1), axis=0).astype(np.complex64)


def ifft2(x):
    return np.fft.ifft(np.fft.ifft(x, axis=1), axis=0).astype(np.complex64)


def circonv2D(x, y):
    h, w = x.shape
    xf = fft2(x)
    yf = fft2(y)
    rf = xf * np.conj(yf) / np.size(xf)
    r = np.real(ifft2(rf))
    curr = np.unravel_index(np.argmax(r, axis=None), r.shape)
    dy = curr[0] - h if curr[0] + 1 > h / 2 else curr[0]
    dx = curr[1] - w if curr[1] + 1 > w / 2 else curr[1]
    return dy, dx


def isgray(img):
    # img = cv2.imread(imgpath)
    if len(img.shape) < 3: return True
    if img.shape[2] == 1: return True
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    if (b == g).all() and (b == r).all(): return True
    return False
