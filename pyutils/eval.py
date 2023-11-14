import numpy as np
import math


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


def ssd(x, y):
    s = 0
    for i in range(len(x)):
        s += (x[i] - y[i])**2
    return math.sqrt(s)


def calc_rect_dist(A, B):
    if(len(np.array(A).shape)==1):
        tmp = list()
        tmp.append(A)
        A = tmp.copy()
    if(len(np.array(B).shape)==1):
        tmp = list()
        tmp.append(B)
        B = tmp.copy()
    centerA = [[r[0]+(r[2]-1)/2.0, r[1]+(r[3]-1)/2.0] for r in A]
    centerB = [[r[0]+(r[2]-1)/2.0, r[1]+(r[3]-1)/2.0] for r in B]
    length = min(len(centerA), len(centerB))
    errCenter = [round(ssd(centerA[i], centerB[i]),4) for i in range(length)]
    return errCenter


def evaluate_seq(A,B):
    seq_overlap = calc_rect_int(A,B)
    seq_errCenter = calc_rect_dist(A,B)
    aveCoverage = np.mean(seq_overlap)
    aveErrCenter = np.mean(seq_errCenter)
    return aveCoverage, aveErrCenter


def calc_metrics(boxes, anno):
    def _intersection(rects1, rects2):
        r"""Rectangle intersection.

        Args:
            rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
                (left, top, width, height).
            rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
                (left, top, width, height).
        """
        assert rects1.shape == rects2.shape
        x1 = np.maximum(rects1[..., 0], rects2[..., 0])
        y1 = np.maximum(rects1[..., 1], rects2[..., 1])
        x2 = np.minimum(rects1[..., 0] + rects1[..., 2],
                        rects2[..., 0] + rects2[..., 2])
        y2 = np.minimum(rects1[..., 1] + rects1[..., 3],
                        rects2[..., 1] + rects2[..., 3])

        w = np.maximum(x2 - x1, 0)
        h = np.maximum(y2 - y1, 0)
        return np.stack([x1, y1, w, h]).T

    def center_error(rects1, rects2):
        r"""Center error.

        Args:
            rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
                (left, top, width, height).
            rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
                (left, top, width, height).
        """
        centers1 = rects1[..., :2] + (rects1[..., 2:] - 1) / 2
        centers2 = rects2[..., :2] + (rects2[..., 2:] - 1) / 2
        errors = np.sqrt(np.sum(np.power(centers1 - centers2, 2), axis=-1))
        return errors

    def rect_iou(rects1, rects2, bound=None):
        r"""Intersection over union.

        Args:
            rects1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
                (left, top, width, height).
            rects2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
                (left, top, width, height).
            bound (numpy.ndarray): A 4 dimensional array, denotes the bound
                (min_left, min_top, max_width, max_height) for ``rects1`` and ``rects2``.
        """
        assert rects1.shape == rects2.shape
        if bound is not None:
            # bounded rects1
            rects1[:, 0] = np.clip(rects1[:, 0], 0, bound[0])
            rects1[:, 1] = np.clip(rects1[:, 1], 0, bound[1])
            rects1[:, 2] = np.clip(rects1[:, 2], 0, bound[0] - rects1[:, 0])
            rects1[:, 3] = np.clip(rects1[:, 3], 0, bound[1] - rects1[:, 1])
            # bounded rects2
            rects2[:, 0] = np.clip(rects2[:, 0], 0, bound[0])
            rects2[:, 1] = np.clip(rects2[:, 1], 0, bound[1])
            rects2[:, 2] = np.clip(rects2[:, 2], 0, bound[0] - rects2[:, 0])
            rects2[:, 3] = np.clip(rects2[:, 3], 0, bound[1] - rects2[:, 1])

        rects_inter = _intersection(rects1, rects2)
        areas_inter = np.prod(rects_inter[..., 2:], axis=-1)

        areas1 = np.prod(rects1[..., 2:], axis=-1)
        areas2 = np.prod(rects2[..., 2:], axis=-1)
        areas_union = areas1 + areas2 - areas_inter

        eps = np.finfo(float).eps
        ious = areas_inter / (areas_union + eps)
        ious = np.clip(ious, 0.0, 1.0)
        return ious

    valid = ~np.any(np.isnan(anno), axis=1)
    if len(valid) == 0:
        print('Warning: no valid annotations')
        return None, None
    else:
        ious = rect_iou(boxes[valid, :], anno[valid, :])
        center_errors = center_error(
            boxes[valid, :], anno[valid, :])
        succ_curve,prec_curve = calc_curves(ious,center_errors)
        return np.mean(succ_curve), prec_curve[20]


def calc_curves(ious, center_errors):
    nbins_iou = 21
    nbins_ce = 51

    ious = np.asarray(ious, float)[:, np.newaxis]
    center_errors = np.asarray(center_errors, float)[:, np.newaxis]

    thr_iou = np.linspace(0, 1, nbins_iou)[np.newaxis, :]
    thr_ce = np.arange(0, nbins_ce)[np.newaxis, :]

    bin_iou = np.greater(ious, thr_iou)
    bin_ce = np.less_equal(center_errors, thr_ce)

    succ_curve = np.mean(bin_iou, axis=0)
    prec_curve = np.mean(bin_ce, axis=0)

    return succ_curve, prec_curve
