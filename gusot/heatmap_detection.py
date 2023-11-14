import math
import time
import numpy as np


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def get_int_curves(mat: np.ndarray):
    smooth_win = 10
    int_x = np.sum(mat, axis=0)
    int_x = smooth(int_x, smooth_win)
    int_y = np.sum(mat, axis=1)
    int_y = smooth(int_y, smooth_win)
    return int_x, int_y


def regress_box_gmc(mat: np.ndarray, bbox: np.ndarray, ratio: float = 1, th: float = 0.1):
    # print('ratio:',ratio)
    m, n = mat.shape[:2]
    x, y, w, h = np.round(bbox).astype(np.int32)
    cx, cy = round(bbox[0] + (bbox[2] + 1) / 2), round(bbox[1] + (bbox[3] + 1) / 2)

    mat_clean = mat.copy()
    mat_clean[:max(0, round(y - ratio * h)), :] = 0
    mat_clean[min(round(y + (ratio + 1) * h), m - 1):, :] = 0
    mat_clean[:, :max(0, round(x - ratio * w))] = 0
    mat_clean[:, min(round(x + (ratio + 1) * w), n - 1):] = 0

    smooth_win = 10
    int_x = np.sum(mat_clean, axis=0)
    if np.max(int_x) < h:
        return np.array([0, 0, 0, 0]), np.array([]), np.array([])
    int_x = smooth(int_x, smooth_win)
    int_x = int_x / np.max(int_x + 1e-16)
    int_y = np.sum(mat_clean, axis=1)
    if np.max(int_y) < w:
        return np.array([0, 0, 0, 0]), np.array([]), np.array([])
    int_y = smooth(int_y, smooth_win)
    int_y = int_y / np.max(int_y + 1e-16)

    max_posx = np.argmax(int_x)
    if True or max_posx >= x - ratio * w and max_posx <= x + w - 1 + ratio * w:
        idx = int_x[:max_posx] < (np.min(int_x[:max_posx]) + th)
        if np.size(idx) > 0:
            xmin = np.max(np.arange(0, max_posx)[idx])
        else:
            xmin = x
        idx = int_x[max_posx + 1:] < (np.min(int_x[max_posx + 1:]) + th)
        if np.size(idx) > 0:
            xmax = np.min(np.arange(max_posx + 1, n)[idx])
        else:
            xmax = x + w - 1
    else:
        xmin = x
        xmax = x + w - 1

    max_posy = np.argmax(int_y)
    if True or max_posy >= y - ratio * h and max_posy <= y + h - 1 + ratio * h:
        idx = int_y[:max_posy] < (np.min(int_y[:max_posy]) + th)
        if np.size(idx) > 0:
            ymin = np.max(np.arange(0, max_posy)[idx])
        else:
            ymin = y
        idx = int_y[max_posy + 1:] < (np.min(int_y[max_posy + 1:]) + th)
        if np.size(idx) > 0:
            ymax = np.min(np.arange(max_posy + 1, m)[idx])
        else:
            ymax = y + h - 1
    else:
        ymin = y
        ymax = y + h - 1

    return np.array([xmin, ymin, xmax - xmin + 1, ymax - ymin + 1]), int_x, int_y


# GMC
import cv2


def GMC_total_all(prev: np.ndarray, curr: np.ndarray, win_size: (int, int), k: int = 1, rate_value: float = 0.0,
                  min_value: float = 0.0, row: int = 0, col: int = 0, radius: int = -1):
    shrink = np.max(prev.shape) > 1000
    _, afm, dif = GMC(prev, curr, shrink)

    prev = np.array(prev).astype(float)
    curr = np.array(curr).astype(float)
    direct_dif = np.abs(curr - prev)
    direct_dif[direct_dif < np.mean(direct_dif)] = 0
    tmp = np.mean(dif)
    if tmp > 10:  # large lv
        return False, np.array([]), np.array([]), np.array([]), None, None, None

    if False and (tmp < 0.05 and np.mean(direct_dif) > 2):
        # print('camera zoom, use direct dif')
        dif_use = direct_dif
        dif_int = mean_filter2(dif_use, win_size)
    else:
        dif_use = dif
        [m, n] = dif_use.shape
        idx1 = direct_dif < np.mean(direct_dif)
        idx2 = np.ones((m, n)).astype(np.bool_)
        r_pad = int(0.2 * m)
        c_pad = int(0.2 * n)
        idx2[r_pad:m - r_pad, c_pad:n - c_pad] = False
        dif_use[np.logical_and(idx1, idx2)] = 0

        dif_int = mean_filter2(dif_use, win_size)

    gmc_pos = argkmax2(dif_int, k, rate_value, min_value, row, col, radius)

    int_x, int_y = get_int_curves(dif_use)
    return True, dif_int, gmc_pos, dif_use, None, int_x, int_y, np.array(afm)


def GMC_total(prev: np.ndarray, curr: np.ndarray):
    shrink = np.max(prev.shape) > 1000
    success, afm, dif = GMC(prev, curr, shrink)

    prev = np.array(prev).astype(float)
    curr = np.array(curr).astype(float)
    direct_dif = np.abs(curr - prev)
    direct_dif[direct_dif < (4 * np.mean(direct_dif) + 0 * np.max(direct_dif)) / 4] = 0
    # print(np.mean(dif),np.mean(direct_dif))
    tmp = np.mean(dif)
    if (tmp < 0.05 and np.mean(direct_dif) > 2):
        # print('camera zoom, use direct dif')
        return success, np.array([[1, 0, 0], [0, 1, 0]]), direct_dif

    [m, n] = dif.shape
    idx1 = direct_dif < (4 * np.mean(direct_dif) + 0 * np.max(direct_dif)) / 4
    idx2 = np.ones((m, n)).astype(np.bool_)
    r_pad = int(0.2 * m)
    c_pad = int(0.2 * n)
    idx2[r_pad:m - r_pad, c_pad:n - c_pad] = False
    dif[np.logical_and(idx1, idx2)] = 0

    return success, np.array(afm), dif


def GMC(prev: np.ndarray, curr: np.ndarray, shrink=False):
    verbose = False
    prev_gray = prev.astype(np.uint8)
    if shrink:
        shrink_rate = 0.5
        shrink_w = int(prev_gray.shape[1] * shrink_rate)
        shrink_h = int(prev_gray.shape[0] * shrink_rate)
        prev_gray_shrink = cv2.resize(prev_gray, (shrink_w, shrink_h), interpolation=cv2.INTER_AREA)

    curr_gray = curr.astype(np.uint8)

    if shrink:
        prev_pts = cv2.goodFeaturesToTrack(prev_gray_shrink,
                                           maxCorners=200,
                                           qualityLevel=0.01,
                                           minDistance=30,
                                           blockSize=3)
        prev_pts = np.round(prev_pts * [prev_gray.shape[1] / shrink_w, prev_gray.shape[0] / shrink_h]).astype(
            np.float32)
    else:
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                           maxCorners=200,
                                           qualityLevel=0.01,
                                           minDistance=30,
                                           blockSize=3)

    try:
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    except Exception as e:
        print(e)
        dif = np.abs(curr_gray.astype(np.float32) - prev_gray.astype(np.float32))
        dif[dif < np.mean(dif) + 30] = 0
        return True, [], dif

    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]
    if prev_pts.shape[0] == 0:
        dif = np.abs(curr_gray.astype(np.float32) - prev_gray.astype(np.float32))
        dif[dif < np.mean(dif) + 30] = 0
        return True, [], dif
    # print('GMC shape:',prev_pts.shape[0],prev_pts.shape, curr_pts.shape)
    # m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False) # cv2 3.4.2
    m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)  # cv2 higher versions
    if m is None:
        # print('GMC failed: cannot find affine transform')
        dif = np.abs(curr_gray.astype(np.float32) - prev_gray.astype(np.float32))
        dif[dif < np.mean(dif) + 30] = 0
        return True, [], dif
    dx = m[0, 2]
    dy = m[1, 2]
    da = np.arctan2(m[1, 0], m[0, 0])
    if verbose:
        print(prev_pts.shape, curr_pts.shape)
        print([dx, dy, da])
    if len(prev_pts) < 50 and abs(da) >= 0.01:
        dif = np.abs(curr_gray.astype(np.float32) - prev_gray.astype(np.float32))
        dif[dif < np.mean(dif) + 30] = 0
        return True, [], dif

    h, w = prev.shape[:2]
    # frame_stabilized = cv2.warpAffine(prev, m, (w,h))
    gray_stabilized = cv2.warpAffine(prev_gray, m, (w, h))
    dif = np.abs(curr_gray.astype(np.float32) - gray_stabilized.astype(np.float32))
    dif = fixBorder(dif, dx, dy)

    # th to clean
    dif[dif < np.mean(dif) + 30] = 0

    return True, m, dif


def fixBorder(dif, dx, dy):
    h, w = dif.shape[:2]
    if dx >= 0:
        xmin = 0
        xmax = min(w - 1, int(dx) + 1)
    else:
        xmin = max(0, int(w - 1 + dx - 5))
        xmax = w - 1
    if dy >= 0:
        ymin = 0
        ymax = min(h - 1, int(dy) + 1)
    else:
        ymin = max(0, int(h - 1 + dy - 5))
        ymax = h - 1
    dif[ymin:ymax, :] = 0
    dif[:, xmin:xmax] = 0
    return dif


def mean_filter2(mat: np.ndarray, win_size: (int, int)):
    mat = np.array(mat)
    # mat = cv2.resize(mat,(im_size[1],im_size[0]))
    m, n = mat.shape
    h, w = win_size

    if m + 1 <= h or n + 1 <= w:
        return np.zeros((0, 0))

    mat = np.vstack((np.zeros((1, n)), mat))
    mat = np.hstack((np.zeros((m + 1, 1)), mat))
    cum = mat.cumsum(axis=0)
    np.cumsum(cum, axis=1, out=cum)

    res = cum[:m - h + 1, :n - w + 1] + cum[h:h + m + 1, w:w + n + 1] - \
          cum[:m - h + 1, w:w + n + 1] - cum[h:h + m + 1, :n - w + 1]
    res /= h * w

    return res


def pad_output(mat: np.array, h: int, w: int):
    m, n = mat.shape
    left = (w - 1) // 2
    right = w - left - 1
    up = (h - 1) // 2
    bot = h - up - 1
    mat = np.hstack((np.zeros((m, left)), mat))
    mat = np.hstack((mat, np.zeros((m, right))))
    mat = np.vstack((np.zeros((up, n + w - 1)), mat))
    mat = np.vstack((mat, np.zeros((bot, n + w - 1))))
    return mat


def argkmax2(mat: np.ndarray, k: int = 1, rate_value: float = 0.0, min_value: float = 0.0, row: int = 0, col: int = 0,
             radius: int = -1):
    # row, col: top-left corner
    m, n = mat.shape[:2]
    if radius > 0:
        rmin = int(max(0, row - radius))
        cmin = int(max(0, col - radius))
        rmax = int(min(m - 1, row + radius))
        cmax = int(min(n - 1, col + radius))
        mat = mat[rmin:rmax, cmin:cmax]
        mat_ravel = mat.ravel()
        # print('row',row,'col',col,'[rmin,cmin,rmax,cmax]:',[rmin,cmin,rmax,cmax])
    else:
        mat_ravel = mat.ravel()

    idx_ravel = np.argsort(-mat_ravel)
    idxs = np.dstack(np.unravel_index(idx_ravel, mat.shape))[0]

    val = mat_ravel[idx_ravel]
    if np.size(val) < 1:
        return np.array([])
    val_th = max(val[0] * rate_value, min_value)
    idxs = idxs[val >= val_th]

    if radius > 0:
        ans = idxs[:k] + [rmin, cmin]
    else:
        ans = idxs[:k]

    return ans


def iou_fixed_size(bbox1: (int, int), bbox2: (int, int), size: (int, int)):
    x1, y1 = bbox1
    x2, y2 = bbox2
    h, w = size

    x = max(0, h + min(x1, x2) - max(x1, x2))
    y = max(0, w + min(y1, y2) - max(y1, y2))

    i_area = x * y
    return i_area / (h * w * 2 - i_area)


def isAdj_fixed_size(bbox1: (int, int), bbox2: (int, int), size: (int, int)):
    x1, y1 = bbox1
    x2, y2 = bbox2
    h, w = size

    overlap_x = h + min(x1, x2) - max(x1, x2) > 0
    adjacency_x = h + min(x1, x2) - max(x1, x2) == 0
    overlap_y = w + min(y1, y2) - max(y1, y2) > 0
    adjacency_y = w + min(y1, y2) - max(y1, y2) == 0

    return (overlap_x and adjacency_y) or (overlap_y and adjacency_x)


def nms_fixed_size(bbox: np.ndarray, size: (int, int), iou_th: float = 0.5):
    nms_bbox = list()
    size = np.array(size).reshape(1, 2)
    while len(bbox) > 0:
        crt_bbox = bbox[0]
        nms_bbox.append(crt_bbox)

        boundmax = np.maximum(bbox, crt_bbox)
        boundmin = np.minimum(bbox, crt_bbox)
        i_area_size = np.maximum(boundmin + size - boundmax, 0)
        i_area = i_area_size[:, 0] * i_area_size[:, 1]
        iou = i_area / (size.prod() * 2 - i_area)

        mask = iou < iou_th
        bbox = bbox[mask]

    return nms_bbox


def heatmap_detection(heatmap: np.ndarray, win_size: (int, int),
                      rate_num: float = 0.1,
                      rate_value: float = 0.2,
                      min_value: float = 0.1,
                      iou_th: float = 0.2):
    heatmap_block_sum = mean_filter2(heatmap, win_size)
    bbox = argkmax2(heatmap_block_sum,
                    k=int(rate_num * heatmap_block_sum.size), rate_value=rate_value, min_value=min_value)
    nms_bbox = nms_fixed_size(bbox, win_size, iou_th=iou_th)

    return nms_bbox


def argcummax(mat: np.ndarray):
    res_val = np.zeros_like(mat, dtype=mat.dtype)
    res_idx = np.zeros_like(mat, dtype=int)
    m, n = mat.shape

    res_val[0, 0] = mat[0, 0]
    res_idx[0, 0] = np.ravel_multi_index((0, 0), mat.shape)

    for j in range(1, n):
        if mat[0, j] > res_val[0, j - 1]:
            res_val[0, j] = mat[0, j]
            res_idx[0, j] = np.ravel_multi_index((0, j), mat.shape)
        else:
            res_val[0, j] = res_val[0, j - 1]
            res_idx[0, j] = res_idx[0, j - 1]

    for i in range(1, m):
        if mat[i, 0] > res_val[i - 1, 0]:
            res_val[i, 0] = mat[i, 0]
            res_idx[i, 0] = np.ravel_multi_index((i, 0), mat.shape)
        else:
            res_val[i, 0] = res_val[i - 1, 0]
            res_idx[i, 0] = res_idx[i - 1, 0]

        for j in range(1, n):
            if res_val[i - 1, j] >= res_val[i, j - 1]:
                val = res_val[i - 1, j]
                idx = res_idx[i - 1, j]
            else:
                val = res_val[i, j - 1]
                idx = res_idx[i, j - 1]
            if mat[i, j] > val:
                res_val[i, j] = mat[i, j]
                res_idx[i, j] = np.ravel_multi_index((i, j), mat.shape)
            else:
                res_val[i, j] = val
                res_idx[i, j] = idx

    return res_val, res_idx


def max_2blocks(mat: np.ndarray, win_size: (int, int),
                overlap_allow: (int, int) = (0, 0),
                cache_block_sum=None,
                cache_argcummax=None):
    block_sum = mean_filter2(mat, win_size) if cache_block_sum is None else cache_block_sum
    max_val, max_idx = argcummax(block_sum) if cache_argcummax is None else cache_argcummax

    m, n = block_sum.shape
    h, w = win_size
    dx, dy = overlap_allow

    res_val = -1
    res_idx = (None, None)

    for i in range(m):
        for j in range(n):
            val = 0
            idx = None
            if i + dx >= h and max_val[i + dx - h, -1] > val:
                val = max_val[i + dx - h, -1]
                idx = max_idx[i + dx - h, -1]
            if j + dy >= w and max_val[-1, j + dy - w] > val:
                val = max_val[-1, j + dy - w]
                idx = max_idx[-1, j + dy - w]

            if block_sum[i, j] + val > res_val:
                res_val = block_sum[i, j] + val
                res_idx = ((i, j),) if idx is None else \
                    (np.unravel_index(idx, block_sum.shape), (i, j))
    return res_val, res_idx


def heatmap_detection_2blocks(img: np.ndarray, win_size: (int, int), iou_th: float = 0.5):
    h, w = win_size
    i_area = 2 * h * w * iou_th / (1 + iou_th)

    overlap_allow = (int(i_area / w), int(i_area / h))
    _, bbox = max_2blocks(img, win_size, overlap_allow=overlap_allow)

    return bbox


def heatmap_detection_2blocks_split(img: np.ndarray, win_size: (int, int)):
    h, w = win_size
    block_sum = mean_filter2(img, win_size)
    argcummaxs = argcummax(block_sum)

    iou_f = 0
    i_area = 2 * h * w * iou_f / (1 + iou_f)
    overlap_f = (int(i_area / w), int(i_area / h))
    _, bboxs_f = max_2blocks(img, win_size, overlap_allow=overlap_f,
                             cache_block_sum=block_sum,
                             cache_argcummax=argcummaxs)

    if isAdj_fixed_size(bboxs_f[0], bboxs_f[1], win_size):
        return bboxs_f

    iou_t = 1
    i_area = 2 * h * w * iou_t / (1 + iou_t)
    overlap_t = (int(i_area / w), int(i_area / h))
    bbox_greedy = np.unravel_index(block_sum.argmax(), shape=block_sum.shape)
    bboxs_t = (bbox_greedy, bbox_greedy)

    while iou_f < iou_t:
        iou_m = (iou_f + iou_t) / 2
        i_area = 2 * h * w * iou_m / (1 + iou_m)
        overlap_m = (int(i_area / w), int(i_area / h))

        if overlap_m == overlap_f or overlap_m == overlap_t:
            break

        _, bboxs_m = max_2blocks(img, win_size, overlap_allow=overlap_m,
                                 cache_block_sum=block_sum,
                                 cache_argcummax=argcummaxs)
        iou = iou_fixed_size(bboxs_m[0], bboxs_m[1], win_size)

        if iou > 0 or isAdj_fixed_size(bboxs_m[0], bboxs_m[1], win_size):
            iou_t = iou_m
            overlap_t = overlap_m
            bboxs_t = bboxs_m
        else:
            iou_f = iou_m
            overlap_f = overlap_m
    return bboxs_t


def draw_bbox_on_img(img: np.ndarray, bbox: [(int, int)], size: (int, int), brightness=0.6):
    img = img.copy()
    white_color = img.max()
    black_color = img.min()
    k = 1 - brightness
    base_color = white_color + k * black_color

    for x, y in bbox:
        img[x:x + size[0] - 1, y] *= -k
        img[x:x + size[0] - 1, y] += base_color

        img[x + 1:x + size[0], y + size[1] - 1] *= -k
        img[x + 1:x + size[0], y + size[1] - 1] += base_color

        img[x, y + 1:y + size[1]] *= -k
        img[x, y + 1:y + size[1]] += base_color

        img[x + size[0] - 1, y:y + size[1] - 1] *= -k
        img[x + size[0] - 1, y:y + size[1] - 1] += base_color

    return img


class Timer(object):
    def __init__(self, tic_instantly=False, total_num=None):
        super().__init__()
        self.reset(tic_instantly)
        self.total_num = total_num
        self.n_sigma = 3.2905  # 99.9%

    def __str__(self):
        k_unit, str_unit = self.get_proper_unit_sec(self.s)
        return 'Usage %.2f %s/%d. %s' % (
            self.s / k_unit, str_unit, self.n, self.str_avg(False))  # ,self.str_spd(False))

    def _str_progress(self, total_num):
        return '%.1f%% = %d/%d. Elapsed %.2f sec. %s %s %s %s' % (
            100.0 * self.n / total_num, self.n, total_num,
            self.s, self.str_avg(True), self.str_spd(True),
            self.str_rem(True), self.str_tot(True)
        )

    def show_progress(self, verbose=True):
        self.toc()
        self.tic()
        if verbose:
            print(self._str_progress(self.total_num))

    def reset(self, tic_instantly=False):
        self.n = 0
        self.s = 0
        self.s2 = 0
        self._max = -1
        self._min = float('+inf')
        self._t0 = None
        if tic_instantly: self.tic()
        return

    def tic(self):
        assert self._t0 is None, 'Error! Already used "Timer.tic()" before this "Timer.tic()".'
        self._t0 = time.time()

    def toc(self):
        assert self._t0 is not None, 'Error! Please use "Timer.tic()"" before "Timer.toc()".'
        t = time.time() - self._t0
        self._t0 = None

        self.n += 1
        self.s += t
        self.s2 += t * t
        self._max = max(t, self._max)
        self._min = min(t, self._min)

    def set_n_sigma(self, n_sigma, verbose=True):
        n = self.n_sigma
        self.n_sigma = n_sigma
        if verbose:
            print('n_sigma was change from %.2f (%.5f%%) to %.2f (%.5f%%).' % (
                n, self.sigma2conf(n) * 100.0,
                self.n_sigma, self.sigma2conf(self.n_sigma) * 100.0,
            ))

        return

    @property
    def avg(self):
        return self.s / self.n if self.n > 0 else float('nan')

    @property
    def std(self):
        return math.sqrt((self.n * self.s2 - self.s * self.s) / (self.n * (self.n - 1))) if \
            self.n > 1 else float('nan')

    @property
    def std_hz(self):
        avg = self.avg
        return self.std / (avg * avg)

    @property
    def max(self):
        return self._max if self._max != -1 else float('nan')

    @property
    def min(self):
        return self._min if self._min != float('+inf') else float('nan')

    @staticmethod
    def sigma2conf(n):
        return math.erf(n / math.sqrt(2))

    def str_avg(self, brief=True):
        k_unit, str_unit = self.get_proper_unit_sec(self.avg)
        str_intv = '' if brief else 'in [%.2f,%.2f] ' % (self.min / k_unit, self.max / k_unit)
        str_out = 'Avg %.2f +-%.2f %s%s.' if brief else 'Avg %.2f +-%.2f %s%s.'
        return str_out % (self.avg / k_unit, self.std / k_unit, str_intv, str_unit)

    def str_spd(self, brief=True):
        k_unit, str_unit = self.get_proper_unit_hz(1 / self.avg)
        str_intv = '' if brief else 'in [%.4f,%.4f] ' % (1.0 / (self.max * k_unit), 1.0 / (self.min * k_unit))
        str_out = 'Spd %.2f +-%.2f %s%s.' if brief else 'Spd %.4f +-%.4f %s%s.'
        return str_out % (1 / (self.avg * k_unit), self.std_hz / k_unit, str_intv, str_unit)

    def str_rem(self, brief=True):
        avg = self.avg
        std = self.std
        n_left = self.total_num - self.n
        t_rem = avg * n_left
        dt_rem = std * math.sqrt(abs(n_left)) * self.n_sigma

        k_unit, str_unit = self.get_proper_unit_sec(t_rem)
        str_intv = '' if brief else 'in [%.4f,%.4f] ' % ((t_rem - dt_rem) / k_unit, (t_rem + dt_rem) / k_unit)
        str_out = 'Rem %.2f +-%.2f %s%s.' if brief else 'Rem %.4f +-%.4f %s%s.'
        return str_out % (t_rem / k_unit, dt_rem / k_unit, str_intv, str_unit)

    def str_tot(self, brief=True):
        avg = self.avg
        std = self.std
        t_tot = avg * self.total_num
        dt_tot = std * math.sqrt(self.total_num) * self.n_sigma

        k_unit, str_unit = self.get_proper_unit_sec(t_tot)
        str_intv = '' if brief else 'in [%.4f,%.4f] ' % ((t_tot - dt_tot) / k_unit, (t_tot + dt_tot) / k_unit)
        str_out = 'Tot %.2f +-%.2f %s%s.' if brief else 'Tot %.4f +-%.4f %s%s.'
        return str_out % (t_tot / k_unit, dt_tot / k_unit, str_intv, str_unit)

    @staticmethod
    def get_proper_unit_sec(sec):
        if sec < 0.0001:
            return 1e-6, 'us'
        elif sec < 0.1:
            return 0.001, 'ms'
        elif sec < 60:
            return 1, 'sec'
        elif sec < 3600:
            return 60, 'min'
        else:
            return 3600, 'h'

    @staticmethod
    def get_proper_unit_hz(hz):
        if hz < 0.1 / 60:
            return 1 / 3600, 'fph'
        elif hz < 0.1:
            return 1 / 60, 'fpm'
        elif hz < 100:
            return 1, 'Hz'
        else:
            return 1000, 'kHz'

    def display(self):
        print(self)
