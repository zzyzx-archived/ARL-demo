import pickle
import numpy as np
import cv2
from .hc_features import _gradient
from .segment import convert_bbox_format


def cos_window(sz):
    cos_window = np.hanning(int(sz[1]))[:, np.newaxis].dot(np.hanning(int(sz[0]))[np.newaxis, :])
    return cos_window


def gaussian2d_rolled_labels(sz, sigma):
    w, h = sz
    xs, ys = np.meshgrid(np.arange(w) - w // 2, np.arange(h) - h // 2)
    dist = (xs ** 2 + ys ** 2) / (sigma ** 2)
    labels = np.exp(-0.5 * dist)
    labels = np.roll(labels, -int(np.floor(sz[0] / 2)), axis=1)
    labels = np.roll(labels, -int(np.floor(sz[1] / 2)), axis=0)
    return labels


def fhog(I, bin_size=8, num_orients=9, clip=0.2, crop=False):
    soft_bin = -1
    M, O = _gradient.gradMag(I.astype(np.float32), 0, True)
    H = _gradient.fhog(M, O, bin_size, num_orients, soft_bin, clip)
    return H


def extract_hog_feature(img, cell_size=4):
    fhog_feature = fhog(img.astype(np.float32), cell_size, num_orients=9, clip=0.2)[:, :, :-1]
    return fhog_feature


def fft2(x):
    return np.fft.fft(np.fft.fft(x, axis=1), axis=0).astype(np.complex64)


def ifft2(x):
    return np.fft.ifft(np.fft.ifft(x, axis=1), axis=0).astype(np.complex64)


class TableFeature:
    def __init__(self, fname, compressed_dim, table_name, use_for_color, cell_size=1):
        self.fname = fname
        self._table_name = table_name
        self._color = use_for_color
        self._cell_size = cell_size
        self._compressed_dim = [compressed_dim]
        self._factor = 32
        self._den = 8
        # load table
        self._table = pickle.load(open("CNnorm.pkl", "rb"))

        self.num_dim = [self._table.shape[1]]
        self.min_cell_size = self._cell_size
        self.penalty = [0.]
        self.sample_sz = None
        self.data_sz = None

    def mround(self, x):
        x_ = x.copy()
        idx = (x - np.floor(x)) >= 0.5
        x_[idx] = np.floor(x[idx]) + 1
        idx = ~idx
        x_[idx] = np.floor(x[idx])
        return x_

    def _sample_patch(self, im, pos, sample_sz, output_sz):
        pos = np.floor(pos)
        sample_sz = np.maximum(self.mround(sample_sz), 1)
        xs = np.floor(pos[1]) + np.arange(0, sample_sz[1] + 1) - np.floor((sample_sz[1] + 1) / 2)
        ys = np.floor(pos[0]) + np.arange(0, sample_sz[0] + 1) - np.floor((sample_sz[0] + 1) / 2)
        xmin = max(0, int(xs.min()))
        xmax = min(im.shape[1], int(xs.max()))
        ymin = max(0, int(ys.min()))
        ymax = min(im.shape[0], int(ys.max()))
        # extract image
        im_patch = im[ymin:ymax, xmin:xmax, :]
        left = right = top = down = 0
        if xs.min() < 0:
            left = int(abs(xs.min()))
        if xs.max() > im.shape[1]:
            right = int(xs.max() - im.shape[1])
        if ys.min() < 0:
            top = int(abs(ys.min()))
        if ys.max() > im.shape[0]:
            down = int(ys.max() - im.shape[0])
        if left != 0 or right != 0 or top != 0 or down != 0:
            im_patch = cv2.copyMakeBorder(im_patch, top, down, left, right, cv2.BORDER_REPLICATE)
        # im_patch = cv2.resize(im_patch, (int(output_sz[0]), int(output_sz[1])))
        im_patch = cv2.resize(im_patch, (int(output_sz[1]), int(output_sz[0])), cv2.INTER_CUBIC)
        if len(im_patch.shape) == 2:
            im_patch = im_patch[:, :, np.newaxis]
        return im_patch

    def _feature_normalization(self, x):
        if hasattr(self.config, 'normalize_power') and self.config.normalize_power > 0:
            if self.config.normalize_power == 2:
                x = x * np.sqrt((x.shape[0] * x.shape[1]) ** self.config.normalize_size * (
                            x.shape[2] ** self.config.normalize_dim) / (x ** 2).sum(axis=(0, 1, 2)))
            else:
                x = x * ((x.shape[0] * x.shape[1]) ** self.config.normalize_size) * (
                            x.shape[2] ** self.config.normalize_dim) / (
                        (np.abs(x) ** (1. / self.config.normalize_power)).sum(axis=(0, 1, 2)))

        if self.config.square_root_normalization:
            x = np.sign(x) * np.sqrt(np.abs(x))
        return x.astype(np.float32)

    def integralVecImage(self, img):
        w, h, c = img.shape
        intImage = np.zeros((w + 1, h + 1, c), dtype=img.dtype)
        intImage[1:, 1:, :] = np.cumsum(np.cumsum(img, 0), 1)
        return intImage

    def average_feature_region(self, features, region_size):
        region_area = region_size ** 2
        if features.dtype == np.float32:
            maxval = 1.
        else:
            maxval = 255
        intImage = self.integralVecImage(features)
        i1 = np.arange(region_size, features.shape[0] + 1, region_size).reshape(-1, 1)
        i2 = np.arange(region_size, features.shape[1] + 1, region_size).reshape(1, -1)
        region_image = (intImage[i1, i2, :] - intImage[i1, i2 - region_size, :] - intImage[i1 - region_size, i2,
                                                                                  :] + intImage[i1 - region_size,
                                                                                       i2 - region_size, :]) / (
                                   region_area * maxval)
        return region_image

    def get_features(self, img, pos, sample_sz, scales, normalization=True):
        feat = []
        if not isinstance(scales, list) and not isinstance(scales, np.ndarray):
            scales = [scales]
        for scale in scales:
            patch = self._sample_patch(img, pos, sample_sz * scale, sample_sz)
            h, w, c = patch.shape
            if c == 3:
                RR = patch[:, :, 0].astype(np.int32)
                GG = patch[:, :, 1].astype(np.int32)
                BB = patch[:, :, 2].astype(np.int32)
                index = RR // self._den + (GG // self._den) * self._factor + (
                            BB // self._den) * self._factor * self._factor
                features = self._table[index.flatten()].reshape((h, w, self._table.shape[1]))
            else:
                features = self._table[patch.flatten()].reshape((h, w, self._table.shape[1]))
            if self._cell_size > 1:
                features = self.average_feature_region(features, self._cell_size)
            feat.append(features)
        feat = np.stack(feat, axis=3)
        if normalization is True:
            feat = self._feature_normalization(feat)
        return [feat]


def extract_cn_feature(image, cell_size=1):
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255 - 0.5
    cn = TableFeature(fname='cn', cell_size=cell_size, compressed_dim=11, table_name="CNnorm",
                      use_for_color=True)

    if np.all(img[:, :, 0] == img[:, :, 1]):
        img = img[:, :, :1]
    else:
        # # pyECO using RGB format
        img = img[:, :, ::-1]
    h, w = img.shape[:2]
    cn_feature = \
        cn.get_features(img, np.array(np.array([h / 2, w / 2]), dtype=np.int16), np.array([h, w]), 1,
                        normalization=False)[
            0][:, :, :, 0]
    # print('cn_feature.shape:', cn_feature.shape)
    # print('cnfeature:',cn_feature.shape,cn_feature.min(),cn_feature.max())
    gray = cv2.resize(gray, (cn_feature.shape[1], cn_feature.shape[0]))[:, :, np.newaxis]
    out = np.concatenate((gray, cn_feature), axis=2)
    return out


class BaseCF:
    def __init__(self):
        raise NotImplementedError

    def init(self, first_frame, bbox):
        raise NotImplementedError

    def update(self, current_frame):
        raise NotImplementedError


class KCF(BaseCF):
    def __init__(self, padding=1.5, features='gray', kernel='gaussian', mask=None, par=None):
        super(KCF).__init__()
        self.padding = padding
        self.lambda_ = 1e-4
        self.features = features
        self.w2c = None
        if self.features == 'hog':
            self.interp_factor = 0.02
            self.sigma = 0.5
            self.cell_size = 4
            self.output_sigma_factor = 0.1
        elif self.features == 'gray' or self.features == 'color' or self.features == 'rgb':
            self.interp_factor = 0.075
            self.sigma = 0.2
            self.cell_size = 1
            self.output_sigma_factor = 0.1
        elif self.features == 'cn':
            self.interp_factor = 0.075
            self.sigma = 0.2
            self.cell_size = 1
            self.output_sigma_factor = 1. / 16
            self.padding = 1
        else:
            raise NotImplementedError
        self.kernel = kernel
        self.mask = mask

    def init(self, first_frame, bbox, scale=1):
        # w,h in bbox are scaled
        assert len(first_frame.shape) == 3 and first_frame.shape[2] == 3
        if self.features == 'gray':
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        bbox = np.array(bbox).astype(np.int64)
        x0, y0, w, h = tuple(bbox)
        self.crop_size = (int(np.floor(w * (1 + self.padding))), int(np.floor(h * (1 + self.padding))))  # for vis
        self._center = (np.floor(x0 + w / 2), np.floor(y0 + h / 2))
        self.w, self.h = w, h
        self.window_size = (int(np.floor(w * (1 + self.padding))) // self.cell_size,
                            int(np.floor(h * (1 + self.padding))) // self.cell_size)
        self._window = cos_window(self.window_size)

        s = np.sqrt(w * h) * self.output_sigma_factor / self.cell_size
        self.yf = fft2(gaussian2d_rolled_labels(self.window_size, s))

        if self.features == 'gray' or self.features == 'color' or self.features == 'rgb':
            first_frame = first_frame.astype(np.float32) / 255
            x = self._crop(first_frame, self._center, (w, h), scale)
            if (self.features != 'rgb'):
                x = x - np.mean(x)
        elif self.features == 'hog':
            x = self._crop(first_frame, self._center, (w, h), scale)
            x = cv2.resize(x, (self.window_size[0] * self.cell_size, self.window_size[1] * self.cell_size))
            x = extract_hog_feature(x, cell_size=self.cell_size)
        elif self.features == 'cn':
            x = cv2.resize(first_frame, (self.window_size[0] * self.cell_size, self.window_size[1] * self.cell_size))
            x = extract_cn_feature(x, self.cell_size)
        else:
            raise NotImplementedError
        if (self.mask is not None):
            self.mask = self._crop(self.mask.astype(np.float32), self._center, (w, h), scale)
            self.mask = cv2.resize(self.mask, (self.window_size[0], self.window_size[1]))
            if (len(self.mask.shape) < 3):
                self.mask = np.expand_dims(self.mask, axis=-1)
            self.mask = np.tile(self.mask, [1, 1, x.shape[-1]])
            x = x * self.mask
        self.xt_raw = x
        self.xf = fft2(self._get_windowed(x, self._window))
        self.init_response_center = (0, 0)
        self.alphaf = self._training(self.xf, self.yf)

        ## add
        self.max_response = 1

    def update(self, current_frame, vis=False, update_flag=True):
        assert len(current_frame.shape) == 3 and current_frame.shape[2] == 3
        if self.features == 'gray':
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        if self.features == 'color' or self.features == 'gray' or self.features == 'rgb':
            current_frame = current_frame.astype(np.float32) / 255
            z = self._crop(current_frame, self._center, (self.w, self.h))
            if (self.features != 'rgb'):
                z = z - np.mean(z)
        elif self.features == 'hog':
            z = self._crop(current_frame, self._center, (self.w, self.h))
            z = cv2.resize(z, (self.window_size[0] * self.cell_size, self.window_size[1] * self.cell_size))
            z = extract_hog_feature(z, cell_size=self.cell_size)
        elif self.features == 'cn':
            z = self._crop(current_frame, self._center, (self.w, self.h))
            z = cv2.resize(z, (self.window_size[0] * self.cell_size, self.window_size[1] * self.cell_size))
            z = extract_cn_feature(z, cell_size=self.cell_size)
        else:
            raise NotImplementedError
        if (self.mask is not None):
            z = z * self.mask

        zf = fft2(self._get_windowed(z, self._window))
        responses = self._detection(self.alphaf, self.xf, zf, kernel=self.kernel)
        ## add
        self.max_response = np.max(responses)
        if vis is True:
            self.score = responses
            self.score = np.roll(self.score, int(np.floor(self.score.shape[0] / 2)), axis=0)
            self.score = np.roll(self.score, int(np.floor(self.score.shape[1] / 2)), axis=1)

        curr = np.unravel_index(np.argmax(responses, axis=None), responses.shape)

        if curr[0] + 1 > self.window_size[1] / 2:
            dy = curr[0] - self.window_size[1]
        else:
            dy = curr[0]
        if curr[1] + 1 > self.window_size[0] / 2:
            dx = curr[1] - self.window_size[0]
        else:
            dx = curr[1]
        dy, dx = dy * self.cell_size, dx * self.cell_size
        x_c, y_c = self._center
        x_c += dx
        y_c += dy
        self._center = (np.floor(x_c), np.floor(y_c))

        if (update_flag):
            if self.features == 'color' or self.features == 'gray' or self.features == 'rgb':
                new_x = self._crop(current_frame, self._center, (self.w, self.h))
            elif self.features == 'hog':
                new_x = self._crop(current_frame, self._center, (self.w, self.h))
                new_x = cv2.resize(new_x, (self.window_size[0] * self.cell_size, self.window_size[1] * self.cell_size))
                new_x = extract_hog_feature(new_x, cell_size=self.cell_size)
            elif self.features == 'cn':
                new_x = self._crop(current_frame, self._center, (self.w, self.h))
                new_x = cv2.resize(new_x, (self.window_size[0] * self.cell_size, self.window_size[1] * self.cell_size))
                new_x = extract_cn_feature(new_x, cell_size=self.cell_size)
            else:
                raise NotImplementedError
            # add
            self.xt = self._get_windowed(new_x, self._window)
            new_xf = fft2(self.xt)
            self.alphaf = self.interp_factor * self._training(new_xf, self.yf, kernel=self.kernel) + (
                        1 - self.interp_factor) * self.alphaf
            self.xf = self.interp_factor * new_xf + (1 - self.interp_factor) * self.xf
        return [(self._center[0] - self.w / 2), (self._center[1] - self.h / 2), self.w, self.h]

    def update_w_scale(self, current_frame, scale=1, vis=False, update_flag=True):
        assert len(current_frame.shape) == 3 and current_frame.shape[2] == 3

        w, h = self.w * scale, self.h * scale

        if self.features == 'gray':
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        if self.features == 'color' or self.features == 'gray' or self.features == 'rgb':
            current_frame = current_frame.astype(np.float32) / 255
            z = self._crop(current_frame, self._center, (w, h))
            z = cv2.resize(z, (self.window_size[0] * self.cell_size, self.window_size[1] * self.cell_size))
            if (self.features != 'rgb'):
                z = z - np.mean(z)
        elif self.features == 'hog':
            z = self._crop(current_frame, self._center, (w, h))
            z = cv2.resize(z, (self.window_size[0] * self.cell_size, self.window_size[1] * self.cell_size))
            z = extract_hog_feature(z, cell_size=self.cell_size)
        elif self.features == 'cn':
            z = self._crop(current_frame, self._center, (w, h))
            z = cv2.resize(z, (self.window_size[0] * self.cell_size, self.window_size[1] * self.cell_size))
            z = extract_cn_feature(z, cell_size=self.cell_size)
        else:
            raise NotImplementedError

        zf = fft2(self._get_windowed(z, self._window))
        responses = self._detection(self.alphaf, self.xf, zf, kernel=self.kernel)
        ## add
        self.max_response = np.max(responses)
        if vis is True:
            self.score = responses
            self.score = np.roll(self.score, int(np.floor(self.score.shape[0] / 2)), axis=0)
            self.score = np.roll(self.score, int(np.floor(self.score.shape[1] / 2)), axis=1)

        curr = np.unravel_index(np.argmax(responses, axis=None), responses.shape)

        if curr[0] + 1 > self.window_size[1] / 2:
            dy = curr[0] - self.window_size[1]
        else:
            dy = curr[0]
        if curr[1] + 1 > self.window_size[0] / 2:
            dx = curr[1] - self.window_size[0]
        else:
            dx = curr[1]
        dy, dx = dy * self.cell_size * scale, dx * self.cell_size * scale
        x_c, y_c = self._center
        x_c += dx
        y_c += dy
        self._center = (np.floor(x_c), np.floor(y_c))

        if (update_flag):
            if self.features == 'color' or self.features == 'gray' or self.features == 'rgb':
                new_x = self._crop(current_frame, self._center, (self.w, self.h))
            elif self.features == 'hog':
                new_x = self._crop(current_frame, self._center, (self.w, self.h))
                new_x = cv2.resize(new_x, (self.window_size[0] * self.cell_size, self.window_size[1] * self.cell_size))
                new_x = extract_hog_feature(new_x, cell_size=self.cell_size)
            elif self.features == 'cn':
                new_x = self._crop(current_frame, self._center, (self.w, self.h))
                new_x = cv2.resize(new_x, (self.window_size[0] * self.cell_size, self.window_size[1] * self.cell_size))
                new_x = extract_cn_feature(new_x, cell_size=self.cell_size)
            else:
                raise NotImplementedError
            # add
            self.xt = self._get_windowed(new_x, self._window)
            new_xf = fft2(self.xt)
            self.alphaf = self.interp_factor * self._training(new_xf, self.yf, kernel=self.kernel) + (
                        1 - self.interp_factor) * self.alphaf
            self.xf = self.interp_factor * new_xf + (1 - self.interp_factor) * self.xf
        return [(self._center[0] - w / 2), (self._center[1] - h / 2), w, h]

    def update_template(self, current_frame, bbox):
        assert len(current_frame.shape) == 3 and current_frame.shape[2] == 3

        x_c, y_c, w, h = convert_bbox_format(bbox, 'center')
        self._center = (np.floor(x_c), np.floor(y_c))

        if self.features == 'color' or self.features == 'gray' or self.features == 'rgb':
            new_x = self._crop(current_frame, self._center, (w, h))
            new_x = cv2.resize(new_x, (self.window_size[0] * self.cell_size, self.window_size[1] * self.cell_size))
        elif self.features == 'hog':
            new_x = self._crop(current_frame, self._center, (w, h))
            new_x = cv2.resize(new_x, (self.window_size[0] * self.cell_size, self.window_size[1] * self.cell_size))
            new_x = extract_hog_feature(new_x, cell_size=self.cell_size)
        elif self.features == 'cn':
            new_x = self._crop(current_frame, self._center, (w, h))
            new_x = cv2.resize(new_x, (self.window_size[0] * self.cell_size, self.window_size[1] * self.cell_size))
            new_x = extract_cn_feature(new_x, cell_size=self.cell_size)
        else:
            raise NotImplementedError
        # add
        self.xt = self._get_windowed(new_x, self._window)
        new_xf = fft2(self.xt)
        self.alphaf = self.interp_factor * self._training(new_xf, self.yf, kernel=self.kernel) + (
                    1 - self.interp_factor) * self.alphaf
        self.xf = self.interp_factor * new_xf + (1 - self.interp_factor) * self.xf
        return

    def _kernel_correlation(self, xf, yf, kernel='gaussian'):
        if kernel == 'gaussian':
            N = xf.shape[0] * xf.shape[1]
            xx = (np.dot(xf.flatten().conj().T, xf.flatten()) / N)
            yy = (np.dot(yf.flatten().conj().T, yf.flatten()) / N)
            xyf = xf * np.conj(yf)
            xy = np.sum(np.real(ifft2(xyf)), axis=2)
            kf = fft2(np.exp(-1 / self.sigma ** 2 * np.clip(xx + yy - 2 * xy, a_min=0, a_max=None) / np.size(xf)))
        elif kernel == 'linear':
            kf = np.sum(xf * np.conj(yf), axis=2) / np.size(xf)
        else:
            raise NotImplementedError
        return kf

    def _training(self, xf, yf, kernel='gaussian'):
        kf = self._kernel_correlation(xf, xf, kernel)
        alphaf = yf / (kf + self.lambda_)
        return alphaf

    def _detection(self, alphaf, xf, zf, kernel='gaussian'):
        kzf = self._kernel_correlation(zf, xf, kernel)
        responses = np.real(ifft2(alphaf * kzf))
        return responses

    def _crop(self, img, center, target_sz, scale=1):
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        w, h = target_sz

        cropped = cv2.getRectSubPix(img, (
        int(np.floor((1 + self.padding) * w / scale)), int(np.floor((1 + self.padding) * h / scale))), center)
        if (scale != 1):
            cropped = cv2.resize(cropped,
                                 (int(np.floor((1 + self.padding) * w)), int(np.floor((1 + self.padding) * h))))
        return cropped

    def _get_windowed(self, img, cos_window):
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        windowed = cos_window[:, :, None] * img
        return windowed


def tracking_kcf(init_bb, frames):
    poses = []
    init_frame = cv2.imread(frames[0])
    # print(init_frame.shape)
    init_gt = np.array(init_bb)
    poses.append(init_gt)

    # initial subspace
    if len(init_frame.shape) == 2:
        init_frame = cv2.cvtColor(init_frame, cv2.COLOR_GRAY2BGR)

    init_gt = tuple(init_gt)
    tracker = KCF(features='hog', kernel='gaussian')
    tracker.init(init_frame, init_gt)

    for idx in range(len(frames)):
        if idx != 0:
            current_frame = cv2.imread(frames[idx])
            bbox = tracker.update(current_frame)

            poses.append(bbox)
    return np.array(poses)
