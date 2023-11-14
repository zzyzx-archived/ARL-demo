import numpy as np
import xgboost as xgb
from skimage.util.shape import view_as_windows


class BinaryClassifier(object):
    def __init__(self, n_estimators=30, learning_rate=0.1, max_depth=5):
        # "silent": True, or "verbosity": 1
        self.boosting_params = {"verbosity": 0,
                                "objective": "binary:logistic",
                                "eta": learning_rate,
                                "max_depth": max_depth}
        self.n_estimators = n_estimators

    def train(self, Xpos, Xneg, flag_balance, verbose=False):
        Xpos = np.array(Xpos)
        Xneg = np.array(Xneg)
        if verbose:
            print('raw input stats: pos', Xpos.shape, ', neg', Xneg.shape)
        if flag_balance == 'inc':
            Xpos, Xneg = balance_sample_inc(Xpos, Xneg)
        elif flag_balance == 'dec':
            Xpos, Xneg = balance_sample_dec(Xpos, Xneg)
        if verbose:
            print('processed input stats: pos', Xpos.shape, ', neg', Xneg.shape)
        X = np.concatenate((Xpos, Xneg), axis=0)
        y = np.ones((len(Xpos) + len(Xneg),), dtype=int)
        y[-len(Xneg):] = 0

        self.clf = train_booster(X, y, self.boosting_params, self.n_estimators)
        return self.clf

    def predict(self, X, output_prob=False):
        return test_booster(X, self.clf, output_prob)


def train_booster(X, y, boosting_params, n_estimators):
    d_train = xgb.DMatrix(X, y.astype(int))

    booster = xgb.train(params=boosting_params,
                        dtrain=d_train,
                        num_boost_round=n_estimators,
                        verbose_eval=False)
    return booster


def test_booster(X, booster, output_prob=False):
    d_test = xgb.DMatrix(X)
    predicted = booster.predict(d_test)
    if output_prob:
        return np.array(predicted)
    else:
        return np.array(predicted > 0.5).astype(int)


def balance_sample_dec(A, B, maxlen=2000):
    np.random.seed(2020)
    length = min(len(A), len(B))
    if (len(A) > length):
        idx = np.random.choice(len(A), length, replace=False)
        A = A[idx]
    if (len(B) > length):
        idx = np.random.choice(len(B), length, replace=False)
        B = B[idx]
    if (length > maxlen):
        idx = np.random.choice(length, maxlen, replace=False)
        A = A[idx]
        B = B[idx]
    return A, B


def balance_sample_inc(A, B, maxlen=2000):
    np.random.seed(2020)
    length = min(maxlen, max(len(A), len(B)))
    if (len(A) < length):
        idx = np.random.choice(len(A), length - len(A), replace=True)
        A = np.concatenate((A, A[idx]), axis=0)
    elif len(A) > length:
        idx = np.random.choice(len(A), length, replace=False)
        A = A[idx]

    if (len(B) < length):
        idx = np.random.choice(len(B), length - len(B), replace=True)
        B = np.concatenate((B, B[idx]), axis=0)
    elif len(B) > length:
        idx = np.random.choice(len(B), length, replace=False)
        B = B[idx]
    return A, B


def patch2sample(patch, sample_size, stride, box=None, verbose=False):
    h, w, c = patch.shape
    mask = box_fill(h, w, box)
    labels = view_as_windows(mask, (sample_size, sample_size), step=(stride, stride))
    if verbose:
        print('shape after window:', labels.shape)  # [n1,n2,sample_size,sample_size]
    labels = labels.reshape(-1, sample_size * sample_size)
    labels = labels.mean(axis=-1)
    if verbose:
        print('labels shape:', labels.shape)
    # print(labels)

    samples = view_as_windows(patch, (sample_size, sample_size, c), step=(stride, stride, c))
    if verbose:
        print('shape after window:', samples.shape)  # [n1,n2,1,sample_size,sample_size,c]
    samples = samples.reshape(-1, sample_size, sample_size, c)
    if verbose:
        print('samples shape:', samples.shape)
    return samples, labels


def pred2prob(pred, h, w, sample_size, stride):
    mask = np.zeros((h, w), dtype=np.float)
    cnt = np.zeros((h, w), dtype=np.float)
    num = 0
    for i in range(0, h - sample_size + 1, stride):
        for j in range(0, w - sample_size + 1, stride):
            # print(pred[num], np.mean(mask))
            mask[i:i + sample_size, j:j + sample_size] += pred[num]
            cnt[i:i + sample_size, j:j + sample_size] += 1
            num += 1
    return mask


def box_fill(h, w, box):
    xmin, ymin, bw, bh = box
    xmax = xmin + bw - 1
    ymax = ymin + bh - 1
    mask = np.ones((h, w), dtype=np.float)
    x = np.arange(w)
    y = np.arange(h)
    mask[:, x < xmin] = 0
    mask[:, x > xmax] = 0
    mask[y < ymin, :] = 0
    mask[y > ymax, :] = 0
    return mask


if __name__ == '__main__':
    ary = np.random.rand(32, 32, 3)
    patch2sample(ary, 8, 4, [8, 4, 10, 15])

    pos = np.random.rand(110, 150)
    neg = np.random.rand(100, 150)
    y = np.zeros(len(pos) + len(neg))
    y[:len(pos)] = 1
    clf = BinaryClassifier(n_estimators=10, learning_rate=0.1, max_depth=5)
    clf.train(Xpos=pos, Xneg=neg, flag_balance='', verbose=False)
    pred = clf.predict(np.concatenate((pos, neg), axis=0), output_prob=False)
    print('training acc:', np.sum(pred == y) / len(y))
