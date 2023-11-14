# v2021.11.23

import numpy as np


def sparse_sample(X, pattern):
    # X: [...,N,N]
    # pattern: 'diamond','cross','ten'
    n, h, w, c, N, _ = X.shape
    X = X.reshape(n, h, w, c, N * N)
    idx = get_pattern_idx(N, pattern)
    # print(pattern, idx)
    X = X[..., idx]
    X = X.reshape(n, h, w, -1)
    return X


def get_pattern_idx(N, pattern):
    if pattern == 'diamond':
        idx = pattern_diamond(N)
    elif pattern == 'cross':
        idx = pattern_cross(N)
    else:
        idx = pattern_ten(N)
    return idx


def pattern_diamond(N):
    radius = (N - 1) // 2
    coord = [(i, radius - i) for i in range(radius)]
    coord.extend([(radius + i, i) for i in range(radius)])
    coord.extend([(N - 1 - i, radius + i) for i in range(radius)])
    coord.extend([(radius - i, N - 1 - i) for i in range(radius)])
    coord.append((radius, radius))
    return [i * N + j for i, j in coord]


def pattern_cross(N):
    radius = (N - 1) // 2
    coord = [(i, i) for i in range(N)]
    coord.extend((N - 1 - i, i) for i in range(N) if i != radius)
    return [i * N + j for i, j in coord]


def pattern_ten(N):
    radius = (N - 1) // 2
    coord = [(radius, i) for i in range(N)]
    coord.extend((i, radius) for i in range(N) if i != radius)
    return [i * N + j for i, j in coord]


if __name__ == "__main__":
    X = np.random.rand(100, 24, 24, 3, 7, 7)
    for pattern in ['diamond', 'cross', 'ten']:
        out = sparse_sample(X, pattern)
        print(X.shape, out.shape)
