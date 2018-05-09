import math

import numba
import numpy as np


@numba.jit(nopython=True)
def euclidean_sum(X, Y):
    """Distance between each pair of the two collections of dense trajectory descriptors.

    Parameters
    ----------
    X : ndarray, shape (n1, m, 2)
    Y : ndarray, shape (n2, m, 2)

    Returns
    -------
    D : ndarray, shape (n1, n2)
    """
    n1 = len(X)
    n2 = len(Y)

    dist = np.empty((n1, n2))
    for i in range(0, n1):
        for j in range(0, n2):
            dist[i, j] = sumeuclidean(X[i], Y[j])

    return dist


@numba.jit(nopython=True)
def euclidean_sum_pairwise(X):
    """
    Pairwise distance between each pair of the dense trajectory descriptors.
    """
    m, n, _ = X.shape
    dist = np.empty((m * (m-1)) // 2)

    k = 0
    for i in range(0, m-1):
        for j in range(i+1, m):
            sum = 0
            for p in range(n):
                d = euclid(X[i, p, 0], X[i, p, 1], X[j, p, 0], X[j, p, 1])
                sum = sum + d
            dist[k] = sum
            k = k + 1

    return dist


@numba.jit(nopython=True)
def sumeuclidean(X1, X2):
    """Distance between two dense trajectory descriptors.

    Parameters
    ----------
    X1 : ndarray, shape (n, 2)
    X2 : ndarray, shape (n, 2)
    """
    n = len(X1)

    s = 0
    for k in range(0, n):
        d = euclid(X1[k, 0], X1[k, 1], X2[k, 0], X2[k, 1])
        s = s + d
    s = s / n

    return s


@numba.jit(nopython=True)
def euclid(x1, y1, x2, y2):
    """Euclidean distance between two (x, y) points."""
    x = x1 - x2
    y = y1 - y2

    dist = x**2 + y**2
    dist = math.sqrt(dist)

    return dist
