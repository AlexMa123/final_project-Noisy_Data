import numpy as np
import random
import matplotlib.pyplot as plt
from numba import njit, prange


def powerlaw(N, alpha):
    x = np.random.rand(N)
    return x ** (- alpha)


def MLE(x, a=1):
    x = x[x > a]
    n = x.size
    mu = 1 - n / (n * np.log(a) - np.sum(np.log(x)))
    return mu


def MLE_error(alpha, N):
    return (alpha - 1) / np.sqrt(N + 1)


@njit
def one_step(x, p):
    random_number = random.random()
    if x == 0:
        if random_number > 0.5:
            return 1
        else:
            return -1
    if x > 0:
        if random_number > p:
            return -1
        else:
            return 1
    if x < 0:
        if random_number > p:
            return 1
        else:
            return -1

@njit
def random_waker(p=0.45, final_X=20):
    i = 0
    X = 0
    while X != final_X and i < 1e10:
        i = i + 1
        X = X + one_step(X, p)
    return i


@njit
def random_wakerN(p=0.45, N=10000):
    Xs = np.zeros(N)
    for i in range(N - 1):
        X = Xs[i]
        X = X + one_step(X, p)
        Xs[i + 1] = X
    return Xs


@njit(parallel=True)
def main(N=200, n_cpu=5, p=0.25):
    first_passage_time = np.zeros((n_cpu, N))
    for i in prange(n_cpu):
        for j in range(N):
            first_passage_time[i, j] = random_waker(p)
    return first_passage_time


if __name__ == '__main__':
    first_passage_time = main(N=10, n_cpu=6, p=0.2)
    first_passage_time = first_passage_time.reshape(-1, 1)
    print(first_passage_time)
    # trace = random_wakerN(0.4, 10000000)
    plt.figure()
    plt.hist(first_passage_time, bins=20)
    plt.yscale("log")
    # plt.plot(trace)
    plt.show()
