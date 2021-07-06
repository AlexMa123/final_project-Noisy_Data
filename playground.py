import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tools.plots import plot_samples

if __name__ == '__main__':
    data = loadmat("trg1.mat")
    X, Y = data['X'], data['y']
    X = torch.Tensor(X)
    Y = torch.Tensor(Y)
    ind = np.random.randint(0, len(X), 6)
    fig, axes = plot_samples(X[ind], Y[ind], nrows=1)
    plt.show()
