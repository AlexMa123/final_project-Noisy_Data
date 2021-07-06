import matplotlib.pyplot as plt
from math import ceil


def plot_one_sample(X, Y, Y_pred=None, ax=None):
    assert ax is not None
    X = X - X.mean()
    X = X / X.std()
    ax.imshow(X.reshape(20, 20).T, cmap="Greys")
    ax.set_xticks([])
    ax.set_yticks([])
    if Y_pred is None:
        ax.set_title(f"{int(Y)}")
    else:
        ax.set_title(f"True: {int(Y)}, Predicted: {int(Y_pred)}")


def plot_samples(X, Y, Y_pred=None, nrows=1, figsize=(8, 6)):
    """Plot samples

    Parameters
    ----------
    X : Images, Tensor or Array
        (N, 400)
    Y : True label
        (N, )
    Y_pred : predicted Label, optional
        (N, ), by default None
    nrows : int, optional
        num of nrows for visualize the images, by default 1
    figsize : tuple, optional
        figure size, by default (8, 6)

    Returns
    -------
    fig, ax
        the same with what you get from plt.subplots()
    """
    images_shape = X.size()
    if len(images_shape) == 2:
        total_image = images_shape[0]
    else:
        total_image = 1
    ncolumes = ceil(total_image / nrows)
    fig, axes = plt.subplots(nrows, ncolumes, figsize=figsize)
    for i in range(nrows):
        for j in range(ncolumes):
            if nrows == ncolumes == 1:
                ax = axes
            elif nrows == 1 or ncolumes == 1:
                ax = axes[max(i, j)]
            else:
                ax = axes[i, j]
            index = i * nrows + j
            if index >= total_image:
                ax.axis('off')
                continue
            if Y_pred is None:
                plot_one_sample(X[index], Y[index] % 10, ax=ax)
            else:
                plot_one_sample(X[index], Y[index] % 10, Y_pred[index], ax=ax)
    return fig, axes
