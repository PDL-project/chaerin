"""
This script is to plot "Figure 6" in the paper
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import numpy as np


def tsplot(data, xlabel="steps", ylabel="metric", save=None, show=True, **kw):
    # data -> n-dims (averaged) x t
    sns.set(style="darkgrid", font_scale=1.5)

    fig, ax = plt.subplots(ncols=1)
    x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    cis = (est - sd, est + sd)
    ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)

    # labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.plot(x, est, **kw)
    ax.margins(x=0)

    if save is not None:
        plt.savefig(save, bbox_inches="tight")

    if show:
        plt.show()


if __name__ == "__main__":
    data = np.random.rand(100, 30)  # n-dims (averaged) x t
    tsplot(data)
