#!/usr/bin python
# -*- coding:utf-8 -*-


"""
UMAP on the MNIST Digits dataset
--------------------------------

A simple example demonstrating how to use UMAP on a larger
dataset such as MNIST. We first pull the MNIST dataset and
then use UMAP to reduce it to only 2-dimensions for
easy visualisation.

Note that UMAP manages to both group the individual digit
classes, but also to retain the overall global structure
among the different digit classes -- keeping 1 far from
0, and grouping triplets of 3,5,8 and 4,7,9 which can
blend into one another in some cases.
"""


import os
#
import numpy as np
import umap
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns
#
from matplotlib.lines import Line2D


COLORS = sns.color_palette()
SURFACE = 1.0
LEGEND_ICON_SIZE = 17
LEGEND_FONT_SIZE = 20
BG_COLOR = (0.975, 0.985, 1)  # rgb
SAVEPATH = os.path.join("umap_plots", "mnist.png")
DPI = 300

sns.set(context="paper", style="white")
print("loading MNIST")
mnist = fetch_openml("mnist_784", version=1)

print("computing UMAP")
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(mnist.data)
labels = mnist.target.astype(int)
colors = [COLORS[i] for i in labels]

fig, ax = plt.subplots(figsize=(15, 6))
ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=SURFACE)
legend_elts = [Line2D([0], [0], marker="o", color=c, label=str(i),
                      markerfacecolor=c, markersize=LEGEND_ICON_SIZE,
                      linestyle="none")
               for i, c in enumerate(COLORS)]
ax.legend(handles=legend_elts, borderpad=0.4, labelspacing=0.5, prop={'size': LEGEND_FONT_SIZE})
plt.setp(ax, xticks=[], yticks=[])
ax.set_facecolor(BG_COLOR)

# hack to make more room for legend (assuming right position)
left_x, right_x = ax.get_xlim()
right_x += (right_x - left_x) * 0.1  # add this ratio to the right
ax.set_xlim(left_x, right_x)


plt.show()
fig.savefig(SAVEPATH, bbox_inches="tight", dpi=DPI)
breakpoint()
