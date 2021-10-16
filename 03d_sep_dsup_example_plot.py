#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
This module plots a dual scatterplot example with 4 different cases for
separability and discriminative support:

python 03d_sep_dsup_example_plot.py FIG_LEGEND_POS=1.01
   SAVEFIG_PATH=umap_plots/sep_dsup_example.png
"""


import os
import pickle
import json
#
from omegaconf import OmegaConf
import numpy as np
from randomcolor import RandomColor
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import PathCollection
#
from d2021umaps.data import ANOMALY_LABELS, get_umap_energies
from d2021umaps.plots import shadow_dot_scatter
from d2021umaps.plots import MulticolorCircles, MulticolorHandler

# ##############################################################################
# # GLOBALS
# ##############################################################################
CONF = OmegaConf.create()
# plot conf
CONF.SHADOW_SIZE = 30
CONF.SHADOW_ALPHA = 0.04
CONF.DOT_SHAPE = "o"
CONF.DOT_SIZE = 10
CONF.SHADOW_COLOR = "gray"
CONF.NORMAL_COLORS = ["cornflowerblue", "steelblue", "mediumblue", "darkblue"]
CONF.ANOMALY_COLORS = ["lightcoral", "indianred", "orangered", "firebrick"]

CONF.FIGSIZE = (20, 10)
CONF.DPI = 300
CONF.SAVEFIG_PATH = None
CONF.BG_COLOR = (0.975, 0.985, 1)  # rgb
#
CONF.LEGEND_SHADOW_ALPHA = 0.5
CONF.LEGEND_SHADOW_RATIO = 2
CONF.LEGEND_DOT_RATIO = 1
CONF.LEGEND_WIDTH_FACTOR = 0.8
CONF.LEGEND_FONT_SIZE = 50
#
CONF.CUT_TOP = 0.07
CONF.CUT_BOTTOM = 0.07
CONF.CUT_LEFT = 0.35
CONF.CUT_RIGHT = 0.0
#
CONF.FIG_MARGIN_RIGHT = 0.5
CONF.FIG_LEGEND_POS = None  # 0.65
CONF.LEGEND_ICON_SIZE = None

cli_conf = OmegaConf.from_cli()
CONF = OmegaConf.merge(CONF, cli_conf)

FIG_BBOX = (CONF.FIG_LEGEND_POS, 1) if CONF.FIG_LEGEND_POS is not None else None
FIG_LOC = "upper left" if FIG_BBOX is not None else None

print(OmegaConf.to_yaml(CONF))


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
fig, (ax_l, ax_r) = plt.subplots(ncols=2, sharex=True, sharey=True,
                                 figsize=CONF.FIGSIZE)
# remove axis ticks, set background color
plt.setp(ax_l, xticks=[], yticks=[])
ax_l.set_facecolor(CONF.BG_COLOR)
plt.setp(ax_r, xticks=[], yticks=[])
ax_r.set_facecolor(CONF.BG_COLOR)


TRAIN_N = 50_000
DOT_N = 1000
DOT_VAR = 0.08
TRAIN_VAR = 8.2
#
NORM1_MEAN = (3, 7)
NORM2_MEAN = (3, 5)
NORM3_MEAN = (5, 3)
NORM4_MEAN = (5, 1)
ANOM1_MEAN = (5, 7)
ANOM2_MEAN = (3, 5)
ANOM3_MEAN = (5, 3)
ANOM4_MEAN = (3, 1)

#

train_y = np.random.rand(TRAIN_N) * TRAIN_VAR
train_x = (np.exp(np.random.rand(TRAIN_N)) - 1) * TRAIN_VAR / (np.e - 1) / 2
train = np.stack([train_x, train_y]).T


norm1 = np.random.multivariate_normal(NORM1_MEAN, np.eye(2)*DOT_VAR, DOT_N)
norm2 = np.random.multivariate_normal(NORM2_MEAN, np.eye(2)*DOT_VAR, DOT_N)
norm3 = np.random.multivariate_normal(NORM3_MEAN, np.eye(2)*DOT_VAR, DOT_N)
norm4 = np.random.multivariate_normal(NORM4_MEAN, np.eye(2)*DOT_VAR, DOT_N)
anom1 = np.random.multivariate_normal(ANOM1_MEAN, np.eye(2)*DOT_VAR, DOT_N)
anom2 = np.random.multivariate_normal(ANOM2_MEAN, np.eye(2)*DOT_VAR, DOT_N)
anom3 = np.random.multivariate_normal(ANOM3_MEAN, np.eye(2)*DOT_VAR, DOT_N)
anom4 = np.random.multivariate_normal(ANOM4_MEAN, np.eye(2)*DOT_VAR, DOT_N)

all_norm_source = np.zeros((100, 2))
all_anom_source = np.zeros((100, 2))
all_source_train_c = ["green"]
all_norm_source_c = ["blue"]
all_anom_source_c = ["red"]


shadow_dot_scatter(ax_l, [], [], train, [],
                   shadows_alpha=CONF.SHADOW_ALPHA,
                   shadows_surface=CONF.SHADOW_SIZE,
                   shadows_low_c=CONF.SHADOW_COLOR)
shadow_dot_scatter(ax_r, [], [], train, [],
                   shadows_alpha=CONF.SHADOW_ALPHA,
                   shadows_surface=CONF.SHADOW_SIZE,
                   shadows_low_c=CONF.SHADOW_COLOR)


for i, (n, a) in enumerate(zip([norm1, norm2, norm3, norm4],
                               [anom1, anom2, anom3, anom4])):
    shadow_dot_scatter(ax_l, n, [], [], [],
                   dots_surface=CONF.DOT_SIZE,
                   dots_low_shape=CONF.DOT_SHAPE,
                   shadows_low_c=CONF.SHADOW_COLOR,
                   dots_low_c=CONF.NORMAL_COLORS[i])
    shadow_dot_scatter(ax_r, a, [], [], [],
                   dots_surface=CONF.DOT_SIZE,
                   dots_low_shape=CONF.DOT_SHAPE,
                   shadows_low_c=CONF.SHADOW_COLOR,
                   dots_low_c=CONF.ANOMALY_COLORS[i])

# https://matplotlib.org/stable/api/legend_handler_api.html
mc_handler = MulticolorHandler(width_factor=CONF.LEGEND_WIDTH_FACTOR)

# Right plot legend
ax_r_handles, ax_r_labels = ax_r.get_legend_handles_labels()
ax_r_labels.append("Training data")
ax_r_handles.append(MulticolorCircles(
    ["none", "none", CONF.SHADOW_COLOR, "none"],
    radius_factor=CONF.LEGEND_SHADOW_RATIO,
    face_alpha=CONF.LEGEND_SHADOW_ALPHA))
ax_r_labels.append("Test/normal data")
ax_r_handles.append(MulticolorCircles(CONF.NORMAL_COLORS,
                                      radius_factor=CONF.LEGEND_DOT_RATIO))
ax_r_labels.append("Test/anomalous data")
ax_r_handles.append(MulticolorCircles(CONF.ANOMALY_COLORS,
                                      radius_factor=CONF.LEGEND_DOT_RATIO))
#
ax_r.legend(ax_r_handles, ax_r_labels, handlelength=CONF.LEGEND_ICON_SIZE,
            borderpad=0.5, labelspacing=0.8,
            bbox_to_anchor=FIG_BBOX,
            loc=FIG_LOC,
            prop={'size': CONF.LEGEND_FONT_SIZE},
            handler_map={MulticolorCircles: mc_handler})

ax_l.set_aspect("equal")
ax_r.set_aspect("equal")

# trim view
left, right = ax_l.get_xlim()
down, top = ax_l.get_ylim()
horiz_dist = right - left
vert_dist = top - down
#
left += horiz_dist * CONF.CUT_LEFT
right -= horiz_dist * CONF.CUT_RIGHT
top -= vert_dist * CONF.CUT_TOP
down += vert_dist * CONF.CUT_BOTTOM
#
ax_l.set_ylim(down, top)
ax_l.set_xlim(left, right)

fig.subplots_adjust(top=0.99, bottom=0.01, left=0.01,
                    right=CONF.FIG_MARGIN_RIGHT,
                    hspace=0, wspace=0.05)

if CONF.SAVEFIG_PATH is not None:
    fig.savefig(CONF.SAVEFIG_PATH, dpi=CONF.DPI)
    print("Saved plot to", CONF.SAVEFIG_PATH)
else:
    fig.show()
    breakpoint()
