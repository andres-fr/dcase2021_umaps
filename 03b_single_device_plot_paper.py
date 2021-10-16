#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
Given a per-device UMAP, this script plots a dual scatterplot with
info about the external datasets, training/test data, source/target domain,
anomaly (right) vs. normal (left) sounds, and color-code by section+domain.
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
CONF.DEVICE_UMAP_PATH = os.path.join(
    "umaps",
    "UMAP_modality=mel_splits=gearbox_stack=5_maxDcaseTrain=10000_" +
    "maxDcaseTest=20000_maxAudioset=10000_maxFraunhofer=10000.pickle")
# plot conf
CONF.DCASE_SHADOW_SIZE = 30
CONF.EXTERNAL_SHADOW_SIZE = 30
CONF.SHADOW_ALPHA = 0.1
CONF.DOT_SHAPE = "."
CONF.DOT_SIZE = 12
CONF.AUDIOSET_COLOR = "grey"
CONF.FRAUNHOFER_COLOR = "violet"
CONF.SHADOW_COLORS = {"source": ["paleturquoise", "lightblue", "skyblue",
                                 "cornflowerblue", "steelblue", "mediumblue"],
                      "target": ["palegreen", "greenyellow", "lime",
                                 "limegreen", "forestgreen", "darkgreen"]}
CONF.DOT_COLORS = {"normal_source": ["mediumpurple", "mediumslateblue",
                                     "slateblue", "blueviolet",
                                     "darkslateblue", "darkviolet",
                                     "mediumblue"],
                   "normal_target": ["palegreen", "greenyellow", "lime",
                                     "limegreen", "forestgreen", "darkgreen"],
                   "anomaly_source": ["palegoldenrod", "khaki", "yellow",
                                      "gold", "goldenrod", "orange"],
                   "anomaly_target": ["lightsalmon", "lightcoral", "orangered",
                                      "indianred", "firebrick", "maroon"]}
CONF.PLOT_AUDIOSET = True
CONF.PLOT_FRAUNHOFER = True
CONF.AUDIOSET_LABEL = "Audioset"
CONF.FRAUNHOFER_LABEL = "Fraunhofer"  # IDMT-ISA-Electric-Engine
CONF.TRAIN_SOURCE_NORMAL_LABEL = "Train/source/normal"
CONF.TRAIN_TARGET_NORMAL_LABEL = "Train/target/normal"
CONF.TEST_SOURCE_NORMAL_LABEL = "Test/source/normal"
CONF.TEST_TARGET_NORMAL_LABEL = "Test/target/normal"
CONF.TEST_SOURCE_ANOMALY_LABEL = "Test/source/anomaly"
CONF.TEST_TARGET_ANOMALY_LABEL = "Test/target/anomaly"
CONF.FIGSIZE = (30, 15)
CONF.DPI = 450
CONF.SAVEFIG_PATH = None
CONF.BG_COLOR = (0.975, 0.985, 1)  # rgb
CONF.WITH_CROSS = False
CONF.CROSS_SIZE = 50
CONF.CROSS_SHAPE = "P"
CONF.CROSS_COLOR = "black"
CONF.CROSS_EXCLUDE_LOWEST = 500
CONF.CROSS_AVERAGE_N = 100
#
CONF.LEGEND_SHADOW_ALPHA = 0.5
CONF.LEGEND_DOT_EXTERNAL_RATIO = 1.7
CONF.LEGEND_DOT_TRAIN_RATIO = 1.7
CONF.LEGEND_DOT_TEST_RATIO = 1
CONF.LEGEND_WIDTH_FACTOR = 2.0
CONF.LEGEND_FONT_SIZE = 35
CONF.LEGEND_ICON_SIZE = None
#
CONF.CUT_TOP = 0.0
CONF.CUT_BOTTOM = 0.0
CONF.CUT_LEFT = 0.0
CONF.CUT_RIGHT = 0.0
#
CONF.FIG_MARGIN_RIGHT = 0.99
CONF.FIG_LEGEND_POS = None  # 0.65

cli_conf = OmegaConf.from_cli()
CONF = OmegaConf.merge(CONF, cli_conf)

FIG_BBOX = (CONF.FIG_LEGEND_POS, 1) if CONF.FIG_LEGEND_POS is not None else None
FIG_LOC = "upper left" if FIG_BBOX is not None else None

assert CONF.DEVICE is not None, "Please specify the device being plotted"
print(OmegaConf.to_yaml(CONF))


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
# Open UMAP file and create figure
with open(CONF.DEVICE_UMAP_PATH, "rb") as f:
    umap_data = pickle.load(f)
fig, (ax_l, ax_r) = plt.subplots(ncols=2, sharex=True, sharey=True,
                                 figsize=CONF.FIGSIZE)

# remove axis ticks, set background color
plt.setp(ax_l, xticks=[], yticks=[])
ax_l.set_facecolor(CONF.BG_COLOR)
plt.setp(ax_r, xticks=[], yticks=[])
ax_r.set_facecolor(CONF.BG_COLOR)


# try to add AudioSet umaps to both left and right plots
if CONF.PLOT_AUDIOSET:
    try:
        audioset_umaps = umap_data["audioset"]["umaps"]
        ax_l = shadow_dot_scatter(ax_l, [], [], audioset_umaps, [],
                                  shadows_alpha=CONF.SHADOW_ALPHA,
                                  shadows_surface=CONF.EXTERNAL_SHADOW_SIZE,
                                  shadows_low_c=CONF.AUDIOSET_COLOR)
        ax_r = shadow_dot_scatter(ax_r, [], [], audioset_umaps, [],
                                  shadows_alpha=CONF.SHADOW_ALPHA,
                                  shadows_surface=CONF.EXTERNAL_SHADOW_SIZE,
                                  shadows_low_c=CONF.AUDIOSET_COLOR)
    except Exception as e:
        print("Error plotting AudioSet UMAPs!", e)


# try to add Fraunhofer umaps to both left and right plots
if CONF.PLOT_FRAUNHOFER:
    try:
        fraunhofer_umaps = umap_data["fraunhofer"]["umaps"]
        ax_l = shadow_dot_scatter(ax_l, [], [], fraunhofer_umaps, [],
                                  shadows_alpha=CONF.SHADOW_ALPHA,
                                  shadows_surface=CONF.EXTERNAL_SHADOW_SIZE,
                                  shadows_low_c=CONF.FRAUNHOFER_COLOR)
        ax_r = shadow_dot_scatter(ax_r, [], [], fraunhofer_umaps, [],
                                  shadows_alpha=CONF.SHADOW_ALPHA,
                                  shadows_surface=CONF.EXTERNAL_SHADOW_SIZE,
                                  shadows_low_c=CONF.FRAUNHOFER_COLOR)
    except Exception as e:
        print("Error plotting Fraunhofer UMAPs!", e)


# Add the DCASE train source and train target entries to both L/R axes
train_source_umaps = {k: v for k, v in umap_data.items()
                      if k[0]=="train" and k[1]==CONF.DEVICE and k[3]=="source"}
train_target_umaps = {k: v for k, v in umap_data.items()
                      if k[0]=="train" and k[1]==CONF.DEVICE and k[3]=="target"}

# gather, shuffle and plot TRAIN/SOURCE data
all_source_train = []
all_source_train_c = []
for (_, dev, sec, dom), v in train_source_umaps.items():
    umaps = v["umaps"]
    color = CONF.SHADOW_COLORS["source"][int(sec)]
    all_source_train.append(umaps)
    all_source_train_c.append([color for _ in umaps])
all_source_train = np.concatenate(all_source_train)
all_source_train_c = np.concatenate(all_source_train_c)
perm = np.random.permutation(len(all_source_train))
all_source_train = all_source_train[perm]
all_source_train_c = all_source_train_c[perm]
# Plot left and right
ax_l.scatter(all_source_train[:, 0], all_source_train[:, 1],
             c=[mcolors.colorConverter.to_rgba(c, alpha=CONF.SHADOW_ALPHA)
                for c in all_source_train_c], s=CONF.DCASE_SHADOW_SIZE,
             edgecolors="none")
ax_r.scatter(all_source_train[:, 0], all_source_train[:, 1],
             c=[mcolors.colorConverter.to_rgba(c, alpha=CONF.SHADOW_ALPHA)
                for c in all_source_train_c], s=CONF.DCASE_SHADOW_SIZE,
             edgecolors="none")

# gather, shuffle and plot TRAIN/TARGET data
all_target_train = []
all_target_train_c = []
for (_, dev, sec, dom), v in train_target_umaps.items():
    umaps = v["umaps"]
    color = CONF.SHADOW_COLORS["target"][int(sec)]
    all_target_train.append(umaps)
    all_target_train_c.append([color for _ in umaps])
all_target_train = np.concatenate(all_target_train)
all_target_train_c = np.concatenate(all_target_train_c)
perm = np.random.permutation(len(all_target_train))
all_target_train = all_target_train[perm]
all_target_train_c = all_target_train_c[perm]
# Plot left and right
ax_l.scatter(all_target_train[:, 0], all_target_train[:, 1],
             c=[mcolors.colorConverter.to_rgba(c, alpha=CONF.SHADOW_ALPHA)
                for c in all_target_train_c], s=CONF.DCASE_SHADOW_SIZE,
             edgecolors="none")
ax_r.scatter(all_target_train[:, 0], all_target_train[:, 1],
             c=[mcolors.colorConverter.to_rgba(c, alpha=CONF.SHADOW_ALPHA)
                for c in all_target_train_c], s=CONF.DCASE_SHADOW_SIZE,
             edgecolors="none")


# TEST/SOURCE: plot normals on the left and anomalies on the right
test_source_umaps = {k: v for k, v in umap_data.items()
                     if k[0]=="test" and k[1]==CONF.DEVICE and k[3]=="source"}
all_norm_source = []
all_norm_source_c = []
all_anom_source = []
all_anom_source_c = []
for (_, dev, sec, dom), v in test_source_umaps.items():
    norm_color = CONF.DOT_COLORS["normal_source"][int(sec) + 2]
    anom_color = CONF.DOT_COLORS["anomaly_source"][int(sec) + 2]
    labels = np.array([ANOMALY_LABELS[json.loads(s)["label"]]
                       for s in v["metadata"]])
    norm_idxs = np.where(labels == 0)[0]
    anom_idxs = np.where(labels == 1)[0]
    norm_umaps = v["umaps"][norm_idxs]
    anom_umaps = v["umaps"][anom_idxs]
    #
    all_norm_source.append(norm_umaps)
    all_norm_source_c.append([norm_color for _ in norm_umaps])
    all_anom_source.append(anom_umaps)
    all_anom_source_c.append([anom_color for _ in anom_umaps])

all_norm_source = np.concatenate(all_norm_source)
all_norm_source_c = np.concatenate(all_norm_source_c)
perm = np.random.permutation(len(all_norm_source))
all_norm_source = all_norm_source[perm]
all_norm_source_c = all_norm_source_c[perm]
#
all_anom_source = np.concatenate(all_anom_source)
all_anom_source_c = np.concatenate(all_anom_source_c)
perm = np.random.permutation(len(all_anom_source))
all_anom_source = all_anom_source[perm]
all_anom_source_c = all_anom_source_c[perm]
#
ax_l.scatter(all_norm_source[:, 0], all_norm_source[:, 1], c=all_norm_source_c,
             s=CONF.DOT_SIZE, marker=CONF.DOT_SHAPE, edgecolors="none")
ax_r.scatter(all_anom_source[:, 0], all_anom_source[:, 1], c=all_anom_source_c,
             s=CONF.DOT_SIZE, marker=CONF.DOT_SHAPE, edgecolors="none")


# TEST/TARGET: plot normals on the left and anomalies on the right
test_target_umaps = {k: v for k, v in umap_data.items()
                     if k[0]=="test" and k[1]==CONF.DEVICE and k[3]=="target"}
all_norm_target = []
all_norm_target_c = []
all_anom_target = []
all_anom_target_c = []
for (_, dev, sec, dom), v in test_target_umaps.items():
    norm_color = CONF.DOT_COLORS["normal_target"][int(sec)]
    anom_color = CONF.DOT_COLORS["anomaly_target"][int(sec)]
    labels = np.array([ANOMALY_LABELS[json.loads(s)["label"]]
                       for s in v["metadata"]])
    norm_idxs = np.where(labels == 0)[0]
    anom_idxs = np.where(labels == 1)[0]
    norm_umaps = v["umaps"][norm_idxs]
    anom_umaps = v["umaps"][anom_idxs]
    #
    all_norm_target.append(norm_umaps)
    all_norm_target_c.append([norm_color for _ in norm_umaps])
    all_anom_target.append(anom_umaps)
    all_anom_target_c.append([anom_color for _ in anom_umaps])

all_norm_target = np.concatenate(all_norm_target)
all_norm_target_c = np.concatenate(all_norm_target_c)
perm = np.random.permutation(len(all_norm_target))
all_norm_target = all_norm_target[perm]
all_norm_target_c = all_norm_target_c[perm]
#
all_anom_target = np.concatenate(all_anom_target)
all_anom_target_c = np.concatenate(all_anom_target_c)
perm = np.random.permutation(len(all_anom_target))
all_anom_target = all_anom_target[perm]
all_anom_target_c = all_anom_target_c[perm]
#
ax_l.scatter(all_norm_target[:, 0], all_norm_target[:, 1], c=all_norm_target_c,
             s=CONF.DOT_SIZE, marker=CONF.DOT_SHAPE, edgecolors="none")
ax_r.scatter(all_anom_target[:, 0], all_anom_target[:, 1], c=all_anom_target_c,
             s=CONF.DOT_SIZE, marker=CONF.DOT_SHAPE, edgecolors="none")


if CONF.WITH_CROSS:
    # We want to signal a low-energy position with a cross, To have some sort of
    # idea for an origin. For that, we fetch the original frames. To prevent
    # outliers, we ignore lowest energies and average the rest (median may be risky)
    energies = get_umap_energies(umap_data, devices=[CONF.DEVICE],
                                 sections=[CONF.SECTION])
    merged_energies_and_umaps = sum([list(zip(v, umap_data[k]["umaps"]))
                                     for k, v in energies.items()], [])
    umaps_by_energy = sorted(merged_energies_and_umaps, key=lambda elt: elt[0])
    sorted_energies, umaps_by_energy = zip(*sorted(merged_energies_and_umaps,
                                                   key=lambda elt: elt[0]))
    umaps_by_energy = np.stack(umaps_by_energy)
    beg = CONF.CROSS_EXCLUDE_LOWEST
    end = beg + CONF.CROSS_AVERAGE_N
    cross_x, cross_y = np.mean(umaps_by_energy[beg:end], axis=0)
    #
    ax_l.scatter(cross_x, cross_y, c=CONF.CROSS_COLOR, s=CONF.CROSS_SIZE,
                 marker=CONF.CROSS_SHAPE, label="Approx. origin")
    ax_r.scatter(cross_x, cross_y, c=CONF.CROSS_COLOR, s=CONF.CROSS_SIZE,
                 marker=CONF.CROSS_SHAPE, label="Approx. origin")


# Get colors that were used
used_source_train_c = np.unique(all_source_train_c)
used_source_train_c = [c for c in CONF.SHADOW_COLORS["source"] if c in
                       used_source_train_c]
used_target_train_c = np.unique(all_target_train_c)
used_target_train_c = [c for c in CONF.SHADOW_COLORS["target"] if c in
                       used_target_train_c]
used_norm_source_test_c = np.unique(all_norm_source_c)
used_norm_source_test_c = [c for c in CONF.DOT_COLORS["normal_source"] if c in
                           used_norm_source_test_c]
used_norm_target_test_c = np.unique(all_norm_target_c)
used_norm_target_test_c = [c for c in CONF.DOT_COLORS["normal_target"] if c in
                           used_norm_target_test_c]
used_anom_source_test_c = np.unique(all_anom_source_c)
used_anom_source_test_c = [c for c in CONF.DOT_COLORS["anomaly_source"] if c in
                           used_anom_source_test_c]
used_anom_target_test_c = np.unique(all_anom_target_c)
used_anom_target_test_c = [c for c in CONF.DOT_COLORS["anomaly_target"] if c in
                           used_anom_target_test_c]


# https://matplotlib.org/stable/api/legend_handler_api.html
mc_handler = MulticolorHandler(width_factor=CONF.LEGEND_WIDTH_FACTOR)




# Right plot legend
ax_r_handles, ax_r_labels = ax_r.get_legend_handles_labels()
if CONF.PLOT_AUDIOSET:
    ax_r_labels.append(CONF.AUDIOSET_LABEL)
    ax_r_handles.append(MulticolorCircles(
        # Create "invisible" circles to adjust space
        ["none", "none", "none", "none", "none", CONF.AUDIOSET_COLOR],
        radius_factor=CONF.LEGEND_DOT_EXTERNAL_RATIO,
        face_alpha=CONF.LEGEND_SHADOW_ALPHA))
if CONF.PLOT_FRAUNHOFER:
    ax_r_labels.append(CONF.FRAUNHOFER_LABEL)
    ax_r_handles.append(MulticolorCircles(
        # Create "invisible" circles to adjust space
        ["none", "none", "none", "none", "none", CONF.FRAUNHOFER_COLOR],
        radius_factor=CONF.LEGEND_DOT_EXTERNAL_RATIO,
        face_alpha=CONF.LEGEND_SHADOW_ALPHA))
ax_r_labels.append(CONF.TRAIN_SOURCE_NORMAL_LABEL)
ax_r_handles.append(MulticolorCircles(used_source_train_c,
                                      radius_factor=CONF.LEGEND_DOT_TRAIN_RATIO,
                                      face_alpha=CONF.LEGEND_SHADOW_ALPHA))
ax_r_labels.append(CONF.TRAIN_TARGET_NORMAL_LABEL)
ax_r_handles.append(MulticolorCircles(used_target_train_c,
                                      radius_factor=CONF.LEGEND_DOT_TRAIN_RATIO,
                                      face_alpha=CONF.LEGEND_SHADOW_ALPHA))

ax_r_labels.append(CONF.TEST_SOURCE_NORMAL_LABEL)
ax_r_handles.append(MulticolorCircles(
    ["none", "none", "none"] + used_norm_source_test_c,
    radius_factor=CONF.LEGEND_DOT_TEST_RATIO))
ax_r_labels.append(CONF.TEST_TARGET_NORMAL_LABEL)
ax_r_handles.append(MulticolorCircles(
    ["none", "none", "none"] + used_norm_target_test_c,
    radius_factor=CONF.LEGEND_DOT_TEST_RATIO))

ax_r_labels.append(CONF.TEST_SOURCE_ANOMALY_LABEL)
ax_r_handles.append(MulticolorCircles(
    ["none", "none", "none"] + used_anom_source_test_c,
    radius_factor=CONF.LEGEND_DOT_TEST_RATIO))
ax_r_labels.append(CONF.TEST_TARGET_ANOMALY_LABEL)
ax_r_handles.append(MulticolorCircles(
    ["none", "none", "none"] + used_anom_target_test_c,
    radius_factor=CONF.LEGEND_DOT_TEST_RATIO))
#


ax_r.legend(ax_r_handles, ax_r_labels, handlelength=CONF.LEGEND_ICON_SIZE,
            borderpad=1, labelspacing=1,
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
                    hspace=0, wspace=0.01)

if CONF.SAVEFIG_PATH is not None:
    fig.savefig(CONF.SAVEFIG_PATH, dpi=CONF.DPI)
    print("Saved plot to", CONF.SAVEFIG_PATH)
else:
    fig.show()
    breakpoint()
