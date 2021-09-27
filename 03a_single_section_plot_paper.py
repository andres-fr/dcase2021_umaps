#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
Given a per-device UMAP and section, this script plots a dual scatterplot with
info about the external datasets, training/test data, source/target domain,
anomaly (right) vs. normal (left) sounds, and color-code by filename.

It is suitable for inspecting the internal structure of a single split in
detail. E.g. it can help understanding how many test anomaly files can be easily
separated from the training data: Inspect a specific section, and see how many
dots of different colors are outside of the training support. Then compare
left/right: if all is good, normals should be supported, anomalies shouldn't.
"""


import os
import pickle
import json
#
from omegaconf import OmegaConf
import numpy as np
from randomcolor import RandomColor
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
#
from d2021umaps.plots import shadow_dot_scatter, opaque_legend
from d2021umaps.data import ANOMALY_LABELS, get_umap_energies


# ##############################################################################
# # GLOBALS
# ##############################################################################
CONF = OmegaConf.create()
CONF.SECTION = 0
CONF.DEVICE_UMAP_PATH = os.path.join(
    "umaps",
    "UMAP_modality=mel_splits=gearbox_stack=5_maxDcaseTrain=10000_" +
    "maxDcaseTest=20000_maxAudioset=10000_maxFraunhofer=10000.pickle")
# plot conf
CONF.DCASE_SHADOW_SIZE = 60
CONF.EXTERNAL_SHADOW_SIZE = 60
CONF.SHADOW_ALPHA = 0.1
CONF.DOT_SIZE = 12
CONF.AUDIOSET_COLOR = "grey"
CONF.FRAUNHOFER_COLOR = "violet"
CONF.SHADOW_SOURCE_COLOR = "lightblue"
CONF.SHADOW_TARGET_COLOR = "palegreen"
CONF.SOURCE_COLOR = "blue"
CONF.TARGET_COLOR = "purple"
CONF.SOURCE_SHAPE = "o"
CONF.TARGET_SHAPE = "^"
CONF.PLOT_AUDIOSET = True
CONF.PLOT_FRAUNHOFER = True
CONF.AUDIOSET_LABEL = "Audioset"
CONF.FRAUNHOFER_LABEL = "Fraunhofer"  # IDMT-ISA-Electric-Engine
CONF.TRAIN_SOURCE_NORMAL_LABEL = "Train/source/normal"
CONF.TRAIN_TARGET_NORMAL_LABEL = "Train/target/normal"
# CONF.TEST_SOURCE_NORMAL_LABEL = "Test/source/normal"
# CONF.TEST_TARGET_NORMAL_LABEL = "Test/target/normal"
CONF.TEST_SOURCE_ANOMALY_LABEL = "Test/source"
CONF.TEST_TARGET_ANOMALY_LABEL = "Test/target"
CONF.FIGSIZE = (30, 15)
CONF.DPI = 300
CONF.SAVEFIG_PATH = None
CONF.BG_COLOR = (0.975, 0.985, 1) # rgb
CONF.WITH_CROSS = False
CONF.CROSS_SIZE = 100
CONF.CROSS_SHAPE = "P"
CONF.CROSS_COLOR = "black"
CONF.CROSS_EXCLUDE_LOWEST = 500
CONF.CROSS_AVERAGE_N = 100
#
CONF.LEGEND_SHADOW_ALPHA = 0.7
CONF.LEGEND_SHADOW_SIZE = 2000
CONF.LEGEND_DOT_RATIO = 0.35
CONF.LEGEND_FONT_SIZE = 55
#
CONF.CUT_TOP = 0.0
CONF.CUT_BOTTOM = 0.0
CONF.CUT_LEFT = 0.0
CONF.CUT_RIGHT = 0.0


cli_conf = OmegaConf.from_cli()
CONF = OmegaConf.merge(CONF, cli_conf)

# convert integer to double-digit string 3->"03"
CONF.SECTION = "{:02d}".format(CONF.SECTION)


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
                                  shadows_low_c=CONF.AUDIOSET_COLOR,
                                  shadows_low_legend=CONF.AUDIOSET_LABEL)
        ax_r = shadow_dot_scatter(ax_r, [], [], audioset_umaps, [],
                                  shadows_alpha=CONF.SHADOW_ALPHA,
                                  shadows_surface=CONF.EXTERNAL_SHADOW_SIZE,
                                  shadows_low_c=CONF.AUDIOSET_COLOR,
                                  shadows_low_legend=CONF.AUDIOSET_LABEL)
    except Exception as e:
        print("Error plotting AudioSet UMAPs!", e)


# try to add Fraunhofer umaps to both left and right plots
if CONF.PLOT_FRAUNHOFER:
    try:
        fraunhofer_umaps = umap_data["fraunhofer"]["umaps"]
        ax_l = shadow_dot_scatter(ax_l, [], [], fraunhofer_umaps, [],
                                  shadows_alpha=CONF.SHADOW_ALPHA,
                                  shadows_surface=CONF.EXTERNAL_SHADOW_SIZE,
                                  shadows_low_c=CONF.FRAUNHOFER_COLOR,
                                  shadows_low_legend=CONF.FRAUNHOFER_LABEL)
        ax_r = shadow_dot_scatter(ax_r, [], [], fraunhofer_umaps, [],
                                  shadows_alpha=CONF.SHADOW_ALPHA,
                                  shadows_surface=CONF.EXTERNAL_SHADOW_SIZE,
                                  shadows_low_c=CONF.FRAUNHOFER_COLOR,
                                  shadows_low_legend=CONF.FRAUNHOFER_LABEL)
    except Exception as e:
        print("Error plotting Fraunhofer UMAPs!", e)


train_source_umaps = umap_data[("train", CONF.DEVICE, CONF.SECTION,
                                "source")]["umaps"]
train_target_umaps = umap_data[("train", CONF.DEVICE, CONF.SECTION,
                                "target")]["umaps"]
ax_l = shadow_dot_scatter(ax_l, [], [], train_source_umaps, train_target_umaps,
                          shadows_alpha=CONF.SHADOW_ALPHA,
                          shadows_surface=CONF.DCASE_SHADOW_SIZE,
                          shadows_low_c=CONF.SHADOW_SOURCE_COLOR,
                          shadows_hi_c=CONF.SHADOW_TARGET_COLOR,
                          shadows_low_legend=CONF.TRAIN_SOURCE_NORMAL_LABEL,
                          shadows_hi_legend=CONF.TRAIN_TARGET_NORMAL_LABEL)
#
ax_r = shadow_dot_scatter(ax_r, [], [], train_source_umaps, train_target_umaps,
                          shadows_alpha=CONF.SHADOW_ALPHA,
                          shadows_surface=CONF.DCASE_SHADOW_SIZE,
                          shadows_low_c=CONF.SHADOW_SOURCE_COLOR,
                          shadows_hi_c=CONF.SHADOW_TARGET_COLOR,
                          shadows_low_legend=CONF.TRAIN_SOURCE_NORMAL_LABEL,
                          shadows_hi_legend=CONF.TRAIN_TARGET_NORMAL_LABEL)

train_source_md = [json.loads(md) for md in umap_data[
    ("train", CONF.DEVICE, CONF.SECTION, "source")]["metadata"]]
train_target_md = [json.loads(md) for md in umap_data[
    ("train", CONF.DEVICE, CONF.SECTION, "target")]["metadata"]]
try:
    test_source_md = [json.loads(md) for md in umap_data[
        ("test", CONF.DEVICE, CONF.SECTION, "source")]["metadata"]]
    test_target_md = [json.loads(md) for md in umap_data[
        ("test", CONF.DEVICE, CONF.SECTION, "target")]["metadata"]]
except KeyError as ke:
    print("ERROR:", ke)
    raise Exception(
        "DCASE test data files provided don't have entries for this split")


# Get colors by filename
test_source_paths = [md["path"] for md in test_source_md]
test_target_paths = [md["path"] for md in test_target_md]
unique_paths = np.unique(test_source_paths + test_target_paths)
rand_colors = RandomColor().generate(count=len(unique_paths))
color_map = dict(zip(unique_paths, rand_colors))
test_source_c = np.array([color_map[p] for p in test_source_paths])
test_target_c = np.array([color_map[p] for p in test_target_paths])

# get normal/anomaly labels
test_source_labels = [ANOMALY_LABELS[md["label"]] for md in test_source_md]
test_target_labels = [ANOMALY_LABELS[md["label"]] for md in test_target_md]
test_source_ano_idxs = np.where(test_source_labels)
test_source_nor_idxs = np.where(np.logical_not(test_source_labels))
test_target_ano_idxs = np.where(test_target_labels)
test_target_nor_idxs = np.where(np.logical_not(test_target_labels))

# Split umaps, labels and colors by domain and label:
test_source_umaps = umap_data[("test", CONF.DEVICE, CONF.SECTION,
                               "source")]["umaps"]
test_target_umaps = umap_data[("test", CONF.DEVICE, CONF.SECTION,
                               "target")]["umaps"]
#

test_source_nor_umaps = test_source_umaps[test_source_nor_idxs]
test_source_ano_umaps = test_source_umaps[test_source_ano_idxs]
test_target_nor_umaps = test_target_umaps[test_target_nor_idxs]
test_target_ano_umaps = test_target_umaps[test_target_ano_idxs]
test_source_nor_colors = test_source_c[test_source_nor_idxs]
test_source_ano_colors = test_source_c[test_source_ano_idxs]
test_target_nor_colors = test_target_c[test_target_nor_idxs]
test_target_ano_colors = test_target_c[test_target_ano_idxs]


# Add the test/source/normal entries to L
ax_l.scatter(test_source_nor_umaps[:, 0],
             test_source_nor_umaps[:, 1],
             c=test_source_nor_colors, s=CONF.DOT_SIZE,
             edgecolors="none", marker=CONF.SOURCE_SHAPE)
# Add the test/target/normal entries to L
ax_l.scatter(test_target_nor_umaps[:, 0],
             test_target_nor_umaps[:, 1],
             c=test_target_nor_colors, s=CONF.DOT_SIZE,
             edgecolors="none", marker=CONF.TARGET_SHAPE)

# Add the test/source/anomaly entries to L
ax_r.scatter(test_source_ano_umaps[:, 0],
             test_source_ano_umaps[:, 1],
             c=test_source_ano_colors, s=CONF.DOT_SIZE,
             edgecolors="none", marker=CONF.SOURCE_SHAPE)
# Add the test/target/normal entries to L
ax_r.scatter(test_target_ano_umaps[:, 0],
             test_target_ano_umaps[:, 1],
             c=test_target_ano_colors, s=CONF.DOT_SIZE,
             edgecolors="none", marker=CONF.TARGET_SHAPE)


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


# Finally tweak the legend
source_dot_marker = plt.Line2D(
    [0], [0], marker=CONF.SOURCE_SHAPE,
    color="black", markerfacecolor="black", linestyle="none",
    markersize=CONF.LEGEND_FONT_SIZE * CONF.LEGEND_DOT_RATIO)
target_dot_marker = plt.Line2D(
    [0], [0], marker=CONF.TARGET_SHAPE,
    color="black", markerfacecolor="black", linestyle="none",
    markersize=CONF.LEGEND_FONT_SIZE * CONF.LEGEND_DOT_RATIO)

ax_r_handles, ax_r_labels = ax_r.get_legend_handles_labels()
ax_r_handles.extend([source_dot_marker, target_dot_marker])
ax_r_labels.extend([CONF.TEST_SOURCE_ANOMALY_LABEL,
                    CONF.TEST_TARGET_ANOMALY_LABEL])

# create legend and make them opaque
leg = ax_r.legend(ax_r_handles, ax_r_labels, borderpad=0.1, labelspacing=0.2,
                  handletextpad=-0.3,
                  bbox_to_anchor=(0.65, 1),
                  loc="upper left",
                  prop={'size': CONF.LEGEND_FONT_SIZE})
for lh in leg.legendHandles:
    if isinstance(lh, PathCollection):
        fc_arr = lh.get_fc().copy()
        fc_arr[:, -1] = CONF.LEGEND_SHADOW_ALPHA
        lh.set_fc(fc_arr)
        lh.set_sizes([CONF.LEGEND_SHADOW_SIZE])

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




fig.subplots_adjust(top=0.99, bottom=0.01, left=0.01, right=0.8, hspace=0,
                    wspace=0.01)

if CONF.SAVEFIG_PATH is not None:
    fig.savefig(CONF.SAVEFIG_PATH, dpi=CONF.DPI)
    print("Saved plot to", CONF.SAVEFIG_PATH)
else:
    fig.show()
    breakpoint()
