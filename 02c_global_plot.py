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
from collections import defaultdict
#
from omegaconf import OmegaConf
import numpy as np
from randomcolor import RandomColor
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import PathCollection
#
from d2021umaps.data import ALL_DEVICES, ALL_SECTIONS, ALL_DOMAINS
from d2021umaps.data import ANOMALY_LABELS, get_umap_energies
from d2021umaps.plots import shadow_dot_scatter
from d2021umaps.plots import MulticolorCircles, MulticolorTriangles, MulticolorHandler

# ##############################################################################
# # GLOBALS
# ##############################################################################
CONF = OmegaConf.create()
CONF.GLOBAL_UMAP_PATH = os.path.join(
    "umaps",
    "UMAP_modality=mel_splits=GLOBAL_stack=5_maxDcaseTrain=1000_" +
    "maxDcaseTest=2000_maxAudioset=50000_maxFraunhofer=50000.pickle")
# plot conf
CONF.DCASE_SHADOW_SIZE = 50
CONF.EXTERNAL_SHADOW_SIZE = 50
CONF.SHADOW_ALPHA = 0.02
CONF.SOURCE_SHAPE = "."
CONF.TARGET_SHAPE = "^"
CONF.TEST_SOURCE_SIZE = 6
CONF.TEST_TARGET_SIZE = 5
CONF.PLOT_AUDIOSET = True
CONF.PLOT_FRAUNHOFER = True
CONF.PLOT_LEGEND = True
CONF.AUDIOSET_COLOR = "grey"
CONF.FRAUNHOFER_COLOR = (0.55, 0.45, 0.55) #  "violet"
PALETTE = mcolors.CSS4_COLORS
FULL_COLORS = {
    # ToyCar is rather blue
    ("ToyCar", "00", "source"): PALETTE["paleturquoise"],
    ("ToyCar", "00", "target"): PALETTE["paleturquoise"],
    ("ToyCar", "01", "source"): PALETTE["lightblue"],
    ("ToyCar", "01", "target"): PALETTE["lightblue"],
    ("ToyCar", "02", "source"): PALETTE["skyblue"],
    ("ToyCar", "02", "target"): PALETTE["skyblue"],
    ("ToyCar", "03", "source"): PALETTE["cornflowerblue"],
    ("ToyCar", "03", "target"): PALETTE["cornflowerblue"],
    ("ToyCar", "04", "source"): PALETTE["steelblue"],
    ("ToyCar", "04", "target"): PALETTE["steelblue"],
    ("ToyCar", "05", "source"): PALETTE["mediumblue"],
    ("ToyCar", "05", "target"): PALETTE["mediumblue"],
    # fan: red/brown
    ("fan", "00", "source"): PALETTE["lightsalmon"],
    ("fan", "00", "target"): PALETTE["lightsalmon"],
    ("fan", "01", "source"): PALETTE["lightcoral"],
    ("fan", "01", "target"): PALETTE["lightcoral"],
    ("fan", "02", "source"): PALETTE["orangered"],
    ("fan", "02", "target"): PALETTE["orangered"],
    ("fan", "03", "source"): PALETTE["indianred"],
    ("fan", "03", "target"): PALETTE["indianred"],
    ("fan", "04", "source"): PALETTE["firebrick"],
    ("fan", "04", "target"): PALETTE["firebrick"],
    ("fan", "05", "source"): PALETTE["maroon"],
    ("fan", "05", "target"): PALETTE["maroon"],
    # gearbox: grey/bluish
    ("gearbox", "00", "source"): PALETTE["gainsboro"],
    ("gearbox", "00", "target"): PALETTE["gainsboro"],
    ("gearbox", "01", "source"): PALETTE["silver"],
    ("gearbox", "01", "target"): PALETTE["silver"],
    ("gearbox", "02", "source"): PALETTE["lightsteelblue"],
    ("gearbox", "02", "target"): PALETTE["lightsteelblue"],
    ("gearbox", "03", "source"): PALETTE["darkgrey"],
    ("gearbox", "03", "target"): PALETTE["darkgrey"],
    ("gearbox", "04", "source"): PALETTE["lightslategrey"],
    ("gearbox", "04", "target"): PALETTE["lightslategrey"],
    ("gearbox", "05", "source"): PALETTE["dimgrey"],
    ("gearbox", "05", "target"): PALETTE["dimgrey"],
    # pump: orange
    ("pump", "00", "source"): PALETTE["palegoldenrod"],
    ("pump", "00", "target"): PALETTE["palegoldenrod"],
    ("pump", "01", "source"): PALETTE["khaki"],
    ("pump", "01", "target"): PALETTE["khaki"],
    ("pump", "02", "source"): PALETTE["gold"],
    ("pump", "02", "target"): PALETTE["gold"],
    ("pump", "03", "source"): PALETTE["orange"],
    ("pump", "03", "target"): PALETTE["orange"],
    ("pump", "04", "source"): PALETTE["darkorange"],
    ("pump", "04", "target"): PALETTE["darkorange"],
    ("pump", "05", "source"): PALETTE["darkgoldenrod"],
    ("pump", "05", "target"): PALETTE["darkgoldenrod"],
    # slider: green
    ("slider", "00", "source"): PALETTE["palegreen"],
    ("slider", "00", "target"): PALETTE["palegreen"],
    ("slider", "01", "source"): PALETTE["greenyellow"],
    ("slider", "01", "target"): PALETTE["greenyellow"],
    ("slider", "02", "source"): PALETTE["lime"],
    ("slider", "02", "target"): PALETTE["lime"],
    ("slider", "03", "source"): PALETTE["limegreen"],
    ("slider", "03", "target"): PALETTE["limegreen"],
    ("slider", "04", "source"): PALETTE["forestgreen"],
    ("slider", "04", "target"): PALETTE["forestgreen"],
    ("slider", "05", "source"): PALETTE["darkgreen"],
    ("slider", "05", "target"): PALETTE["darkgreen"],
    # ToyTrain: pink
    ("ToyTrain", "00", "source"): PALETTE["pink"],
    ("ToyTrain", "00", "target"): PALETTE["pink"],
    ("ToyTrain", "01", "source"): PALETTE["hotpink"],
    ("ToyTrain", "01", "target"): PALETTE["hotpink"],
    ("ToyTrain", "02", "source"): PALETTE["deeppink"],
    ("ToyTrain", "02", "target"): PALETTE["deeppink"],
    ("ToyTrain", "03", "source"): PALETTE["palevioletred"],
    ("ToyTrain", "03", "target"): PALETTE["palevioletred"],
    ("ToyTrain", "04", "source"): PALETTE["orchid"],
    ("ToyTrain", "04", "target"): PALETTE["orchid"],
    ("ToyTrain", "05", "source"): PALETTE["mediumvioletred"],
    ("ToyTrain", "05", "target"): PALETTE["mediumvioletred"],
    # valve: purple
    ("valve", "00", "source"): PALETTE["plum"],
    ("valve", "00", "target"): PALETTE["plum"],
    ("valve", "01", "source"): PALETTE["violet"],
    ("valve", "01", "target"): PALETTE["violet"],
    ("valve", "02", "source"): PALETTE["magenta"],
    ("valve", "02", "target"): PALETTE["magenta"],
    ("valve", "03", "source"): PALETTE["mediumorchid"],
    ("valve", "03", "target"): PALETTE["mediumorchid"],
    ("valve", "04", "source"): PALETTE["darkviolet"],
    ("valve", "04", "target"): PALETTE["darkviolet"],
    ("valve", "05", "source"): PALETTE["rebeccapurple"],
    ("valve", "05", "target"): PALETTE["rebeccapurple"]}
DOT_COLORS = {(dev, sec, dom): FULL_COLORS[dev, "04", dom]
                 for dev, sec, dom in FULL_COLORS}
SHADOW_COLORS = {(dev, sec, dom): FULL_COLORS[dev, "01", dom]
                 if dev in {"fan", "valve"} else FULL_COLORS[dev, "00", dom]
                 for dev, sec, dom in FULL_COLORS}
CONF.AUDIOSET_LABEL = "Audioset"
CONF.FRAUNHOFER_LABEL = "Fraunhofer"  # IDMT-ISA-Electric-Engine
CONF.TRAIN_SOURCE_NORMAL_LABEL = "Train/source/normal"
CONF.TRAIN_TARGET_NORMAL_LABEL = "Train/target/normal"
CONF.TEST_SOURCE_NORMAL_LABEL = "Test/source/normal"
CONF.TEST_TARGET_NORMAL_LABEL = "Test/target/normal"
CONF.TEST_SOURCE_ANOMALY_LABEL = "Test/source/anomaly"
CONF.TEST_TARGET_ANOMALY_LABEL = "Test/target/anomaly"
CONF.FIGSIZE = (30, 15)
CONF.DPI = 500
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
CONF.LEGEND_TRAIN_RATIO = 1.7
CONF.LEGEND_TEST_SOURCE_RATIO = 1
CONF.LEGEND_TEST_TARGET_RATIO = 2
CONF.LEGEND_WIDTH_FACTOR = 1.5
CONF.LEGEND_FONT_SIZE = 14
CONF.LEGEND_ICON_SIZE = None


cli_conf = OmegaConf.from_cli()
CONF = OmegaConf.merge(CONF, cli_conf)

print(OmegaConf.to_yaml(CONF))


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
# Open UMAP file and create figure
with open(CONF.GLOBAL_UMAP_PATH, "rb") as f:
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
                      if k[0]=="train" and k[3]=="source"}
train_target_umaps = {k: v for k, v in umap_data.items()
                      if k[0]=="train" and k[3]=="target"}

# gather, shuffle and plot TRAIN/SOURCE data
all_source_train = []
all_source_train_c = []
used_source_train_c = {}
for (_, dev, sec, dom), v in train_source_umaps.items():
    umaps = v["umaps"]
    color = SHADOW_COLORS[(dev, sec, dom)]
    all_source_train.append(umaps)
    all_source_train_c.append([color for _ in umaps])
    used_source_train_c[(dev, sec, dom)] = color


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
used_target_train_c = {}
for (_, dev, sec, dom), v in train_target_umaps.items():
    umaps = v["umaps"]
    color = SHADOW_COLORS[(dev, sec, dom)]
    all_target_train.append(umaps)
    all_target_train_c.append([color for _ in umaps])
    used_target_train_c[(dev, sec, dom)] = color
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
                     if k[0]=="test" and k[3]=="source"}
all_norm_source = []
all_norm_source_c = []
all_anom_source = []
all_anom_source_c = []
used_norm_source_c = {}
used_anom_source_c = {}
for (_, dev, sec, dom), v in test_source_umaps.items():
    norm_color = DOT_COLORS[(dev, sec, dom)]
    anom_color = DOT_COLORS[(dev, sec, dom)]
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
    #
    used_norm_source_c[(dev, sec, dom)] = norm_color
    used_anom_source_c[(dev, sec, dom)] = anom_color

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
             s=CONF.TEST_SOURCE_SIZE, marker=CONF.SOURCE_SHAPE,
             edgecolors="none")
ax_r.scatter(all_anom_source[:, 0], all_anom_source[:, 1], c=all_anom_source_c,
             s=CONF.TEST_SOURCE_SIZE, marker=CONF.SOURCE_SHAPE,
             edgecolors="none")

# TEST/TARGET: plot normals on the left and anomalies on the right
test_target_umaps = {k: v for k, v in umap_data.items()
                     if k[0]=="test" and k[3]=="target"}
all_norm_target = []
all_norm_target_c = []
all_anom_target = []
all_anom_target_c = []
used_norm_target_c = {}
used_anom_target_c = {}
for (_, dev, sec, dom), v in test_target_umaps.items():
    norm_color = DOT_COLORS[(dev, sec, dom)]
    anom_color = DOT_COLORS[(dev, sec, dom)]
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
    #
    used_norm_target_c[(dev, sec, dom)] = norm_color
    used_anom_target_c[(dev, sec, dom)] = anom_color

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
             s=CONF.TEST_TARGET_SIZE, marker=CONF.TARGET_SHAPE, edgecolors="none")
ax_r.scatter(all_anom_target[:, 0], all_anom_target[:, 1], c=all_anom_target_c,
             s=CONF.TEST_TARGET_SIZE, marker=CONF.TARGET_SHAPE, edgecolors="none")


if CONF.WITH_CROSS:
    # We want to signal a low-energy position with a cross, To have some sort of
    # idea for an origin. For that, we fetch the original frames. To prevent
    # outliers, we ignore lowest energies and average the rest (median may be risky)
    energies = get_umap_energies(umap_data)
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


# used_target_train_c[(...)] = color
legend_train_source = defaultdict(list)
for dev, sec, dom in SHADOW_COLORS:
    if (dev, sec, dom) in used_source_train_c:
        col = used_source_train_c[(dev, sec, dom)]
        legend_train_source[f"Train/source/{dev}"].append(col)
legend_train_target = defaultdict(list)
for dev, sec, dom in SHADOW_COLORS:
    if (dev, sec, dom) in used_target_train_c:
        col = used_target_train_c[(dev, sec, dom)]
        legend_train_target[f"Train/target/{dev}"].append(col)

legend_test_source_norm = defaultdict(list)
for dev, sec, dom in DOT_COLORS:
    if (dev, sec, dom) in used_norm_source_c:
        col = used_norm_source_c[(dev, sec, dom)]
        legend_test_source_norm[f"Test/source/{dev}/normal"].append(col)
legend_test_source_anom = defaultdict(list)
for dev, sec, dom in DOT_COLORS:
    if (dev, sec, dom) in used_anom_source_c:
        col = used_anom_source_c[(dev, sec, dom)]
        legend_test_source_anom[f"Test/source/{dev}/anomaly"].append(col)
legend_test_target_norm = defaultdict(list)
for dev, sec, dom in DOT_COLORS:
    if (dev, sec, dom) in used_norm_target_c:
        col = used_norm_target_c[(dev, sec, dom)]
        legend_test_target_norm[f"Test/target/{dev}/normal"].append(col)
legend_test_target_anom = defaultdict(list)
for dev, sec, dom in DOT_COLORS:
    if (dev, sec, dom) in used_anom_target_c:
        col = used_anom_target_c[(dev, sec, dom)]
        legend_test_target_anom[f"Test/target/{dev}/anomaly"].append(col)

def test_marker(shape, colors, radius_factor):
    """
    """
    if shape == "^":
        mc = MulticolorTriangles
    elif shape == ".":
        mc = MulticolorCircles
    else:
        print("WARNING! shape not supported:", shape, "using circles")
    result = mc(colors, radius_factor=radius_factor)
    return result


# https://matplotlib.org/stable/api/legend_handler_api.html
mc_handler = MulticolorHandler(width_factor=CONF.LEGEND_WIDTH_FACTOR)


def plot_legend(ax, test_source_colordict, test_target_colordict,
                borderpad=1, labelspacing=1):
    """
    """
    ax_handles, ax_labels = ax.get_legend_handles_labels()
    if CONF.PLOT_AUDIOSET:
        ax_labels.append(CONF.AUDIOSET_LABEL)
        ax_handles.append(MulticolorCircles(
            # Create "invisible" circles to adjust space
            ["none", "none", "none", "none", "none", CONF.AUDIOSET_COLOR],
            radius_factor=CONF.LEGEND_DOT_EXTERNAL_RATIO,
            face_alpha=CONF.LEGEND_SHADOW_ALPHA))
    if CONF.PLOT_FRAUNHOFER:
        ax_labels.append(CONF.FRAUNHOFER_LABEL)
        ax_handles.append(MulticolorCircles(
            # Create "invisible" circles to adjust space
            ["none", "none", "none", "none", "none", CONF.FRAUNHOFER_COLOR],
            radius_factor=CONF.LEGEND_DOT_EXTERNAL_RATIO,
            face_alpha=CONF.LEGEND_SHADOW_ALPHA))
    #
    for lbl, colors in legend_train_source.items():
        ax_labels.append(lbl)
        ax_handles.append(MulticolorCircles(
            colors,radius_factor=CONF.LEGEND_TRAIN_RATIO,
            face_alpha=CONF.LEGEND_SHADOW_ALPHA))
    for lbl, colors in legend_train_target.items():
        ax_labels.append(lbl)
        ax_handles.append(MulticolorCircles(
            colors,radius_factor=CONF.LEGEND_TRAIN_RATIO,
            face_alpha=CONF.LEGEND_SHADOW_ALPHA))
    #
    for lbl, colors in test_source_colordict.items():
        colors = ["none", "none", "none"] + colors
        handle = test_marker(CONF.SOURCE_SHAPE, colors,
                             CONF.LEGEND_TEST_SOURCE_RATIO)
        ax_labels.append(lbl)
        ax_handles.append(handle)
    for lbl, colors in test_target_colordict.items():
        colors = ["none", "none", "none"] + colors
        handle = test_marker(CONF.TARGET_SHAPE, colors,
                             CONF.LEGEND_TEST_TARGET_RATIO)
        ax_labels.append(lbl)
        ax_handles.append(handle)
    ax.legend(ax_handles, ax_labels, handlelength=CONF.LEGEND_ICON_SIZE,
                borderpad=borderpad, labelspacing=labelspacing,
                prop={'size': CONF.LEGEND_FONT_SIZE},
                handler_map={MulticolorCircles: mc_handler,
                             MulticolorTriangles: mc_handler})

if CONF.PLOT_LEGEND:
    plot_legend(ax_l, legend_test_source_norm, legend_test_target_norm,
                0.2, 0.1)
    plot_legend(ax_r, legend_test_source_anom, legend_test_target_anom,
                0.2, 0.1)


ax_l.set_aspect("equal")
ax_r.set_aspect("equal")
fig.subplots_adjust(top=0.99, bottom=0.01, left=0.01, right=0.99, hspace=0,
                    wspace=0)

if CONF.SAVEFIG_PATH is not None:
    fig.savefig(CONF.SAVEFIG_PATH, dpi=CONF.DPI)
    print("Saved plot to", CONF.SAVEFIG_PATH)
else:
    fig.show()
    breakpoint()
