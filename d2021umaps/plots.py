#!/usr/bin python
# -*- coding:utf-8 -*-


"""
"""


import random
#
import numpy as np
# import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.text as pltext
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt


# ##############################################################################
# # GLOBALS
# ##############################################################################
PALETTE = mcolors.CSS4_COLORS

GLOBAL_COLORS = {("other", "other", "other"): NotImplemented,
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


# ##############################################################################
# #
# ##############################################################################
def shadow_dot_scatter(ax, dots_low, dots_hi, shadows_low, shadows_hi,
                       max_dl=None, max_dh=None, max_sl=None, max_sh=None,
                       dots_alpha=0.8, shadows_alpha=0.4,
                       dots_surface=0.5, shadows_surface=1,
                       dots_low_c="seagreen", dots_hi_c="orange",
                       shadows_low_c="navy", shadows_hi_c="purple",
                       dots_low_shape=".", dots_hi_shape="^",
                       dots_low_legend=None, dots_hi_legend=None,
                       shadows_low_legend=None, shadows_hi_legend=None):
    """
    Inputs are arrays of shape ``(N_i, 2)``, or empty lists
    """
    if max_dl is not None and len(dots_low) > max_dl:
        dots_low = random.sample(dots_low, max_dl)
    if max_dh is not None and len(dots_hi) > max_dh:
        dots_hi = random.sample(dots_hi, max_dh)
    if max_sl is not None and len(shadows_low) > max_sl:
        shadows_low = random.sample(shadows_low, max_sl)
    if max_sh is not None and len(shadows_hi) > max_sh:
        shadows_hi = random.sample(shadows_hi, max_sh)
    #
    alpha_dl = mcolors.colorConverter.to_rgba(
        dots_low_c, alpha=dots_alpha)
    alpha_dh = mcolors.colorConverter.to_rgba(
        dots_hi_c, alpha=dots_alpha)
    alpha_sl = mcolors.colorConverter.to_rgba(
        shadows_low_c, alpha=shadows_alpha)
    alpha_sh = mcolors.colorConverter.to_rgba(
        shadows_hi_c, alpha=shadows_alpha)
    # Plot shadows before dots, and low before high
    if len(shadows_low) > 0:
        ax.scatter(shadows_low[:, 0], shadows_low[:, 1], c=[alpha_sl],
                   s=shadows_surface, edgecolors="none",
                   label=shadows_low_legend)
    if len(shadows_hi) > 0:
        ax.scatter(shadows_hi[:, 0], shadows_hi[:, 1], c=[alpha_sh],
                   s=shadows_surface, edgecolors="none",
                   label=shadows_hi_legend)
    if len(dots_low) > 0:
        ax.scatter(dots_low[:, 0], dots_low[:, 1], c=[alpha_dl],
                   s=dots_surface, edgecolors="none",
                   marker=dots_low_shape, label=dots_low_legend)
    if len(dots_hi) > 0:
        ax.scatter(dots_hi[:, 0], dots_hi[:, 1], c=[alpha_dh],
                   s=dots_surface, edgecolors="none",
                   marker=dots_hi_shape, label=dots_hi_legend)
    #
    return ax


def opaque_legend(ax, fontsize=None, handler_map=None):
    """
    Calls legend, and sets all the legend colors opacity to 100%.
    Returns the legend handle.
    """
    leg = ax.legend(fontsize=fontsize, handler_map=handler_map)
    for lh in leg.legendHandles:
        fc_arr = lh.get_fc().copy()
        fc_arr[:, -1] = 1
        lh.set_fc(fc_arr)
    return leg



# ##############################################################################
# # LEGEND
# ##############################################################################
class MulticolorCircles:
    """
    For different shapes, override the ``get_patch`` method, and add the new
    class to the handler map, e.g. via

    ax_r.legend(ax_r_handles, ax_r_labels, handlelength=CONF.LEGEND_ICON_SIZE,
            borderpad=1.2, labelspacing=1.2,
            handler_map={MulticolorCircles: MulticolorHandler})
    """

    def __init__(self, face_colors, edge_colors=None, face_alpha=1,
                 radius_factor=1):
        """
        """
        assert 0 <= face_alpha <= 1, f"Invalid face_alpha: {face_alpha}"
        assert radius_factor > 0, "radius_factor must be positive"
        self.rad_factor = radius_factor
        self.fc = [mcolors.colorConverter.to_rgba(fc, alpha=face_alpha)
                   for fc in face_colors]
        self.ec = edge_colors
        if edge_colors is None:
            self.ec = ["none" for _ in self.fc]
        self.N = len(self.fc)

    def get_patch(self, width, height, idx, fc, ec):
        """
        """
        w_chunk = width / self.N
        radius = min(w_chunk / 2, height) * self.rad_factor
        xy = (w_chunk * idx + radius, radius)
        patch = plt.Circle(xy, radius, facecolor=fc, edgecolor=ec)
        return patch

    def __call__(self, width, height):
        """
        """
        w_chunk = width / self.N
        patches = []
        for i, (fc, ec) in enumerate(zip(self.fc, self.ec)):
            patch = self.get_patch(width, height, i, fc, ec)
            patches.append(patch)
        result = PatchCollection(patches, match_original=True)
        return result


class MulticolorTriangles(MulticolorCircles):
    """
    """
    def get_patch(self, width, height, idx, fc, ec):
        """
        """
        w_chunk = width / self.N
        radius = min(w_chunk / 2, height) * self.rad_factor
        xy = np.array([[0, 0], [1, 0], [0.5, 0.866]]) * radius
        xy[:, 0] += w_chunk * idx  # horizontal offset
        patch = plt.Polygon(xy, radius, facecolor=fc, edgecolor=ec)
        return patch


class MulticolorHandler:
    """
    """
    def __init__(self, width_factor=1):
        """
        """
        self.w_factor = width_factor

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        """
        """
        handlebox.set_width(handlebox.width * self.w_factor)
        width, height = handlebox.width, handlebox.height
        patch = orig_handle(width, height)
        handlebox.add_artist(patch)
        return patch


class StrHandler:
    """
    Allows to use ``str(o)`` for an arbitrary python object ``o``
    as a legend handle. Modified from:
    https://stackoverflow.com/a/27175052/4511978
    """

    def __init__(self, fontsize=16, weight=500, color="black", rotation=0,
                 left_margin_ratio=0, width_factor=1):
        """
        """
        self.fs = fontsize
        self.weight = weight
        self.color = color
        self.rotation = rotation
        self.lmargin = left_margin_ratio
        self.w_factor = width_factor

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        """
        """
        handlebox.set_width(handlebox.width * self.w_factor)
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        x0 += width * self.lmargin
        txt = pltext.Text(x0, y0, text=str(orig_handle),
                          color=self.color, verticalalignment="baseline",
                          horizontalalignment="right", multialignment=None,
                          fontsize=fontsize, fontweight=self.weight,
                          rotation=self.rotation)
        handlebox.add_artist(txt)
        return txt
