#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
Given the HDF5 datasets for the DCASE, AudioSet and Fraunhofer representations,
the DCASE splits and the max number of samples per dataset, this script samples
the datasets and computes the corresponding UMAP. The results are saved in a
dictionary with the following keys: ``config, audioset, fraunhofer,
(train, valve, 00, source), ...``. Config contains a string with the CONFIG
parameters used. Each of the other entries contains a dictionary with 4 keys:
umaps, metadata, global_idxs, relative_idxs. The umaps are arrays of shape
``(N, 2)`` containing ``N`` samples from the computed UMAP. The others
are N-element lists containing per-sample info: metadata about file path and
labels, global index to find the frame in the original HDF5 matrix, and relative
index to find the frame in the original file.
"""


import os
import random
import pickle
#
from omegaconf import OmegaConf
import numpy as np
import umap
#
from d2021umaps.logging import ColorLogger, make_timestamp
from d2021umaps.data import HDF5Dataset, DcaseHDF5Dataset


# ##############################################################################
# # GLOBALS
# ##############################################################################
CONF = OmegaConf.create()
cli_conf = OmegaConf.from_cli()

CONF.DCASE_TRAIN_PATH = os.path.join(
    "precomputed_features",
    "dcase2021_t2_train_wavnorm=absmax_mel_win1024_hop512_m128.h5")
CONF.DCASE_TEST_PATH = os.path.join(
    "precomputed_features",
    "dcase2021_t2_cv_wavnorm=absmax_mel_win1024_hop512_m128.h5")
CONF.AUDIOSET_PATH = os.path.join(
    "precomputed_features",
    "audioset_wavnorm=absmax_mel_win1024_hop512_m128.h5")
CONF.FRAUNHOFER_PATH = os.path.join(
    "precomputed_features",
    "fraunhofer_wavnorm=absmax_mel_win1024_hop512_m128.h5")
CONF.MODALITY = None
# CLI example: 'DCASE_SPLITS=[[valve,"00",source],[valve,"00",target]]'
CONF.DCASE_SPLITS = [["fan", "00", "source"], ["fan", "00", "target"]]
CONF.SPLITS_NAME = "||".join("|".join(x) for x in CONF.DCASE_SPLITS)
CONF.STACK = 5
# Set max size due to hardware limitations
CONF.MAX_DCASE_TRAIN = 1_000
CONF.MAX_DCASE_TEST = 1_000
CONF.MAX_AUDIOSET = 1_000
CONF.MAX_FRAUNHOFER = 1_000

log_ts = make_timestamp(timezone="Europe/London", with_tz_output=False)
CONF.LOG_OUTPATH = os.path.join("logs", "{}_[{}].log".format(log_ts, __file__))

# Finally merge again (priority order: 1.CLI 2.given file 3.defaults)
CONF = OmegaConf.merge(CONF, cli_conf)

CONF.STACK = int(CONF.STACK)
CONF.MAX_DCASE_TRAIN = int(CONF.MAX_DCASE_TRAIN)
CONF.MAX_DCASE_TEST = int(CONF.MAX_DCASE_TEST)
CONF.MAX_AUDIOSET = int(CONF.MAX_AUDIOSET)
CONF.MAX_FRAUNHOFER = int(CONF.MAX_FRAUNHOFER)

assert CONF.MODALITY is not None, \
    "Please specify MODALITY=... of the input data (e.g. stft, mel, l3...)"

CONF.OUTPUT_PATH = os.path.join(
    "umaps", f"UMAP_modality={CONF.MODALITY}_splits={CONF.SPLITS_NAME}_" +
    f"stack={CONF.STACK}_maxDcaseTrain={CONF.MAX_DCASE_TRAIN}_" +
    f"maxDcaseTest={CONF.MAX_DCASE_TEST}_maxAudioset={CONF.MAX_AUDIOSET}_" +
    f"maxFraunhofer={CONF.MAX_FRAUNHOFER}.pickle")


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
LOGGER = ColorLogger(__file__, CONF.LOG_OUTPATH, filemode="w")
LOGGER.info(f"\n\n\nSTARTED SCRIPT: {__file__}")
LOGGER.info(OmegaConf.to_yaml(CONF))

# load processors
umapper = umap.UMAP()

# load DCASE datasets
dcase_train_datasets = {}
for (device, section, domain) in CONF.DCASE_SPLITS:
    LOGGER.info(f"Loading DCASE train dataset: {(device, section, domain)}")
    ds = DcaseHDF5Dataset(CONF.DCASE_TRAIN_PATH, CONF.STACK,
                          filter_devices=[device],filter_sections=[section],
                          filter_domains=[domain], filter_splits=["train"])
    dcase_train_datasets[(device, section, domain)] = ds
    LOGGER.info(f" {len(ds.metadata)} files, {len(ds)} chunks")

dcase_test_datasets = {}
for (device, section, domain) in CONF.DCASE_SPLITS:
    LOGGER.info(f"Loading DCASE test dataset: {(device, section, domain)}")
    ds = DcaseHDF5Dataset(CONF.DCASE_TEST_PATH, CONF.STACK,
                          filter_devices=[device], filter_sections=[section],
                          filter_domains=[domain], filter_splits=["test"])
    dcase_test_datasets[(device, section, domain)] = ds
    LOGGER.info(f" {len(ds.metadata)} files, {len(ds)} chunks")

# gather chunks from DCASE datasets
train_data = {}
train_metadata = {}
train_rel_idxs = {}
train_glob_idxs = {}
test_data = {}
test_metadata = {}
test_rel_idxs = {}
test_glob_idxs = {}

for (dev, sec, domain) in dcase_test_datasets:
    # gather train chunks:
    train_ds = dcase_train_datasets[(dev, sec, domain)]
    train_idxs = range(len(train_ds))
    if (CONF.MAX_DCASE_TRAIN is not None and
        len(train_idxs) > CONF.MAX_DCASE_TRAIN):
        train_idxs = random.sample(train_idxs, CONF.MAX_DCASE_TRAIN)
    if len(train_idxs) > 0:
        LOGGER.info(
            f"Train {(dev, sec, domain)}, gathering {len(train_idxs)} chunks")
        train_d, gi, md, rel = train_ds.getitems(train_idxs)
        train_d = train_d.reshape(len(train_d), -1)
        #
        train_data[(dev, sec, domain)] = train_d
        train_metadata[(dev, sec, domain)] = md
        train_rel_idxs[(dev, sec, domain)] = rel
        train_glob_idxs[(dev, sec, domain)] = gi

    # gather test chunks:
    test_ds = dcase_test_datasets[(dev, sec, domain)]
    test_idxs = range(len(test_ds))
    if (CONF.MAX_DCASE_TEST is not None and
        len(test_idxs) > CONF.MAX_DCASE_TEST):
        test_idxs = random.sample(test_idxs, CONF.MAX_DCASE_TEST)
    if len(test_idxs) > 0:
        LOGGER.info(
            f"Test {(dev, sec, domain)}, gathering {len(test_idxs)} chunks")
        test_d, gi, md, rel = test_ds.getitems(test_idxs)
        test_d = test_d.reshape(len(test_d), -1)
        #
        test_data[(dev, sec, domain)] = test_d
        test_metadata[(dev, sec, domain)] = md
        test_rel_idxs[(dev, sec, domain)] = rel
        test_glob_idxs[(dev, sec, domain)] = gi

# free up memory
del dcase_train_datasets, dcase_test_datasets
del train_ds, test_ds
del train_idxs, test_idxs

# load and gather chunks from external datasets
LOGGER.info("Loading external datasets")
audioset_ds = HDF5Dataset(CONF.AUDIOSET_PATH, chunk_length=CONF.STACK)
fraunhofer_ds = HDF5Dataset(CONF.FRAUNHOFER_PATH, chunk_length=CONF.STACK)
try:
    audioset_idxs = random.sample(range(len(audioset_ds)), CONF.MAX_AUDIOSET)
    LOGGER.info(f"Gathering AudioSet chunks ({len(audioset_idxs)})")
except ValueError as ve:  # if we ask for more samples than ds length, pick all
    LOGGER.warning(
        f"AudioSet has less samples than requested! picking {len(audioset_ds)}")
    audioset_idxs = list(range(len(audioset_ds)))
(audioset_data, _, audioset_glob_idxs, audioset_metadata,
 audioset_rel_idxs) = audioset_ds.getitems(audioset_idxs)
audioset_data = audioset_data.reshape(len(audioset_data), -1)


try:
    fraunhofer_idxs = random.sample(range(len(fraunhofer_ds)),
                                    CONF.MAX_FRAUNHOFER)
    LOGGER.info(f"Gathering Fraunhofer chunks ({len(fraunhofer_idxs)})")
except ValueError as ve:  # if we ask for more samples than ds length, pick all
    LOGGER.warning("Fraunhofer has less samples than requested! " +
                   f"picking {len(fraunhofer_ds)}")
    fraunhofer_idxs = list(range(len(fraunhofer_ds)))
LOGGER.info(f"Gathering Fraunhofer chunks ({len(fraunhofer_idxs)})")
(fraunhofer_data, _, fraunhofer_glob_idxs, fraunhofer_metadata,
 fraunhofer_rel_idxs) = fraunhofer_ds.getitems(fraunhofer_idxs)
fraunhofer_data = fraunhofer_data.reshape(len(fraunhofer_data), -1)

# free up memory
del audioset_ds, fraunhofer_ds
del audioset_idxs, fraunhofer_idxs


# concatenate all data and compute UMAP (memory hungry, may take a while)
all_data = np.concatenate(list(train_data.values()) + list(test_data.values()))
len_dcase_data = len(all_data)
all_data = np.concatenate([all_data, audioset_data, fraunhofer_data])
LOGGER.info(f"Computing UMAP for shape {all_data.shape}")
all_umaps = umapper.fit_transform(all_data)

# split back umap data
audioset_umaps = all_umaps[len_dcase_data:len_dcase_data+len(audioset_data)]
del audioset_data
fraunhofer_umaps = all_umaps[-len(fraunhofer_data):]
del fraunhofer_data

train_umaps = {}
train_ranges = np.cumsum([0] + [len(x) for x in train_data.values()])
for i, (dev, sec, dom) in enumerate(train_data.keys()):
    beg, end = train_ranges[i], train_ranges[i+1]
    train_umaps[(dev, sec, dom)] = all_umaps[beg:end]
del train_data

test_umaps = {}
test_ranges = np.cumsum([0] + [len(x) for x
                               in test_data.values()])  + train_ranges[-1]
for i, (dev, sec, dom) in enumerate(test_data.keys()):
    beg, end = test_ranges[i], test_ranges[i+1]
    test_umaps[(dev, sec, dom)] = all_umaps[beg:end]
del test_data

# gather all data and save to disk
result = {"config": OmegaConf.to_yaml(CONF)}

for (dev, sec, dom) in train_umaps.keys():
    result[("train", dev, sec, dom)] = {
        "umaps": train_umaps[(dev, sec, dom)],
        "metadata": train_metadata[(dev, sec, dom)],
        "relative_idxs": train_rel_idxs[(dev, sec, dom)],
        "global_idxs": train_glob_idxs[(dev, sec, dom)]}

for (dev, sec, dom) in test_umaps.keys():
    result[("test", dev, sec, dom)] = {
        "umaps": test_umaps[(dev, sec, dom)],
        "metadata": test_metadata[(dev, sec, dom)],
        "relative_idxs": test_rel_idxs[(dev, sec, dom)],
        "global_idxs": test_glob_idxs[(dev, sec, dom)]}

result["audioset"] = {"umaps": audioset_umaps,
                      "global_idxs": audioset_glob_idxs,
                      "metadata": audioset_metadata,
                      "relative_idxs": audioset_rel_idxs}

result["fraunhofer"] = {"umaps": fraunhofer_umaps,
                        "global_idxs": fraunhofer_glob_idxs,
                        "metadata": fraunhofer_metadata,
                        "relative_idxs": fraunhofer_rel_idxs}


# save results to pickle:
with open(CONF.OUTPUT_PATH, "wb") as f:
    pickle.dump(result, f)
    LOGGER.info(f"Saved results to {CONF.OUTPUT_PATH}")
