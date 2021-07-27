#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
Precompute DCASE2021 Task 2 Dataset fixed representations (logSTFT, logMel)
"""


import os
from pathlib import Path
import json
#
from omegaconf import OmegaConf
import numpy as np
#
from d2021umaps.utils import IncrementalHDF5
from d2021umaps.logging import ColorLogger, make_timestamp
from d2021umaps.features import wavpath_to_mel, wavpath_to_stft
from d2021umaps.data import DCASE2021t2Frames


# ##############################################################################
# # GLOBALS
# ##############################################################################
CONF = OmegaConf.create()

CONF.ROOT_PATH = None  # must be given by user!
#
CONF.WAV_NORM = "none"
CONF.WAV_SR = 16000  # WAVs will be resampled to this when loaded
CONF.STFT_WINSIZE = 1024  # powers of 2 ideally
CONF.STFT_HOPSIZE = 512
CONF.NUM_MELS = 128
CONF.OUT_DIR = "precomputed_features"

log_ts = make_timestamp(timezone="Europe/London", with_tz_output=False)
CONF.LOG_OUTPATH = os.path.join("logs", "{}_[{}].log".format(log_ts, __file__))

cli_conf = OmegaConf.from_cli()
CONF = OmegaConf.merge(CONF, cli_conf)

assert CONF.ROOT_PATH is not None, \
    "Please provide a ROOT_PATH=... containing the DCASE dev and eval folders"
CONF.ROOT_PATH = str(Path(CONF.ROOT_PATH).resolve())  # in case of softlinks


# these variables may depend on CLI input so we set them at the end
STFT_FREQBINS = int(CONF.STFT_WINSIZE / 2 + 1)
DEV_PATH = os.path.join(CONF.ROOT_PATH, "dev")
EVAL_PATH = os.path.join(CONF.ROOT_PATH, "eval")
STFT_OUTPATH_TRAIN = os.path.join(
    CONF.OUT_DIR,
    f"dcase2021_t2_train_wavnorm={CONF.WAV_NORM}_stft_win{CONF.STFT_WINSIZE}_" +
    f"hop{CONF.STFT_HOPSIZE}.h5")
STFT_OUTPATH_CV = os.path.join(
    CONF.OUT_DIR,
    f"dcase2021_t2_cv_wavnorm={CONF.WAV_NORM}_stft_win{CONF.STFT_WINSIZE}_" +
    f"hop{CONF.STFT_HOPSIZE}.h5")
MEL_OUTPATH_TRAIN = os.path.join(
    CONF.OUT_DIR,
    f"dcase2021_t2_train_wavnorm={CONF.WAV_NORM}_mel_win{CONF.STFT_WINSIZE}_" +
    f"hop{CONF.STFT_HOPSIZE}_m{CONF.NUM_MELS}.h5")
MEL_OUTPATH_CV = os.path.join(
    CONF.OUT_DIR,
    f"dcase2021_t2_cv_wavnorm={CONF.WAV_NORM}_mel_win{CONF.STFT_WINSIZE}_" +
    f"hop{CONF.STFT_HOPSIZE}_m{CONF.NUM_MELS}.h5")


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
LOGGER = ColorLogger(__file__, CONF.LOG_OUTPATH, filemode="w")
LOGGER.info(f"\n\n\nSTARTED SCRIPT: {__file__}")
LOGGER.info(OmegaConf.to_yaml(CONF))


def save_stft_dataset(out_path, df_dataset, in_db=True, root_path=None):
    """
    """
    ds_len = len(df_dataset)
    with IncrementalHDF5(out_path, STFT_FREQBINS, np.float32) as ihdf5:
        LOGGER.info(f"Writing to {out_path}")
        for i, (_, row) in enumerate(df_dataset.iterrows(), 1):
            arr = wavpath_to_stft(row["path"], CONF.WAV_SR,
                                  wav_norm=CONF.WAV_NORM,
                                  n_fft=CONF.STFT_WINSIZE,
                                  hop_length=CONF.STFT_HOPSIZE,
                                  pad_mode="constant", in_decibels=in_db,
                                  logger=LOGGER)
            #
            rowp = Path(row["path"])
            metadata = row.to_dict()
            if root_path is not None:
                metadata["path"] = str(rowp.relative_to(root_path))
            else:
                metadata["path"] = rowp.name
            if i%1000 == 0:
                LOGGER.info(f"[{i}/{ds_len}] stft_dataset: {metadata}")
            ihdf5.append(arr, json.dumps(metadata))
            # check that file is indeed storing the exact array
            _, arr_w = arr.shape
            assert (arr == ihdf5.data_ds[:, -arr_w:]).all(), \
                "Should never happen"
        LOGGER.info(f"Finished writing to {out_path}")


def save_mel_dataset(out_path, df_dataset, in_db=True, root_path=None):
    """
    """
    ds_len = len(df_dataset)
    with IncrementalHDF5(out_path, CONF.NUM_MELS, np.float32) as ihdf5:
        LOGGER.info(f"Writing to {out_path}")
        for i, (_, row) in enumerate(df_dataset.iterrows(), 1):
            arr = wavpath_to_mel(
                row["path"], CONF.WAV_SR, wav_norm=CONF.WAV_NORM,
                n_mels=CONF.NUM_MELS,
                n_fft=CONF.STFT_WINSIZE, hop_length=CONF.STFT_HOPSIZE,
                pad_mode="constant", in_decibels=in_db, logger=LOGGER)
            #
            rowp = Path(row["path"])
            metadata = row.to_dict()
            if root_path is not None:
                metadata["path"] = str(rowp.relative_to(root_path))
            else:
                metadata["path"] = rowp.name
            if i%1000 == 0:
                LOGGER.info(f"[{i}/{ds_len}] mel_dataset: {metadata}")
            ihdf5.append(arr, json.dumps(metadata))
            # check that file is indeed storing the exact array
            _, arr_w = arr.shape
            assert (arr == ihdf5.data_ds[:, -arr_w:]).all(), \
                "Should never happen"
        LOGGER.info(f"Finished writing to {out_path}")


dcase_df = DCASE2021t2Frames(DEV_PATH, EVAL_PATH)
dcase_train = dcase_df.query_dev(filter_split=lambda x: x=="train")
dcase_cv = dcase_df.query_dev(filter_split=lambda x: x=="test")
#
save_mel_dataset(MEL_OUTPATH_CV, dcase_cv, root_path=CONF.ROOT_PATH)
save_stft_dataset(STFT_OUTPATH_CV, dcase_cv, root_path=CONF.ROOT_PATH)
# these are bigger
save_mel_dataset(MEL_OUTPATH_TRAIN, dcase_train, root_path=CONF.ROOT_PATH)
save_stft_dataset(STFT_OUTPATH_TRAIN, dcase_train, root_path=CONF.ROOT_PATH)
