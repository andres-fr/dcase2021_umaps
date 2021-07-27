#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
Precompute Fraunhofer fixed representations (logSTFT, logMel)
"""


import os
from pathlib import Path
#
from omegaconf import OmegaConf
import numpy as np
#
from d2021umaps.utils import IncrementalHDF5
from d2021umaps.logging import ColorLogger, make_timestamp
from d2021umaps.features import wavpath_to_mel, wavpath_to_stft


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
    "Please provide a ROOT_PATH=... containing train_cut and test_cut!"
CONF.ROOT_PATH = str(Path(CONF.ROOT_PATH).resolve())  # in case of softlinks

# these variables may depend on CLI input so we set them at the end
TRAIN_PATH = os.path.join(CONF.ROOT_PATH, "train_cut")
TEST_PATH = os.path.join(CONF.ROOT_PATH, "test_cut")
STFT_FREQBINS = int(CONF.STFT_WINSIZE / 2 + 1)
STFT_OUTPATH = os.path.join(
    CONF.OUT_DIR,
    f"fraunhofer_wavnorm={CONF.WAV_NORM}_stft_win{CONF.STFT_WINSIZE}_" +
    f"hop{CONF.STFT_HOPSIZE}.h5")
MEL_OUTPATH = os.path.join(
    CONF.OUT_DIR,
    f"fraunhofer_wavnorm={CONF.WAV_NORM}_mel_win{CONF.STFT_WINSIZE}_" +
    f"hop{CONF.STFT_HOPSIZE}_m{CONF.NUM_MELS}.h5")


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
LOGGER = ColorLogger(__file__, CONF.LOG_OUTPATH, filemode="w")
LOGGER.info(f"\n\n\nSTARTED SCRIPT: {__file__}")
LOGGER.info(OmegaConf.to_yaml(CONF))


def save_stft_dataset(out_path, *paths, in_db=True, root_path=None):
    """
    """
    ds_len = len(paths)
    with IncrementalHDF5(out_path, STFT_FREQBINS, np.float32) as ihdf5:
        LOGGER.info(f"Writing to {out_path}")
        for i, abspath in enumerate(paths, 1):
            if root_path is not None:
                metadata_str = str(abspath.relative_to(root_path))
            else:
                metadata_str = str(abspath)
            if i % 100 == 0:
                LOGGER.info(f"[{i}/{ds_len}] save_stft_dataset: {metadata_str}")
            arr = wavpath_to_stft(
                str(abspath), CONF.WAV_SR, wav_norm=CONF.WAV_NORM,
                n_fft=CONF.STFT_WINSIZE, hop_length=CONF.STFT_HOPSIZE,
                pad_mode="constant", in_decibels=in_db, logger=LOGGER)
            if arr is None:  # if None, wav had zero samples
                continue
            ihdf5.append(arr, metadata_str)
            # check that file is indeed storing the exact array
            _, arr_w = arr.shape
            assert (arr == ihdf5.data_ds[:, -arr_w:]).all(), \
                "Should never happen"
        LOGGER.info(f"Finished writing to {out_path}")


def save_mel_dataset(out_path, *paths, in_db=True, root_path=None):
    """
    """
    ds_len = len(paths)
    with IncrementalHDF5(out_path, CONF.NUM_MELS, np.float32) as ihdf5:
        LOGGER.info(f"Writing to {out_path}")
        for i, abspath in enumerate(paths, 1):
            if root_path is not None:
                metadata_str = str(abspath.relative_to(root_path))
            else:
                metadata_str = str(abspath)
            if i % 100 == 0:
                LOGGER.info(f"[{i}/{ds_len}] save_mel_dataset: {metadata_str}")
            arr = wavpath_to_mel(
                str(abspath), CONF.WAV_SR, wav_norm=CONF.WAV_NORM,
                n_mels=CONF.NUM_MELS, hop_length=CONF.STFT_HOPSIZE,
                pad_mode="constant", in_decibels=in_db, logger=LOGGER)
            if arr is None:  # if None, wav had zero samples
                continue
            ihdf5.append(arr, metadata_str)
            # check that file is indeed storing the exact array
            _, arr_w = arr.shape
            assert (arr == ihdf5.data_ds[:, -arr_w:]).all(), \
                "Should never happen"
        LOGGER.info(f"Finished writing to {out_path}")


train_paths = list(Path(TRAIN_PATH).glob("**/*.wav"))
test_paths = list(Path(TEST_PATH).glob("**/*.wav"))
paths = train_paths + test_paths
save_mel_dataset(MEL_OUTPATH, *paths, root_path=CONF.ROOT_PATH)
save_stft_dataset(STFT_OUTPATH, *paths, root_path=CONF.ROOT_PATH)
