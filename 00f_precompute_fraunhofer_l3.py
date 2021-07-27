#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
Precompute Fraunhofer L3 representations
"""


import os
from pathlib import Path
#
from omegaconf import OmegaConf
import numpy as np
import openl3
#
from d2021umaps.utils import IncrementalHDF5
from d2021umaps.logging import ColorLogger, make_timestamp
from d2021umaps.features import wavpath_to_arr


# OpenL3 uses TF2. This prevents "CUBLAS_STATUS_NOT_INITIALIZED" when using CUDA
# https://stackoverflow.com/a/55541385
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# ##############################################################################
# # GLOBALS
# ##############################################################################
CONF = OmegaConf.create()

CONF.ROOT_PATH = None  # must be given by user!
#
CONF.WAV_NORM = "none"
CONF.WAV_SR = 16000  # WAVs will be resampled to this when loaded
# OPENL3 OPTIONS: https://openl3.readthedocs.io/en/latest/tutorial.html
CONF.L3_DOMAIN = "env"  # music
CONF.L3_SOURCE = "linear"  # mel128, mel256
CONF.L3_DIMS = 512
CONF.L3_HOP_SECONDS = 0.1
CONF.L3_BATCHSIZE = 16
CONF.NUM_FILES_PER_L3_RUN = 200
CONF.OUT_DIR = "precomputed_features"

log_ts = make_timestamp(timezone="Europe/London", with_tz_output=False)
CONF.LOG_OUTPATH = os.path.join("logs", "{}_[{}].log".format(log_ts, __file__))

cli_conf = OmegaConf.from_cli()
CONF = OmegaConf.merge(CONF, cli_conf)

assert CONF.ROOT_PATH is not None, \
    "Please provide a ROOT_PATH=... containing the wav files"
CONF.ROOT_PATH = str(Path(CONF.ROOT_PATH).resolve())  # in case of softlinks

# these variables may depend on CLI input so we set them at the end
TRAIN_PATH = os.path.join(CONF.ROOT_PATH, "train_cut")
TEST_PATH = os.path.join(CONF.ROOT_PATH, "test_cut")
OUTPATH = os.path.join(
    CONF.OUT_DIR,
    f"fraunhofer_wavnorm={CONF.WAV_NORM}_l3{CONF.L3_DOMAIN}_" +
    f"hop{CONF.L3_HOP_SECONDS}_{CONF.L3_SOURCE}{CONF.L3_DIMS}.h5")


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
LOGGER = ColorLogger(__file__, CONF.LOG_OUTPATH, filemode="w")
LOGGER.info(f"\n\n\nSTARTED SCRIPT: {__file__}")
LOGGER.info(OmegaConf.to_yaml(CONF))


def save_l3_dataset(out_path, *paths, root_path=None):
    """
    """
    ds_len = len(paths)
    with IncrementalHDF5(out_path, CONF.L3_DIMS, np.float32) as ihdf5:
        LOGGER.info(f"Writing to {out_path}")
        # we run multiple files at once to greatly speed up L3 process
        for i in range(0, ds_len, CONF.NUM_FILES_PER_L3_RUN):
            LOGGER.info(f"[{i}/{ds_len}] Processing L3")
            arrs = []
            pp = []
            for p in paths[i:i+CONF.NUM_FILES_PER_L3_RUN]:
                loaded_wav = wavpath_to_arr(str(p), CONF.WAV_SR, CONF.WAV_NORM,
                                            logger=LOGGER)
                if loaded_wav is not None:
                    arrs.append(loaded_wav[0])
                    pp.append(p)
            embeddings, _ = openl3.get_audio_embedding(
              arrs, CONF.WAV_SR, center=False,
              content_type=CONF.L3_DOMAIN, input_repr=CONF.L3_SOURCE,
              embedding_size=CONF.L3_DIMS, batch_size=CONF.L3_BATCHSIZE,
              hop_size=CONF.L3_HOP_SECONDS)
            for emb, p in zip(embeddings, pp):
                emb = emb.T
                if root_path is not None:
                    p = str(p.relative_to(root_path))
                else:
                    p = p.name
                # timestamps aren't needed: start on 0 and add L3_HOP_SECONDS
                # md["timestamps"] = [round(x, 3) for x in ts.tolist()]
                ihdf5.append(emb, p)
                # check that file is indeed storing the exact array
                l3_dims, arr_w = emb.shape
                assert (emb == ihdf5.data_ds[:, -arr_w:]).all(), \
                    "Should never happen"
                assert l3_dims == CONF.L3_DIMS, "Should never happen (dims)"
        LOGGER.info(f"Finished writing to {out_path}")


train_paths = list(Path(TRAIN_PATH).glob("**/*.wav"))
test_paths = list(Path(TEST_PATH).glob("**/*.wav"))
paths = train_paths + test_paths
assert len(set(paths)) == len(paths), "Should never happen (paths)"
save_l3_dataset(OUTPATH, *paths, root_path=CONF.ROOT_PATH)
