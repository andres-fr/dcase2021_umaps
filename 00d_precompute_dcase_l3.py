#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
Precompute DCASE2021 Task 2 Dataset L3 representations
"""


import os
from pathlib import Path
import json
#
from omegaconf import OmegaConf
import numpy as np
import openl3
#
from d2021umaps.utils import IncrementalHDF5
from d2021umaps.logging import ColorLogger, make_timestamp
from d2021umaps.features import wavpath_to_arr
from d2021umaps.data import DCASE2021t2Frames


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
    "Please provide a ROOT_PATH=... containing the DCASE dev and eval folders"
CONF.ROOT_PATH = str(Path(CONF.ROOT_PATH).resolve())  # in case of softlinks

# these variables may depend on CLI input so we set them at the end
DEV_PATH = os.path.join(CONF.ROOT_PATH, "dev")
EVAL_PATH = os.path.join(CONF.ROOT_PATH, "eval")
OUTPATH_TRAIN = os.path.join(
    CONF.OUT_DIR,
    f"dcase2021_t2_train_wavnorm={CONF.WAV_NORM}_l3{CONF.L3_DOMAIN}_" +
    f"hop{CONF.L3_HOP_SECONDS}_{CONF.L3_SOURCE}{CONF.L3_DIMS}.h5")
OUTPATH_CV = os.path.join(
    CONF.OUT_DIR,
    f"dcase2021_t2_cv_wavnorm={CONF.WAV_NORM}_l3{CONF.L3_DOMAIN}_" +
    f"hop{CONF.L3_HOP_SECONDS}_{CONF.L3_SOURCE}{CONF.L3_DIMS}.h5")


# ##############################################################################
# # MAIN ROUTINE
# ##############################################################################
LOGGER = ColorLogger(__file__, CONF.LOG_OUTPATH, filemode="w")
LOGGER.info(f"\n\n\nSTARTED SCRIPT: {__file__}")
LOGGER.info(OmegaConf.to_yaml(CONF))


def save_l3_dataset(out_path, df_dataset, root_path=None):
    """
    """
    ds_len = len(df_dataset)
    with IncrementalHDF5(out_path, CONF.L3_DIMS, np.float32) as ihdf5:
        LOGGER.info(f"Writing to {out_path}")
        # we run multiple files at once to greatly speed up L3 process
        for i in range(0, ds_len, CONF.NUM_FILES_PER_L3_RUN):
            LOGGER.info(f"[{i}/{ds_len}] Processing L3")
            arrs = [wavpath_to_arr(row["path"], CONF.WAV_SR, CONF.WAV_NORM,
                                   logger=LOGGER)[0] for (_, row)
                    in df_dataset[i:i+CONF.NUM_FILES_PER_L3_RUN].iterrows()]
            embeddings, timestamps = openl3.get_audio_embedding(
              arrs, CONF.WAV_SR, center=False,
              content_type=CONF.L3_DOMAIN, input_repr=CONF.L3_SOURCE,
              embedding_size=CONF.L3_DIMS, batch_size=CONF.L3_BATCHSIZE,
              hop_size=CONF.L3_HOP_SECONDS)
            metadatas = [r.to_dict() for (_, r) in
                         df_dataset[i:i+CONF.NUM_FILES_PER_L3_RUN].iterrows()]
            for j, (emb, md, ts) in enumerate(zip(embeddings, metadatas,
                                                  timestamps)):
                pp = Path(md["path"])
                # timestamps aren't needed: start on 0 and add L3_HOP_SECONDS
                # md["timestamps"] = [round(x, 3) for x in ts.tolist()]
                if root_path is not None:
                    md["path"] = str(pp.relative_to(root_path))
                else:
                    md["path"] = pp.name
                ihdf5.append(emb.T, json.dumps(md))
                # check that file is indeed storing the exact array
                arr_w, l3_dims = emb.shape
                assert (emb.T == ihdf5.data_ds[:, -arr_w:]).all(), \
                    "Should never happen"
                assert l3_dims == CONF.L3_DIMS, "Should never happen (dims)"
        LOGGER.info(f"Finished writing to {out_path}")


dcase_df = DCASE2021t2Frames(DEV_PATH, EVAL_PATH)
dcase_train = dcase_df.query_dev(filter_split=lambda x: x=="train")
dcase_cv = dcase_df.query_dev(filter_split=lambda x: x=="test")
#
save_l3_dataset(OUTPATH_CV, dcase_cv, root_path=CONF.ROOT_PATH)
save_l3_dataset(OUTPATH_TRAIN, dcase_train, root_path=CONF.ROOT_PATH)
