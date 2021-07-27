#!/usr/bin python
# -*- coding:utf-8 -*-


"""
Dataset-specific functionality
"""


from pathlib import Path
import glob
import random
import json
#
import yaml
import h5py
import numpy as np
import pandas as pd
import librosa


# ##############################################################################
# # CLASS LABELS
# ##############################################################################
# these are properties of the native dataset
ALL_DEVICES = {"fan", "gearbox", "pump", "slider", "valve",
               "ToyCar", "ToyTrain"}
ALL_SECTIONS = {"00", "01", "02", "03", "04", "05"}
ALL_DOMAINS = {"source", "target"}
ALL_SPLITS = {"train", "test"}
# these are the pandas columns after converting
DEV_COLUMNS = ["device", "section", "domain", "split", "label", "ID",
               "extra", "path"]
EVAL_COLUMNS = ["device", "section", "domain", "split", "ID", "path"]

ANOMALY_LABELS = {"normal": 0, "anomaly": 1}

DOMAIN_LABELS = {("other", "other", "other"): NotImplemented,
                 ("ToyCar", "00", "source"): 1,
                 ("ToyCar", "00", "target"): 2,
                 ("ToyCar", "01", "source"): 3,
                 ("ToyCar", "01", "target"): 4,
                 ("ToyCar", "02", "source"): 5,
                 ("ToyCar", "02", "target"): 6,
                 ("ToyCar", "03", "source"): 7,
                 ("ToyCar", "03", "target"): 8,
                 ("ToyCar", "04", "source"): 9,
                 ("ToyCar", "04", "target"): 10,
                 ("ToyCar", "05", "source"): 11,
                 ("ToyCar", "05", "target"): 12,
                 ("ToyTrain", "00", "source"): 13,
                 ("ToyTrain", "00", "target"): 14,
                 ("ToyTrain", "01", "source"): 15,
                 ("ToyTrain", "01", "target"): 16,
                 ("ToyTrain", "02", "source"): 17,
                 ("ToyTrain", "02", "target"): 18,
                 ("ToyTrain", "03", "source"): 19,
                 ("ToyTrain", "03", "target"): 20,
                 ("ToyTrain", "04", "source"): 21,
                 ("ToyTrain", "04", "target"): 22,
                 ("ToyTrain", "05", "source"): 23,
                 ("ToyTrain", "05", "target"): 24,
                 ("fan", "00", "source"): 25,
                 ("fan", "00", "target"): 26,
                 ("fan", "01", "source"): 27,
                 ("fan", "01", "target"): 28,
                 ("fan", "02", "source"): 29,
                 ("fan", "02", "target"): 30,
                 ("fan", "03", "source"): 31,
                 ("fan", "03", "target"): 32,
                 ("fan", "04", "source"): 33,
                 ("fan", "04", "target"): 34,
                 ("fan", "05", "source"): 35,
                 ("fan", "05", "target"): 36,
                 ("gearbox", "00", "source"): 37,
                 ("gearbox", "00", "target"): 38,
                 ("gearbox", "01", "source"): 39,
                 ("gearbox", "01", "target"): 40,
                 ("gearbox", "02", "source"): 41,
                 ("gearbox", "02", "target"): 42,
                 ("gearbox", "03", "source"): 43,
                 ("gearbox", "03", "target"): 44,
                 ("gearbox", "04", "source"): 45,
                 ("gearbox", "04", "target"): 46,
                 ("gearbox", "05", "source"): 47,
                 ("gearbox", "05", "target"): 48,
                 ("pump", "00", "source"): 49,
                 ("pump", "00", "target"): 50,
                 ("pump", "01", "source"): 51,
                 ("pump", "01", "target"): 52,
                 ("pump", "02", "source"): 53,
                 ("pump", "02", "target"): 54,
                 ("pump", "03", "source"): 55,
                 ("pump", "03", "target"): 56,
                 ("pump", "04", "source"): 57,
                 ("pump", "04", "target"): 58,
                 ("pump", "05", "source"): 59,
                 ("pump", "05", "target"): 60,
                 ("slider", "00", "source"): 61,
                 ("slider", "00", "target"): 62,
                 ("slider", "01", "source"): 63,
                 ("slider", "01", "target"): 64,
                 ("slider", "02", "source"): 65,
                 ("slider", "02", "target"): 66,
                 ("slider", "03", "source"): 67,
                 ("slider", "03", "target"): 68,
                 ("slider", "04", "source"): 69,
                 ("slider", "04", "target"): 70,
                 ("slider", "05", "source"): 71,
                 ("slider", "05", "target"): 72,
                 ("valve", "00", "source"): 73,
                 ("valve", "00", "target"): 74,
                 ("valve", "01", "source"): 75,
                 ("valve", "01", "target"): 76,
                 ("valve", "02", "source"): 77,
                 ("valve", "02", "target"): 78,
                 ("valve", "03", "source"): 79,
                 ("valve", "03", "target"): 80,
                 ("valve", "04", "source"): 81,
                 ("valve", "04", "target"): 82,
                 ("valve", "05", "source"): 83,
                 ("valve", "05", "target"): 84}

DEVICE_CODES = {device: set(v for k, v in DOMAIN_LABELS.items()
                            if k[0] == device)
                for device in ALL_DEVICES}
SECTION_CODES = {section: set(v for k, v in DOMAIN_LABELS.items()
                              if k[1] == section)
                 for section in ALL_SECTIONS}
DOMAIN_CODES = {domain: set(v for k, v in DOMAIN_LABELS.items()
                            if k[2] == domain)
                for domain in ALL_DOMAINS}


# ##############################################################################
# # H5 HELPERS
# ##############################################################################
def fetch_from_h5(h5matrix, idxs):
    """
    Fetching from HDF5 datasets entails an overhead. For that reason, it's best
    to fetch multiple indexes at once. The constraint is that all indexes must
    be in non-repeating, strictly-increasing order. This function is a wrapper
    that allows to overcome that constraint maintaining a single fetch.
    Specifically, given a h5 matrix of shape ``(h, w)``, it fetches columns
    by their ``idxs``.

    At some point this function should be factored into HDF5Dataset.getitems().
    """
    unique_sorted_idxs = np.unique(idxs)
    unique_sorted_idxs.sort()
    data_fetch = h5matrix[:, unique_sorted_idxs]  # (h, N)
    mapping = dict(zip(unique_sorted_idxs, data_fetch.T))
    #
    data_out = np.empty((len(idxs), h5matrix.shape[0]), dtype=data_fetch.dtype)
    for i, idx in enumerate(idxs):
        data_out[i] = mapping[idx]
    return data_out


def get_umap_energies(umap_obj, devices=None, sections=None):
    """
    :param umap_obj: Dictionary like the ones pickled by the 01a UMAP creation
      script.
    :returns: For each ``umap`` in the given object, the corresponding energy
      via ``librosa.db_to_power`` and per-frame summation. For this reason it
      only produces meaningful results for LogMel and LogSTFT.
    """
    if devices is None:
        devices = ALL_DEVICES
    if sections is None:
        sections = ALL_SECTIONS
    # Load HDF5 files
    dct = yaml.safe_load(umap_obj["config"])
    train_h5 = h5py.File(dct["DCASE_TRAIN_PATH"], "r")
    test_h5 = h5py.File(dct["DCASE_TEST_PATH"], "r")
    audioset_h5 = h5py.File(dct["AUDIOSET_PATH"], "r")
    fraunhofer_h5 = h5py.File(dct["FRAUNHOFER_PATH"], "r")
    #
    result = {}
    for k, v in umap_obj.items():
        if k == "audioset":
            h5 = audioset_h5
        elif k == "fraunhofer":
            h5 = fraunhofer_h5
        elif k[0] == "train" and k[1] in devices and k[2] in sections:
            h5 = train_h5
        elif k[0] == "test" and k[1] in devices and k[2] in sections:
            h5 = test_h5
        else:
            print("Energies: ignored:", k)
            continue
        #
        print("Energies: processing", k)
        gidxs = v["global_idxs"]
        # fetch data. this may take a while
        data = fetch_from_h5(h5["data"], gidxs)
        energies = librosa.db_to_power(data).sum(axis=1)
        result[k] = energies
    #
    train_h5.close()
    test_h5.close()
    audioset_h5.close()
    fraunhofer_h5.close()
    #
    return result


# ##############################################################################
# # PANDAS ADAPTER
# ##############################################################################
class DCASE2021t2Frames:
    """
    This class loads the DCASE 2021, Task 2 metadata, given through file names,
    into 2 Pandas dataframes: one for entries with a label (normal/anomaly)
    which can be used for supervised training, and other with entries lacking a
    label (usually belonging to the test set, and prohibited to be trained on).

    It also provides static functionality to filter the dataframes by device,
    section, domain, etc. Check the class methods and attributes for more info.

    ..note::
      Instead of the original dev/eval split, we consider here a different split
      based on the rules and the nature of the data: eval is all the files
      without a label (this corresponds exactly to the DCASE eval figures), and
      our dev is the rest of the files (note that test data is *not* allowed for
      training). To recover the original datasets, simply filter by section:
      (00, 01, 02) for dev and (03, 04, 05) for eval.
    """

    def __init__(self, *dataset_paths):
        """
        Given the path(s) to the DCASE 2021 Task 2 dataset (i.e. a location
        with fan, gearbox... etc directories inside), this constructor collects
        all the wav files inside them into 2 Pandas dataframes:

        .. note::
          All files inside ``dataset_paths`` ending with ``.wav`` will be
          processed! Make sure all such files belong to the dataset.
        """
        dev_rows = []
        eval_rows = []
        for dp in dataset_paths:
            dp_matcher = str((Path(dp) / "**" / "*.wav").resolve())
            for wavpath in glob.iglob(dp_matcher, recursive=True):
                wavinfo = Path(wavpath).stem.split("_")
                device = Path(wavpath).parts[-3]
                is_test = "normal" not in wavinfo and "anomaly" not in wavinfo
                #
                if is_test:
                    section, domain, split, wav_id = wavinfo[1:]
                    eval_rows.append((device, section, domain, split, wav_id,
                                      wavpath))
                else:
                    section = wavinfo[1]
                    domain = wavinfo[2]
                    split = wavinfo[3]
                    label = wavinfo[4]
                    wav_id = wavinfo[5]
                    extra = wavinfo[6:]
                    dev_rows.append((device, section, domain, split, label,
                                     wav_id, extra, wavpath))
        # build and return datagrames
        self.dev_df = pd.DataFrame(dev_rows, columns=DEV_COLUMNS)
        self.eval_df = pd.DataFrame(eval_rows, columns=EVAL_COLUMNS)

    def query_dev(self, filter_device=None, filter_section=None,
                  filter_domain=None, filter_split=None, filter_label=None,
                  filter_id=None, filter_extra=None):
        """
        Filters are lambdas that receive an entry for the corresponding field,
        and return a boolean. If true, the entry is kept, otherwise discarded.
        If filter is ``None``, everything gets accepted.
        :returns: A COPY of the subset of ``self.dev_df`` with all-true rows.
        """
        df = self.dev_df
        cond = np.array([True for _ in range(len(df))])
        #
        if filter_device is not None:
            cond &= np.array([filter_device(x) for x in df["device"]])
        if filter_section is not None:
            cond &= np.array([filter_section(x) for x in df["section"]])
        if filter_domain is not None:
            cond &= np.array([filter_domain(x) for x in df["domain"]])
        if filter_split is not None:
            cond &= np.array([filter_split(x) for x in df["split"]])
        if filter_label is not None:
            cond &= np.array([filter_label(x) for x in df["label"]])
        if filter_id is not None:
            cond &= np.array([filter_id(x) for x in df["ID"]])
        if filter_extra is not None:
            cond &= np.array([filter_extra(x) for x in df["extra"]])
        #
        return df[cond].copy()

    def query_eval(self, filter_device=None, filter_section=None,
                   filter_domain=None, filter_split=None, filter_id=None):
        """
        See ``query_dev`` docstring
        """
        df = self.eval_df
        cond = np.array([True for _ in range(len(df))])
        #
        if filter_device is not None:
            cond &= np.array([filter_device(x) for x in df["device"]])
        if filter_section is not None:
            cond &= np.array([filter_section(x) for x in df["section"]])
        if filter_domain is not None:
            cond &= np.array([filter_domain(x) for x in df["domain"]])
        if filter_split is not None:
            cond &= np.array([filter_split(x) for x in df["split"]])
        if filter_id is not None:
            cond &= np.array([filter_id(x) for x in df["ID"]])
        #
        return df[cond].copy()


# ##############################################################################
# # PRECOMPUTED FEATURE LOADER
# ##############################################################################
class HDF5Dataset:
    """
    Given paths to HDF5 files (like the ones generated by IncrementalHDF5)
    the form:
    * "data": Pointing to a big numpy matrix of shape ``(h, w)`` that contains
      the whole dataset as concatenated matrices of same ``h``. Note that all
      of the given files are expected to have the same ``h`` and ``dtype``.
    * "data_idxs": Pairs of indexes signaling the start (included) and end
      (excluded) index of each file in data and metadata.
    * "metadata": Array of JSON strings, each one corresponding to the pair of
      indexes in same order.

    this dataset "merges" the files from all given paths, calculates how many
    chunks of the given size can be made, and retrieves them sorted by index,
    with the corresponding metadata. Furthermore, it can filter out files by
    their metadata. Usage example::

      dcase_ds = HDF5Dataset(DCASE_TRAIN_PATH, metadata_filter_fn=lambda md:
                       md["device"] == "gearbox" and
                       md["domain"] == "target")
      merged_ds = HDF5Dataset(DS1_PATH, DS2_PATH, chunk_length=3)

    Since interacting with the filesystem can be a performance bottleneck,
    this dataset also implements the ``getitems`` method, which admits and
    fetches multiple indexes in a single HDF5 access.
    """

    def __init__(self, *h5_paths, chunk_length=5, metadata_filter_fn=None):
        """
        :param metadata_filter_fn: A function with signature ``metadata->bool``,
          only elements with true output will be used.
        """
        self.h5_paths = h5_paths
        self.chunk_length = chunk_length
        #
        self.h5_files = [h5py.File(p, "r") for p in h5_paths]
        self.all_metadatas = [h5["metadata"][()] for h5 in self.h5_files]
        self.all_indexes = [h5["data_idxs"][()].T for h5 in self.h5_files]
        #
        (self.metadata, self.file_access,
         self.cumul_entries) = self.filter_metadata(metadata_filter_fn)
        #
        self.arrs = [h5["data"] for h5 in self.h5_files]
        self.height = self.arrs[0].shape[0]
        self.dtype = self.arrs[0].dtype
        assert all([a.shape[0] == self.height for a in self.arrs]), \
            "All HDF5 files must contain data matrix of same height!"
        assert all([a.dtype == self.dtype for a in self.arrs]), \
            "All HDF5 files must contain data matrix of same dtype!"

    def filter_metadata(self, filter_fn=None):
        """
        See constructor docstring
        """
        if filter_fn is None:
            filter_fn = lambda md: True
        # merge metadata entries from all given files, if they satisfy
        # filter_fn AND are shorter than given chunk_length
        metadata = {}
        for file_i, (mds, idxs) in enumerate(zip(self.all_metadatas,
                                                 self.all_indexes)):
            assert len(mds) == len(idxs), "This should never happen"
            for md, (beg, end) in zip(mds, idxs):
                if filter_fn(md) and (end - beg) >= self.chunk_length:
                    metadata[(file_i, beg, end)] = md
        # file_access: list of (file_idx, beg_included, end_excluded)
        file_access = sorted(metadata, key=lambda x: (x[0], x[1]))
        chunks_per_file =  [(end - beg) + 1 - self.chunk_length
                            for _, beg, end in file_access]
        cumul_entries = np.cumsum([0] + chunks_per_file)
        #
        return metadata, file_access, cumul_entries

    def __len__(self):
        """
        """
        return self.cumul_entries[-1]

    def getitems(self, idxs):
        """
        Getting multiple idxs at once from the HDF5 file speeds up sampling.
        :returns: ``(data, arr_idxs, global_idxs, metadatas, file_rel_idxs)``,
          corresponding to the returned data of shape ``(N, bins, chunk)``,
          the corresponding ``N`` array indexes and per-array global indexes,
          the corresponding ``N`` metadata strings and the ``N`` indexes of each
          chunk beginning with respect to the file beginning. This information
          should be enough to locate the returned data chunk in the dataset and
          other HDF5 files.
        """
        arr_idxs = []
        global_idxs = []
        metadatas = []
        file_relative_idxs = []
        for i in idxs:
            # convert dataset idx into file idx and within-file frame idx:
            access_idx = np.searchsorted(
                self.cumul_entries, i, side="right") - 1
            rel_idx = i - self.cumul_entries[access_idx]
            # convert access_idx and relative_idx to the global "data" idx.
            arr_idx, file_beg_idx, file_end_idx = self.file_access[access_idx]
            metadata = self.metadata[(arr_idx, file_beg_idx, file_end_idx)]
            global_idx = file_beg_idx + rel_idx
            #
            arr_idxs.append(arr_idx)
            global_idxs.append(global_idx)
            metadatas.append(metadata)
            file_relative_idxs.append(rel_idx)
        # To each index, add its N following neighbours
        arr_idxs = np.array(arr_idxs)
        all_glob_idxs = np.repeat(global_idxs, self.chunk_length).reshape(
            -1, self.chunk_length) + range(self.chunk_length)
        data_out = np.empty((len(idxs), self.height, self.chunk_length),
                            dtype=self.dtype)  # (N, bins, chunk)
        # we fetch only once per HDF5 file:
        for i, arr in enumerate(self.arrs):
            this_arr_idxs = np.where(arr_idxs==i)[0]
            this_glob_idxs = all_glob_idxs[this_arr_idxs]  # (N_i, stack)
            #
            unique_sorted_idxs = np.unique(this_glob_idxs)
            unique_sorted_idxs.sort()
            # Fetch data from disk!
            data_fetch = arr[:, unique_sorted_idxs]  # (bins, N)
            # map a given glob idx (of this array) back to the joint output
            mapping = dict(zip(unique_sorted_idxs, data_fetch.T))
            for data_idx, this_g in zip(this_arr_idxs, this_glob_idxs):
                for g_i, g in enumerate(this_g):
                    data_out[data_idx, :, g_i] = mapping[g]
        # at this point we have data(N, bins, chunk),
        return data_out, arr_idxs, global_idxs, metadatas, file_relative_idxs

    def __getitem__(self, idx):
        """
        See ``getitems`` docstring
        """
        dt, ai, gi, md, rel = self.getitems([idx])
        # we only fetched 1 element
        data = dt[0]
        arr_idx = ai[0]
        global_idx0 = gi[0]
        metadata = md[0]
        file_relative_idx = rel[0]
        #
        return data, arr_idx, global_idx0, metadata, file_relative_idx


class DcaseHDF5Dataset(HDF5Dataset):
    """
    Like HDF5Dataset but incorporates domain-specific filter functions,
    and expects 1 path only.
    """

    def __init__(self, dcase_h5_path, chunk_length=5,
                 filter_devices=None, filter_sections=None,
                 filter_domains=None, filter_splits=None):
        """
        """
        # these are like class attributes, but less manual work, LOC, bugs
        if filter_devices is None:
            filter_devices = ALL_DEVICES
        if filter_sections is None:
            filter_sections = ALL_SECTIONS
        if filter_domains is None:
            filter_domains = ALL_DOMAINS
        if filter_domains is None:
            filter_domains = ALL_SPLITS
        self.filter_devices = filter_devices
        self.filter_sections = filter_sections
        self.filter_domains = filter_domains
        self.filter_splits = filter_splits
        #
        super().__init__(dcase_h5_path,
                         chunk_length=chunk_length,
                         metadata_filter_fn=self._metadata_filter)

    def _metadata_filter(self, md):
        md = json.loads(md)
        return (md["device"] in self.filter_devices and
                md["section"] in self.filter_sections and
                md["domain"] in self.filter_domains and
                md["split"]  in self.filter_splits)

    def getitems(self, idxs):
        """
        Like ``super().getitems``, but doesn't return array indexes since we
        only have one array.
        """
        data, _, global_idxs, metadatas, rel_idxs = super().getitems(idxs)
        return data, global_idxs, metadatas, rel_idxs

    def __getitem__(self, idx):
        """
        See ``getitems`` docstring
        """
        dt, gi, md, rel = self.getitems([idx])
        # we only fetched 1 element
        data = dt[0]
        global_idx0 = gi[0]
        metadata = md[0]
        file_relative_idx = rel[0]
        #
        return data, global_idx0, metadata, file_relative_idx
