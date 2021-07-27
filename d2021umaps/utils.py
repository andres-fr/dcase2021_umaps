#!/usr/bin python
# -*- coding:utf-8 -*-


"""
"""


import numpy as np
import h5py


# ##############################################################################
# # I/O
# ##############################################################################
class IncrementalHDF5:
    """
    Incrementally concatenate matrices of same height. Note the usage of very
    few datasets, to prevent slow loading times.
    """

    def __init__(self, out_path, height, dtype=np.float32, compression="lzf",
                 data_chunk_length=20, metadata_chunk_length=20):
        """
        :param height: This class incrementally stores a matrix of shape
          ``(height, w++)``, where ``height`` is always fixed.
        :param compression: ``lzf`` is fast, ``gzip`` slower but provides
          better compression
        """
        self.out_path = out_path
        self.height = height
        self.dtype = dtype
        self.compression = compression
        #
        self.h5f = h5py.File(out_path, "w")
        self.data_ds = self.h5f.create_dataset(
            "data", shape=(height, 0), maxshape=(height, None), dtype=dtype,
            compression=compression, chunks=(height, data_chunk_length))
        self.metadata_ds = self.h5f.create_dataset(
            "metadata", shape=(0,), maxshape=(None,), compression=compression,
            dtype=h5py.string_dtype(), chunks=(metadata_chunk_length,))
        self.data_idxs_ds = self.h5f.create_dataset(
            "data_idxs", shape=(2, 0), maxshape=(2, None), dtype=np.int64,
            compression=compression, chunks=(2, metadata_chunk_length))
        self._current_data_width = 0
        self._num_entries = 0

    def __enter__(self):
        """
        """
        return self

    def __exit__(self, type, value, traceback):
        """
        """
        self.close()

    def close(self):
        """
        """
        self.h5f.close()

    def append(self, matrix, metadata_str):
        """
        """
        n = self._num_entries
        h, w = matrix.shape
        assert h == self.height, \
            f"Shape was {(h, w)} but should be ({self.height}, ...). "
        # update arr size and add data
        new_data_w = self._current_data_width + w
        self.data_ds.resize((self.height, new_data_w))
        self.data_ds[:, self._current_data_width:new_data_w] = matrix
        # # update meta-arr size and add metadata
        self.metadata_ds.resize((n + 1,))
        self.metadata_ds[n] = metadata_str
        # update data-idx size and add entry
        self.data_idxs_ds.resize((2, n + 1))
        self.data_idxs_ds[:, n] = (self._current_data_width, new_data_w)
        #
        self.h5f.flush()
        self._current_data_width = new_data_w
        self._num_entries += 1
