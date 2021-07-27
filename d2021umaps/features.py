#!/usr/bin python
# -*- coding:utf-8 -*-


"""
Helper functions for fixed-feature audio computations
"""


import librosa


# ##############################################################################
# # WAV FEATURE EXTRACTORS
# ##############################################################################
def wavpath_to_arr(path, sr=None, norm="none", min_samples=1, logger=None):
    """
    :param sr: see ``librosa.load``
    :param str norm: if ``"std"`` or ``"absmax"``, scale the corresponding
      magnitude to 1. Otherwise don't scale.
    :returns: A float numpy array of shape ``(num_samples)``, or None
      if the wav had less than ``min_samples``.
    """
    try:
        arr, sr = librosa.load(path, sr=sr, mono=True)
        # optionally normalize audio
        arr -= arr.mean()
        # if audio has all-zeros, just warn and continue
        if arr.std() == 0 or abs(arr).max() == 0:
            if logger is not None:
                logger.warning(f"All-zeros wav file: {path}")
            else:
                print((f"All-zeros wav file: {path}"))
        elif norm == "std":
            arr /= arr.std()
        elif norm == "absmax":
            arr /= abs(arr).max()
        return arr, sr
    # If e.g. audio has zero samples, ignore and return None
    except ValueError as ve:
        arr, sr = librosa.load(path, sr=None, mono=True)
        if len(arr) < min_samples:
            if logger is not None:
                logger.warning(
                    f"Wav file too short! ({len(arr)}), ignored: {path}")
            else:
                logger.warning(
                    f"Wav file too short! ({len(arr)}), ignored: {path}")
            return None
        else:
            raise ve


def wavpath_to_stft(wavpath, wav_sr=None, wav_norm="none", in_decibels=True,
                    n_fft=1024, hop_length=512, pad_mode="constant",
                    logger=None, min_wav_samples=1):
    """
    See ``wavpath_to_arr`` docstring.

    :param pad_mode: Librosa default is 'reflect'. Constant pads with zeros
    :param in_decibels: If false, a power STFT is returned. If true, the
      corresponding power in decibel scale.
    :returns: A float numpy array of shape ``(num_bins, num_frames)``, or None
      if the wav had less than ``min_wav_samples``.
    """
    loaded_wav = wavpath_to_arr(wavpath, wav_sr, wav_norm, min_wav_samples,
                                logger)
    if loaded_wav is None:
        return None
    else:
        arr, sr = loaded_wav
    # compute STFT (and optionally log)
    result = abs(librosa.stft(arr, n_fft, hop_length, pad_mode=pad_mode)) ** 2
    if in_decibels:
        result = librosa.power_to_db(result)
    #
    return result


def wavpath_to_mel(wavpath, wav_sr=None, wav_norm="none", in_decibels=True,
                   n_fft=1024, hop_length=512, n_mels=128, pad_mode="constant",
                   logger=None, min_wav_samples=1):
    """
    See ``wavpath_to_arr`` docstring.

    :param pad_mode: Librosa default is 'reflect'. Constant pads with zeros
    :param in_decibels: If false, a power melgram is returned. If true, the
      corresponding power in decibel scale.
    :returns: A float numpy array of shape ``(num_mels, num_frames)``, or None
      if the wav had less than ``min_wav_samples``.
    """
    loaded_wav = wavpath_to_arr(wavpath, wav_sr, wav_norm, min_wav_samples,
                                logger)
    if loaded_wav is None:
        return None
    else:
        arr, sr = loaded_wav
    # compute mel (and optionally logmel)
    result = librosa.feature.melspectrogram(
        arr, sr=sr, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length,
        pad_mode=pad_mode, power=2.0)
    if in_decibels:
        result = librosa.power_to_db(result)
    #
    return result
