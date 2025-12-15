"""Contains function for extracting timestamps from a trigger channel."""

# License: BSD-3-Clause
# Copyright (c) 2014 BioMag Laboratory, Helsinki University Central Hospital
# Copyright (c) 2025 Aalto University

import logging

import mne
import numpy as np
import numpy.typing as npt

# Parameters for detecting timestamps. Should match the parameters used for
# generating the timing sequence
_BASELINE = 5  # seconds
_TRAIN_INTRVL = 10  # seconds
_TRAIN_STEP = 0.015  # seconds
_NBITS = 43  # including the parity bit

logger = logging.getLogger(__name__)


def compute_raw_timestamps(
    raw: mne.io.Raw, timing_channel: str
) -> npt.NDArray[np.float64]:
    """Get the timestamps from raw data having Helsinki VideoMEG timing channel.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw data object.
    timing_channel : str
        Channel name string for the timing channel.

    Returns
    -------
    NDArray[np.float64]
        Array of timestamps corresponding to each sample in the raw data.
    """
    timing_data = raw.get_data(picks=timing_channel, return_times=False)
    assert isinstance(timing_data, np.ndarray), (
        "With return_times=False, timing data should be an ndarray"
    )
    timing_data = timing_data.squeeze()  # remove the channel dimension

    return _comp_tstamps(timing_data, raw.info["sfreq"])


def _read_timestamp(dtrigs, cur, step, nbits):
    """Read and decode one timestamp.

    Return the timestamp on success or -1 otherwise.
    """
    ts = 0
    parity = False

    if cur + nbits >= len(dtrigs):
        logger.warning("end of input reached before all the bits read")
        return -1

    # Read the bits
    for i in range(nbits):
        # check the interval between the two triggers
        if (dtrigs[cur + i + 1] < step * 1.5) or (dtrigs[cur + i + 1] > step * 4.5):
            logger.warning("invalid interval between two triggers")
            return -1

        # check whether the next bit is 0 or 1
        if dtrigs[cur + i + 1] > step * 3:
            parity = not parity

            if i < nbits - 1:  # don't read the parity bit into the timestamp
                ts = ts + 2**i

    if parity:
        logger.warning("parity check failed")
        return -1
    else:
        return ts


def _comp_tstamps_1bit(inp, sfreq) -> npt.NDArray[np.float64]:
    """Extract timestamps from a "normal" (not composite) trigger channel.

    Parameters
    ----------
        inp - vector of samples for the trigger channel
        sfreq - sampling frequency

    Return the vector of the same length as inp, containing timestamps for
    each entry of inp. For detecting timestamps use parameters in the beginning
    of the file. Assume that the input values are either 0 or 1.

    TODO: this function does not handle the boundary case for the first train
    of pulses correctly. This is because there is no trigger before the train
    and there will be no dtrigs value before the first trigger of the train.
    Thus the first pulse train will always be ignored. It would be neat to fix
    this.
    """
    THRESH = 0.5

    # input should be a 1-d vector
    assert inp.ndim == 1

    # find all triggers (threshold crossings)
    trigs = np.where((inp[:-1] < THRESH) & (inp[1:] > THRESH))[0] + 1

    # iterate over all timestamp candidates
    samps = []
    tss = []
    dtrigs = np.diff(trigs)

    for i in np.where(dtrigs > _BASELINE * sfreq)[0]:
        ts = _read_timestamp(dtrigs, i, _TRAIN_STEP * sfreq, _NBITS)

        if ts != -1:
            samps.append(trigs[i + 1])
            tss.append(ts)

    # do some sanity checking
    if len(tss) < 2:
        raise Exception("Less than 2 timestamps found")

    if len(tss) * _TRAIN_INTRVL * sfreq < len(inp) * 0.1:
        raise Exception("Too few timestamps detected")

    # fit timestamps to samples with linear regression
    p = np.polyfit(samps, tss, 1)
    data_tstamps = np.polyval(p, np.arange(len(inp)))
    errs = np.abs(np.polyval(p, samps) - tss)

    if data_tstamps.dtype != np.float64:
        logger.warning(
            f"Data type of raw data timestamps is {data_tstamps.dtype}, "
            "converting to float64"
        )
    data_tstamps = data_tstamps.astype(np.float64)

    logger.info(
        f"Raw timestamp computation: regression fit errors (abs): mean {errs.mean():f},"
        f" median {np.median(errs):f}, max {errs.max():f}"
    )

    return data_tstamps


def _comp_tstamps(inp, sfreq) -> npt.NDArray[np.float64]:
    """Extract timestamps from a trigger channel.

    Parameters
    ----------
        inp - vector of samples for the trigger channel
        sfreq - sampling frequency

    Check individual bits of the inp to see whether any of them contains timing
    information. Return the vector of the same length as inp, containing
    timestamps for each entry of inp.
    """
    if inp.min() < 0:
        raise ValueError("Negative values in composite? channel")

    # Try different bits until succeeding or running out of the bits
    while inp.max() > 0:
        try:
            return _comp_tstamps_1bit(inp % 2, sfreq)
        except Exception:
            inp = inp // 2

    raise Exception("No timing information found")
