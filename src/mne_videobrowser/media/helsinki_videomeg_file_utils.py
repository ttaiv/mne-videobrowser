"""Functions shared between Helsinki VideoMEG project audio and video file readers."""

# License: BSD-3-Clause
# Copyright (c) 2014 BioMag Laboratory, Helsinki University Central Hospital

import struct


class UnknownVersionError(Exception):
    """Error due to unknown file version."""

    pass


def read_block_attributes(data_file, ver: int) -> tuple[int, int, int]:
    """Read attributes of a data block.

    Reads the header in the beginning of a data block, advancing the file
    position to the payload part of the data block (right after the header).
    After calling this function, you can read the payload of the data block
    with data_file.read(sz).


    If cannot read the attributes (EOF?), return -1 in ts.

    Parameters
    ----------
    data_file : file object
        Opened file object to read from.
    ver : int
        Version of the file format.

    Returns
    -------
    timestamp : int
        Timestamp of the data block in milliseconds, or -1 if cannot read.
    payload_size : int
        Size of the payload part of the data block in bytes, or -1 if cannot read.
    total_block_size : int
        Total size of the data block (header + payload) in bytes.
    """
    if ver == 0 or ver == 1:
        attributes = data_file.read(12)
        if len(attributes) == 12:
            timestamp, payload_size = struct.unpack("QI", attributes)
        else:
            timestamp = -1
            payload_size = -1

        total_block_size = payload_size + 12

    elif ver == 2 or ver == 3:
        attributes = data_file.read(20)
        if len(attributes) == 20:
            timestamp, block_id, payload_size = struct.unpack("QQI", attributes)
        else:
            timestamp = -1
            payload_size = -1

        total_block_size = payload_size + 20

    else:
        raise UnknownVersionError(ver)

    return timestamp, payload_size, total_block_size
