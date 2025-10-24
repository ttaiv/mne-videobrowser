"""Contains AudioFile interface and its implementations for reading audio files."""

# License: BSD-3-Clause
# Copyright (c) 2014 BioMag Laboratory, Helsinki University Central Hospital
# Copyright (c) 2025 Aalto University

import logging
import struct
from abc import ABC, abstractmethod
from fractions import Fraction

import numpy as np
import numpy.typing as npt
from scipy import signal

from .helsinki_videomeg_file_utils import UnknownVersionError, read_block_attributes

logger = logging.getLogger(__name__)


class AudioFile(ABC):
    """Handles reading audio files."""

    def __init__(self, fname: str) -> None:
        """Initialize the audio file reader with the given file name."""
        self._fname = fname

    @abstractmethod
    def get_audio_all_channels(
        self, sample_range: tuple[int, int] | None = None
    ) -> npt.NDArray[np.float32]:
        """Get audio data for all channels in the specified sample range.

        Parameters
        ----------
        sample_range : tuple[int, int] | None
            A tuple specifying the start and end (exclusive) sample indices to include
            in the output. If None (default), all the samples are included.

        Returns
        -------
        npt.NDArray[np.float32]
            A 2D array of shape (n_channels, n_samples) containing the audio data.
        """
        pass

    @abstractmethod
    def get_audio_mean(
        self, sample_range: tuple[int, int] | None = None
    ) -> npt.NDArray[np.float32]:
        """Get mean audio data across channels in the specified sample range.

        Parameters
        ----------
        sample_range : tuple[int, int] | None
            A tuple specifying the start and end (exclusive) sample indices to include
            in the output. If None (default), all the samples are included.

        Returns
        -------
        npt.NDArray[np.float32]
            A 1D array containing the mean audio data for the specified sample range.
        """
        pass

    def get_min_max_envelope(
        self,
        window_size: int,
        channel_idx: int | None,
        sample_range: tuple[int, int] | None = None,
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float32], npt.NDArray[np.float32]
    ]:
        """Calculate min-max envelope of the audio data using non-overlapping windows.

        Divides the audio signal into consecutive non-overlapping windows of fixed size
        and computes the minimum and maximum values in each window, capturing amplitude
        variations over time.

        Parameters
        ----------
        window_size : int
            The number of audio samples in each window.
        channel_idx : int | None
            The zero-based index of the channel to calculate the envelope for. If None,
            the envelope is calculated for the mean signal across all channels.
        sample_range : tuple[int, int] | None, optional
            A tuple specifying the start and end (exclusive) sample indices to include
            in the calculation. If None (default), all the samples are included.

        Returns
        -------
        times : npt.NDArray[np.float64]
            A 1D array of time points corresponding to the start time of each window.
        min_envelope : npt.NDArray[np.float32]
            A 1D array containing the minimum values of the audio signal in each window.
        max_envelope : npt.NDArray[np.float32]
            A 1D array containing the maximum values of the audio signal in each window.
        """
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")
        if channel_idx is not None and (
            channel_idx < 0 or channel_idx >= self.n_channels
        ):
            raise ValueError(
                f"Invalid channel index: {channel_idx}. "
                f"Must be in range [0, {self.n_channels - 1}]."
            )

        if channel_idx is None:
            audio_data = self.get_audio_mean(sample_range)
        else:
            audio_data = self.get_audio_all_channels(sample_range)[channel_idx, :]

        n_samples = len(audio_data)
        if n_samples < window_size:
            raise ValueError(
                f"Audio data length {len(audio_data)} is less than the window "
                f"size {window_size}."
            )

        # Pad the audio data with the last sample if necessary.
        remainder = n_samples % window_size
        if remainder != 0:
            pad_size = window_size - remainder
            audio_data = np.pad(audio_data, (0, pad_size), mode="edge")
        n_samples = len(audio_data)  # Update n_samples after padding
        assert n_samples % window_size == 0, "Remainder should be zero after padding."

        # Calculate the min-max envelope
        n_windows = n_samples // window_size
        audio_windows = audio_data.reshape(n_windows, window_size)
        min_envelope = np.min(audio_windows, axis=1)
        max_envelope = np.max(audio_windows, axis=1)

        # Calculate the time points for the start of each window
        start_sample = 0 if sample_range is None else sample_range[0]
        window_start_samples = np.arange(n_windows) * window_size + start_sample
        times = window_start_samples / self.sampling_rate  # Convert to seconds

        return times, min_envelope, max_envelope

    def resample_poly(
        self, target_rate: int, channel_idx: int | None
    ) -> npt.NDArray[np.float32]:
        """Resample the audio to the target sampling rate using polyphase filtering.

        Parameters
        ----------
        target_rate : int
            The desired sampling rate to resample the audio data to.
        channel_idx : int | None
            The zero-based index of the channel to resample. If None, the mean signal
            across all channels is resampled.

        Returns
        -------
        npt.NDArray[np.float32]
            A 1D array containing the resampled audio data.
        """
        if target_rate <= 0:
            raise ValueError("Target sampling rate must be a positive integer.")
        # Get the audio data to resample.
        if channel_idx is None:
            audio_data = self.get_audio_mean()
        else:
            audio_data = self.get_audio_all_channels()[channel_idx, :]

        if target_rate == self.sampling_rate:
            logger.info(
                "Target sampling rate is the same as the original. "
                "Returning original audio data without resampling."
            )
            return audio_data

        up, down = self._find_resample_factors(target_rate)
        if max(up, down) > 1000:
            logger.warning(
                f"Resampling factors are large {up}:{down}. This may lead to "
                "significant computational overhead. Consider using different "
                "resampling method or adjusting the target rate."
            )
        logger.info(
            f"Resampling audio from {self.sampling_rate} Hz to {target_rate} Hz "
            f"using polyphase filtering with factors {up}:{down}."
        )
        return signal.resample_poly(audio_data, up, down)

    @property
    def fname(self) -> str:
        """Return full path to the audio file that is being read."""
        return self._fname

    @property
    @abstractmethod
    def sampling_rate(self) -> int:
        """Return the nominal sampling rate of the audio."""
        pass

    @property
    @abstractmethod
    def n_channels(self) -> int:
        """Return the number of channels in the audio."""
        pass

    @property
    @abstractmethod
    def bit_depth(self) -> int:
        """Return the bit depth of the audio."""
        pass

    @property
    @abstractmethod
    def duration(self) -> float:
        """Return the duration of the audio in seconds."""
        pass

    @property
    @abstractmethod
    def n_samples(self) -> int:
        """Return the number of samples (per channel) in the audio."""
        pass

    def print_stats(self) -> None:
        """Print basic statistics about the audio file."""
        print(f"Stats for audio: {self.fname}")
        print(f"  - Number of channels: {self.n_channels}")
        print(f"  - Sampling rate: {self.sampling_rate} Hz")
        print(f"  - Bit depth: {self.bit_depth} bits")
        print(f"  - Duration: {self.duration:.2f} seconds")
        print(f"  - Number of samples per channel: {self.n_samples}")

    def _find_resample_factors(self, target_rate: int) -> tuple[int, int]:
        """Find the factors for up-and downsampling to match the target rate."""
        frac = Fraction(target_rate, self.sampling_rate)
        up, down = frac.numerator, frac.denominator
        return up, down


class AudioFileHelsinkiVideoMEG(AudioFile):
    """Read an audio file in the Helsinki videoMEG project format.

    In addition to the properties of AudioFile interface, the following
    attributes are available:
        buffer_timestamps_ms  - buffers' timestamps (unix time in milliseconds)
        raw_audio             - raw audio data
        format_string         - format string for the audio data
        buffer_size           - buffer size (bytes)

    To access the unpacked audio data ((n_channels, n_samples) numpy array) and its
    timestamps, call `unpack_audio()` method and then use the getter methods.
    Timestamps that can be used for synchronization are available via
    `get_audio_timestamps()` method.

    Parameters
    ----------
    fname : str
        Full path to the audio file.
    magic_str : str, optional
        Magic string that should be at the beginning of video file.
        Default is "HELSINKI_VIDEO_MEG_PROJECT_AUDIO_FILE".
    regression_segment_length : int, optional
        Length of segments (in seconds) used in piecewise linear regression
        to compute timestamps for all audio samples. Default is 20 seconds.
    """

    def __init__(
        self,
        fname: str,
        magic_str: str = "HELSINKI_VIDEO_MEG_PROJECT_AUDIO_FILE",
        regression_segment_length: int = 20,
    ) -> None:
        super().__init__(fname)
        self._regression_segment_length = regression_segment_length
        # Open the file to parse metadata and read the audio bytes into memory.
        self._data_file = open(self._fname, "rb")
        # Check the magic string
        if not self._data_file.read(len(magic_str)) == magic_str.encode("utf8"):
            raise ValueError(
                f"File {fname} does not start with the expected "
                f"magic string: {magic_str}."
            )
        self.ver = struct.unpack("I", self._data_file.read(4))[0]
        if self.ver != 0:
            # Can only read version 0 for the time being
            raise UnknownVersionError()

        self._sampling_rate, self._n_channels = struct.unpack(
            "II", self._data_file.read(8)
        )
        self.format_string = self._data_file.read(2).decode("ascii")

        begin_data = self._data_file.tell()
        self._data_file.seek(0, 2)  # seek to end of file
        end_data = self._data_file.tell()
        # Seek back to the beginning of audio data blocks.
        self._data_file.seek(begin_data, 0)

        # Get the size of the payload in one audio data block and the total size
        # of the block (header + payload). Advances file position!
        _, first_payload_size, first_block_size = read_block_attributes(
            self._data_file, self.ver
        )
        self.buffer_size_bytes = first_payload_size  # size of audio data in one block
        self._data_file.seek(begin_data, 0)  # return to beginning

        assert (end_data - begin_data) % first_block_size == 0

        self._n_blocks = (end_data - begin_data) // first_block_size
        self.raw_audio = bytearray(self._n_blocks * self.buffer_size_bytes)
        self.buffer_timestamps_ms = np.zeros(self._n_blocks, dtype=np.int64)
        self._audio_block_positions: list[int] = []

        for i in range(self._n_blocks):
            timestamp, payload_size, block_size = read_block_attributes(
                self._data_file, self.ver
            )
            assert block_size == first_block_size, (
                "Inconsistent block size while reading audio data."
                f" Expected {first_block_size}, got {block_size}."
            )
            self._audio_block_positions.append(self._data_file.tell())
            self._data_file.seek(payload_size, 1)  # skip the audio data
            self.buffer_timestamps_ms[i] = timestamp
        # close the file

        # Make sure that the timestamps are increasing
        if not np.all(np.diff(self.buffer_timestamps_ms) >= 0):
            raise ValueError(
                "Audio buffer timestamps must be non-decreasing but found "
                "decreasing values."
            )

        # Calculate stats for a single sample.
        self._bit_depth = self._get_bit_depth(self.format_string)
        self._n_bytes_per_sample = struct.calcsize(self.format_string)

        # Calculate how many samples there is in one raw audio data buffer,
        # taking into account that the buffer contains interleaved samples
        # from all channels.
        one_sample_from_all_channels_size = self._n_channels * self._n_bytes_per_sample

        assert self.buffer_size_bytes % one_sample_from_all_channels_size == 0, (
            "Audio buffer size is not a multiple of one sample from all channels."
        )
        self._n_samples_per_channel_per_buffer = (
            self.buffer_size_bytes // one_sample_from_all_channels_size
        )
        # Calculate total number of samples per channel in the whole audio.
        self._n_samples = self._n_samples_per_channel_per_buffer * self._n_blocks

        self._compute_audio_timestamps()  # will set self._audio_timestamps_ms

    def __del__(self) -> None:
        """Destructor to ensure the audio file is closed."""
        self.close()

    def close(self) -> None:
        """Close the audio file."""
        self._data_file.close()

    def get_audio_all_channels(
        self, sample_range: tuple[int, int] | None = None
    ) -> npt.NDArray[np.float32]:
        """Get audio data for all channels in the specified sample range.

        Parameters
        ----------
        sample_range : tuple[int, int] | None
            A tuple specifying the start and end (exclusive) sample indices to include
            in the output. If None (default), all the samples are included.

        Returns
        -------
        npt.NDArray[np.float32]
            A 2D array of shape (n_channels, n_samples) containing the audio data.
        """
        """
        self._ensure_unpacked_audio()
        assert self._unpacked_audio is not None, (
            "Audio data should be unpacked after calling _ensure_unpacked_audio."
        )
        """
        return self._get_audio_samples(
            sample_range if sample_range is not None else (0, self.n_samples)
        )

    def get_audio_mean(
        self, sample_range: tuple[int, int] | None = None
    ) -> npt.NDArray[np.float32]:
        """Get mean audio data across channels in the specified sample range.

        Triggers unpacking of audio data if it has not been done yet.

        Parameters
        ----------
        sample_range : tuple[int, int] | None
            A tuple specifying the start and end (exclusive) sample indices to include
            in the output. If None (default), all the samples are included.

        Returns
        -------
        npt.NDArray[np.float32]
            A 1D array containing the mean audio data for the specified sample range.
        """
        audio_all_channels = self._get_audio_samples(
            sample_range if sample_range is not None else (0, self.n_samples)
        )
        return audio_all_channels.mean(axis=0)

    def get_audio_timestamps_ms(self) -> npt.NDArray[np.float64]:
        """Get timestamps for all audio samples in milliseconds.

        Triggers unpacking of audio data if it has not been done yet.

        Returns
        -------
        npt.NDArray[np.float64]
            A 1D array containing timestamps for all audio samples in milliseconds.
        """
        return self._audio_timestamps_ms

    @property
    def sampling_rate(self) -> int:
        return self._sampling_rate

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def bit_depth(self) -> int:
        return self._bit_depth

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def duration(self) -> float:
        return self.n_samples / self.sampling_rate

    def _get_audio_samples(
        self, sample_range: tuple[int, int]
    ) -> npt.NDArray[np.float32]:
        """Get audio samples in the specified range (start inclusive, end exclusive).

        Determines the correct audio blocks to read from file and unpacks the samples.
        """
        start_sample, end_sample = sample_range
        if start_sample < 0 or end_sample > self.n_samples:
            raise ValueError("Sample range is out of bounds.")
        if start_sample >= end_sample:
            raise ValueError("Invalid sample range: start must be less than end.")

        n_samples_to_read = end_sample - start_sample

        # Determine which blocks to read.
        first_block_idx = start_sample // self._n_samples_per_channel_per_buffer
        last_block_idx = (end_sample - 1) // self._n_samples_per_channel_per_buffer
        n_blocks_to_read = last_block_idx - first_block_idx + 1

        # Allocate space for raw audio data from the blocks.
        block_data = bytearray(n_blocks_to_read * self.buffer_size_bytes)

        # Read the necessary blocks and concatenate their payloads (ignore headers).
        for block_idx in range(first_block_idx, last_block_idx + 1):
            # Determine where to copy the block data in the allocated bytearray.
            relative_block_idx = block_idx - first_block_idx
            block_start = relative_block_idx * self.buffer_size_bytes
            block_end = block_start + self.buffer_size_bytes

            # Read the block and copy its data.
            block_data[block_start:block_end] = self._read_block(block_idx)

        # Unpack the audio data from the read blocks to (n_channels, n_samples) array.
        unpacked_audio = self._unpack_audio(block_data, n_blocks_to_read)

        # Because we might have read more samples than requested (we read whole blocks),
        # determine the correct slice to return.
        first_block_start_sample = (
            first_block_idx * self._n_samples_per_channel_per_buffer
        )
        copy_start = start_sample - first_block_start_sample
        copy_end = copy_start + n_samples_to_read

        return unpacked_audio[:, copy_start:copy_end]

    def _read_block(self, block_idx: int) -> bytes:
        """Read the raw audio data from the specified block in the file."""
        # Seek to the beginning of the block.
        block_pos = self._audio_block_positions[block_idx]
        self._data_file.seek(block_pos, 0)

        # Read the block data.
        return self._data_file.read(self.buffer_size_bytes)

    def _unpack_audio(
        self, audio_bytes: bytearray, n_blocks: int
    ) -> npt.NDArray[np.float32]:
        """Unpack given raw audio bytes from adjacent blocks.

        Parameters
        ----------
        audio_bytes : bytes
            Raw audio bytes from adjacent blocks to unpack.
        n_blocks : int
            Number of blocks contained in audio_bytes.

        Returns
        -------
        npt.NDArray[np.float32]
            A 2D array of shape (n_channels, n_samples) containing the unpacked audio
            data.
        """
        n_samples_per_channel = self._n_samples_per_channel_per_buffer * n_blocks
        total_samples = n_samples_per_channel * self.n_channels

        # Create a format string for unpacking all samples at once.
        endian_char = self.format_string[0]
        sample_type = self.format_string[1]
        bulk_format_string = f"{endian_char}{total_samples}{sample_type}"

        unpacked_samples = struct.unpack(bulk_format_string, audio_bytes)
        # Convert the tuple to numpy array.
        audio = np.array(unpacked_samples, dtype=np.float32)

        # Reshape (n_channels, n_samples) layout.
        # The data is interleaved, so reshape to (n_samples, n_channels) first
        # and then transpose.
        return audio.reshape(n_samples_per_channel, self.n_channels).T

    def _get_bit_depth(self, format_string: str) -> int:
        """Get the bit depth from the format string."""
        # Dictionary mapping format characters to bit depths
        bit_depth_map = {
            "b": 8,  # signed char
            "B": 8,  # unsigned char
            "h": 16,  # short
            "H": 16,  # unsigned short
            "i": 32,  # int
            "I": 32,  # unsigned int
            "l": 32,  # long
            "L": 32,  # unsigned long
            "q": 64,  # long long
            "Q": 64,  # unsigned long long
            "f": 32,  # float
            "d": 64,  # double
        }
        # Extract the format character, ignoring endianness indicators
        bit_depth_char = format_string[-1]

        if bit_depth_char not in bit_depth_map:
            raise ValueError(
                f"Unsupported bit depth character: {bit_depth_char} in format "
                f"string {format_string}"
            )
        return bit_depth_map[bit_depth_char]

    def _compute_audio_timestamps(self) -> None:
        """Transform sparse buffer timestamps into dense sample timestamps.

        Uses piecewise linear regression to estimate timestamps for all samples
        based on the buffer timestamps.
        """
        # Create an array that contains the indices of the last sample in each buffer.
        # These indices correspond to the timestamps we have.
        buffer_end_indices = np.arange(
            self._n_samples_per_channel_per_buffer - 1,
            self.n_samples,
            self._n_samples_per_channel_per_buffer,
        )

        # Prepare arrays to hold the regression errors and the computed timestamps.
        regression_errors = -np.ones(self._n_blocks, dtype=np.float64)
        # Double precision is important here!
        audio_timestamps_ms = -np.ones(self.n_samples, dtype=np.float64)

        # Split the data into segments for piecewise linear regression.
        split_indices = list(
            range(
                0, self.n_samples, self._regression_segment_length * self._sampling_rate
            )
        )
        # the last segment might be up to twice as long as the others
        split_indices[-1] = self.n_samples

        # Loop over the segments and perform linear regression.
        for i in range(len(split_indices) - 1):
            segment_start_idx = split_indices[i]
            segment_end_idx = split_indices[i + 1]

            # Find the buffers that have timestamps within the current segment.
            segment_mask = (buffer_end_indices >= segment_start_idx) & (
                buffer_end_indices < segment_end_idx
            )
            # Take the samples indices and timestamps.
            timestamp_indices = buffer_end_indices[segment_mask]
            timestamps_ms = self.buffer_timestamps_ms[segment_mask]

            # Fit a linear regression.
            p = np.polyfit(
                timestamp_indices,
                timestamps_ms,
                1,
            )
            # Compute the regression error for the known timestamps.
            regression_errors[segment_mask] = np.abs(
                np.polyval(p, timestamp_indices)
                - self.buffer_timestamps_ms[segment_mask]
            )
            # Compute timestamps for all samples in the segment.
            audio_timestamps_ms[segment_start_idx:segment_end_idx] = np.polyval(
                p, np.arange(segment_start_idx, segment_end_idx)
            )

        assert audio_timestamps_ms.min() >= 0, "All timestamps should be set"
        assert regression_errors.min() >= 0, "All regression errors should be set"

        logger.info(
            "Audio regression fit errors (abs): mean %.3f ms, median %.3f ms, "
            "max %.3f ms",
            regression_errors.mean(),
            np.median(regression_errors),
            regression_errors.max(),
        )

        # Make sure that the timestamps are non-decreasing.
        timestamps_diff = np.diff(audio_timestamps_ms)
        if not np.all(timestamps_diff >= 0):
            logger.warning(
                "Piecewise linear regression produced %d decreasing timestamps. "
                "Replacing the decreasing timestamps with the previous valid timestamp "
                "to ensure non-decreasing timestamps.",
                np.sum(timestamps_diff < 0),
            )
            audio_timestamps_ms = np.maximum.accumulate(audio_timestamps_ms)

        self._audio_timestamps_ms = audio_timestamps_ms
