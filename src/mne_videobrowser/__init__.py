"""MNE-Python extension for synchronized viewing of MEG/EEG, video, and audio."""

from importlib.metadata import PackageNotFoundError, version

from .media.audio import AudioFileHelsinkiVideoMEG
from .media.video import VideoFileCV2, VideoFileHelsinkiVideoMEG
from .raw_timestamp_computaton import compute_raw_timestamps
from .synced_raw_media_browser import (
    browse_raw_with_audio,
    browse_raw_with_video,
    browse_raw_with_video_and_audio,
)
from .timestamp_aligner import TimestampAligner

try:
    __version__ = version("mne-videobrowser")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "unknown"

__all__ = [
    "browse_raw_with_video",
    "browse_raw_with_audio",
    "browse_raw_with_video_and_audio",
    "TimestampAligner",
    "compute_raw_timestamps",
    "VideoFileHelsinkiVideoMEG",
    "VideoFileCV2",
    "AudioFileHelsinkiVideoMEG",
    "__version__",
]
