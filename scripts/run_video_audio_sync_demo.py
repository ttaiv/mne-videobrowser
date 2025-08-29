"""Demonstrates browsing raw data along with synchronized video and audio.

Running this requires video and audio files recorded with Helsinki videoMEG project
software. File paths also need adjustment.
"""

import logging
import os.path as op

import mne
import numpy as np
from mne.datasets import sample
from qtpy.QtWidgets import QApplication

from videomeg_browser.audio import AudioFileHelsinkiVideoMEG
from videomeg_browser.raw_media_aligner import RawMediaAligner
from videomeg_browser.synced_raw_media_browser import browse_raw_with_video_and_audio
from videomeg_browser.video import VideoFileHelsinkiVideoMEG

BASE_PATH = "/u/69/taivait1/unix/video_meg_testing/2025-07-11_MEG2MEG_test/"


def main() -> None:
    """Run the video and audio sync demo."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] [%(levelname)s] %(name)s:%(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load sample raw data.
    data_path = sample.data_path()
    raw_fname = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    raw.crop(tmax=60)  # Crop to 60 seconds

    # Load video and audio.
    video = VideoFileHelsinkiVideoMEG(
        op.join(BASE_PATH, "2025-07-11--18-18-41_video_01.vid")
    )
    audio = AudioFileHelsinkiVideoMEG(
        op.join(BASE_PATH, "2025-07-11--18-18-41_audio_00.aud")
    )
    audio.unpack_audio()

    # Print stats for video and audio.
    video.print_stats()
    audio.print_stats()

    # Extract video and audio timestamps.
    video_timestamps_ms = video.timestamps_ms
    audio_timestamps_ms = audio.get_audio_timestamps_ms()

    # Create artificial timestamps for raw data.
    start_ts = audio_timestamps_ms[0]  # Start at the first audio timestamp
    end_ts = start_ts + 60 * 1000  # End at 60 seconds later (convert to milliseconds)
    raw_timestamps_ms = np.linspace(start_ts, end_ts, raw.n_times, endpoint=False)

    # Create a separate aligner for video and audio.
    video_aligner = RawMediaAligner(
        raw_timestamps=raw_timestamps_ms,
        media_timestamps=video_timestamps_ms,
        timestamp_unit="milliseconds",
    )
    audio_aligner = RawMediaAligner(
        raw_timestamps=raw_timestamps_ms,
        media_timestamps=audio_timestamps_ms,
        timestamp_unit="milliseconds",
    )

    # Start the browser.
    app = QApplication([])
    raw_browser = raw.plot(block=False, show=False)
    browser = browse_raw_with_video_and_audio(
        raw_browser, raw, [video], [video_aligner], audio, audio_aligner
    )
    app.exec_()


if __name__ == "__main__":
    main()
