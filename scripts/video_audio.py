"""Synchronize a single video file with a single audio file (no raw data).

Running this requires a .vid and .aud recorded with Helsinki VideoMEG project
software. Update the file paths as needed.
"""

import logging
import argparse

from qtpy.QtWidgets import QApplication

from mne_videobrowser import (
    AudioFileHelsinkiVideoMEG,
    TimestampAligner,
    VideoFileHelsinkiVideoMEG,
)
from mne_videobrowser.browser_synchronizer import BrowserSynchronizer
from mne_videobrowser.browsers import AudioBrowser, VideoBrowser


def main() -> None:
    """Run the video/audio synchronization demo."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s:%(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('fname', type=str, metavar='file', help='Path to the `.vid` and `.aud` files, without extension')
    args = parser.parse_args()
    video_file = args.fname + '.vid'
    audio_file = args.fname + '.aud'

    with (
        VideoFileHelsinkiVideoMEG(video_file) as video,
        AudioFileHelsinkiVideoMEG(audio_file) as audio_file,
    ):
        video.print_stats()
        audio_file.print_stats()

        video_timestamps_ms = video.timestamps_ms
        audio_timestamps_ms = audio_file.get_audio_timestamps_ms()

        aligner = TimestampAligner(
            timestamps_a=video_timestamps_ms,
            timestamps_b=audio_timestamps_ms,
            timestamp_unit="milliseconds",
            name_a="video",
            name_b="audio",
        )

        app = QApplication([])

        video_browser = VideoBrowser([video], show_sync_status=True)
        audio_browser = AudioBrowser(audio_file)
        video_browser.resize(1000, 800)
        audio_browser.resize(1000, 400)
        video_browser.show()
        audio_browser.show()

        synchronizer = BrowserSynchronizer(
            primary_browser=video_browser,
            secondary_browsers=[audio_browser],
            aligners=[[aligner]],
            max_sync_fps=10,
        )

        app.exec_()


if __name__ == "__main__":
    main()
