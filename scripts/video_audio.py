#!/usr/bin/env python3

import argparse
from contextlib import ExitStack
from pathlib import Path

from qtpy.QtWidgets import QApplication

from mne_videobrowser import (
    AudioFileHelsinkiVideoMEG,
    TimestampAligner,
    VideoFileHelsinkiVideoMEG,
)
from mne_videobrowser.browser_synchronizer import BrowserSynchronizer
from mne_videobrowser.browsers import AudioBrowser, VideoBrowser
from mne_videobrowser.media import VideoFile


def main() -> None:

    parser = argparse.ArgumentParser(description="Display multiple synchronized video files with one common audio file. ")
    parser.add_argument(
        "folder",
        type=str,
        metavar="folder",
        help="Folder containing one `.aud` file and one or more `.vid` files",
    )
    args = parser.parse_args()  

    folder = Path(args.folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Folder does not exist: {folder}")

    video_paths = list(sorted(folder.glob("*.vid")))
    if not video_paths:
        raise FileNotFoundError(f"No .vid files found in folder '{folder}'.")

    audio_paths = list(folder.glob("*.aud"))
    if len(audio_paths) != 1:
        raise FileNotFoundError(f"Expected exactly one .aud file in '{folder}', found {len(audio_paths)}.")
    audio_path = audio_paths[0]

    with ExitStack() as stack:
        videos: list[VideoFileHelsinkiVideoMEG] = [
            stack.enter_context(VideoFileHelsinkiVideoMEG(str(path)))
            for path in video_paths
        ]
        audio = stack.enter_context(
            AudioFileHelsinkiVideoMEG(str(audio_path))
        )

        audio_timestamps_ms = audio.get_audio_timestamps_ms()
        video_aligners = []
        for idx, video in enumerate(videos, start=1):
            video_aligners.append(
                TimestampAligner(
                    timestamps_a=audio_timestamps_ms,
                    timestamps_b=video.timestamps_ms,
                    timestamp_unit="milliseconds",
                    name_a="audio",
                    name_b=f"video-{idx}",
                )
            )

        app = QApplication([])

        video_browser = VideoBrowser(videos, show_sync_status=True)
        audio_browser = AudioBrowser(audio)
        video_browser.resize(1000, 800)
        audio_browser.resize(1000, 400)
        video_browser.show()
        audio_browser.show()

        synchronizer = BrowserSynchronizer(
            primary_browser=audio_browser,
            secondary_browsers=[video_browser],
            aligners=[video_aligners],
            max_sync_fps=10,
        )

        app.exec_()


if __name__ == "__main__":
    main()
