API Reference
=============

This page contains the API reference for the mne-videobrowser package.

Main Functions
--------------

.. currentmodule:: mne_videobrowser

.. autosummary::
   :toctree: generated/
   :nosignatures:

   browse_raw_with_video
   browse_raw_with_audio
   browse_raw_with_video_and_audio

Timestamp Alignment
-------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   TimestampAligner
   compute_raw_timestamps

Media Classes
-------------

Video
~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   VideoFileCV2
   VideoFileHelsinkiVideoMEG

Audio
~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   AudioFileHelsinkiVideoMEG

Browser Components
------------------

These do not have to be interacted with if using any of the main functions above.
However, they are useful if:

1. You want to browse just video or audio without MEG/EEG data (requires manual management of Qt application loop, see examples).
2. You want to extend this package with new browser types.
3. You want to build something custom using the browsers.

.. currentmodule:: mne_videobrowser.browsers

.. autosummary::
   :toctree: generated/
   :nosignatures:

   AudioBrowser
   VideoBrowser
   SyncableBrowser
   SyncableBrowserObject
   SyncableBrowserWidget
   RawBrowserManager
