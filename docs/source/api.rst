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

Detailed API
------------

Main Functions
~~~~~~~~~~~~~~

.. autofunction:: browse_raw_with_video

.. autofunction:: browse_raw_with_audio

.. autofunction:: browse_raw_with_video_and_audio

Timestamp Alignment
~~~~~~~~~~~~~~~~~~~

.. autoclass:: TimestampAligner
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: compute_raw_timestamps

Media Classes
~~~~~~~~~~~~~

.. autoclass:: VideoFileCV2
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: VideoFileHelsinkiVideoMEG
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: AudioFileHelsinkiVideoMEG
   :members:
   :undoc-members:
   :show-inheritance:

Browser Components
------------------

.. currentmodule:: mne_videobrowser.browsers

.. autosummary::
   :toctree: generated/
   :nosignatures:

   AudioBrowser
   VideoBrowser
   SyncableBrowser
   RawBrowserManager

.. autoclass:: AudioBrowser
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: VideoBrowser
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: SyncableBrowser
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: RawBrowserManager
   :members:
   :undoc-members:
   :show-inheritance:

Media Module
------------

.. currentmodule:: mne_videobrowser.media

.. autosummary::
   :toctree: generated/
   :nosignatures:

   video
   audio
   helsinki_videomeg_file_utils

Video Module
~~~~~~~~~~~~

.. automodule:: mne_videobrowser.media.video
   :members:
   :undoc-members:
   :show-inheritance:

Audio Module
~~~~~~~~~~~~

.. automodule:: mne_videobrowser.media.audio
   :members:
   :undoc-members:
   :show-inheritance:

Helsinki VideoMEG Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: mne_videobrowser.media.helsinki_videomeg_file_utils
   :members:
   :undoc-members:
   :show-inheritance:
