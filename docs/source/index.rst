mne-videobrowser
================

Video and audio browser extension for MNE-Python's Qt data browser.

**mne-videobrowser** is an open-source Python package for browsing video and audio 
time-synchronized to MEG/EEG data. It serves as an add-on for `mne-qt-browser 
<https://github.com/mne-tools/mne-qt-browser>`_, which is part of `MNE-Python 
<https://mne.tools/stable/>`_, an open-source Python package for exploring, 
visualizing, and analyzing human neurophysiological data.

This project also complements `Helsinki VideoMEG project 
<https://github.com/Helsinki-VideoMEG-Project>`_ by supporting video and audio 
files recorded with their software.

.. image:: https://raw.githubusercontent.com/ttaiv/mne-videobrowser/main/browser_screenshot.png
   :alt: VideoMEG browser screenshot

Features
--------

* Time-synchronized video playback with MEG/EEG data
* Time-synchronized audio playback with MEG/EEG data
* Support for multiple video and audio files simultaneously
* Integration with MNE-Python's data browser
* Support for Helsinki VideoMEG format files
* Standard video format support (MP4, AVI, etc.) via OpenCV

Installation
------------

In addition to MNE-Python, this project requires package ``OpenCV`` for standard 
video file (such as .mp4) reading and ``sounddevice`` for audio playback. For the 
qt backend to work correctly, MNE-Python should be installed using 
`conda <https://github.com/conda/conda>`_.

1. Create a new conda environment (named ``mne-videobrowser``) with MNE-Python installed:

   .. code-block:: bash

      conda create --channel=conda-forge --strict-channel-priority --name=mne-videobrowser mne

2. Activate the environment:

   .. code-block:: bash

      conda activate mne-videobrowser

3. Install this package with rest of the dependencies:

   .. code-block:: bash

      pip install mne-videobrowser

4. Only on linux: If you do not have `PortAudio library <https://www.portaudio.com/>`_, 
   which is dependency of ``sounddevice`` installed, install it. For example on Ubuntu/Debian:

   .. code-block:: bash

      sudo apt install libportaudio2

Quick Start
-----------

Here's a simple example of how to browse MEG data with synchronized video:

.. code-block:: python

   import mne
   from mne_videobrowser import browse_raw_with_video

   # Load your MEG/EEG data
   raw = mne.io.read_raw_fif('your_data.fif', preload=True)

   # Browse with video
   browse_raw_with_video(raw, 'your_video.mp4')

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api
   examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

