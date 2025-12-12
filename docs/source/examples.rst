Examples
========

This page provides examples of how to use mne-videobrowser for various use cases.

The examples directory contains several Python scripts demonstrating different features 
of the package. You can find them in the `examples/ 
<https://github.com/ttaiv/mne-videobrowser/tree/main/examples>`_ directory of the 
GitHub repository.

Basic Examples
--------------

Video Browser
~~~~~~~~~~~~~

Simple video browser without MEG/EEG data synchronization:

.. literalinclude:: ../../examples/video_browser.py
   :language: python

Audio Browser
~~~~~~~~~~~~~

Simple audio browser without MEG/EEG data synchronization:

.. literalinclude:: ../../examples/audio_browser.py
   :language: python

Synchronized Examples
---------------------

Video with MEG Data
~~~~~~~~~~~~~~~~~~~

Browse MEG data with synchronized video. This example uses MNE-Python's sample 
dataset and creates a fake video file:

.. literalinclude:: ../../examples/video_sample_meg_sync.py
   :language: python

Video and Audio with MEG Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Browse MEG data with both video and audio synchronized:

.. literalinclude:: ../../examples/video_audio_meg_sync.py
   :language: python

Multiple Videos with MEG Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Browse MEG data with multiple synchronized videos:

.. literalinclude:: ../../examples/multiple_meg_video_sync.py
   :language: python

Two Videos with MEG Data
~~~~~~~~~~~~~~~~~~~~~~~~

Browse MEG data with two synchronized videos:

.. literalinclude:: ../../examples/two_video_meg_sync.py
   :language: python

Two Videos and Audio with MEG Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Browse MEG data with two videos and audio all synchronized:

.. literalinclude:: ../../examples/two_video_audio_meg_sync.py
   :language: python

Advanced Examples
-----------------

Audio and MEG Synchronization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detailed example of audio-MEG synchronization:

.. literalinclude:: ../../examples/audio_meg_sync.py
   :language: python

Video and MEG Synchronization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detailed example of video-MEG synchronization:

.. literalinclude:: ../../examples/video_meg_sync.py
   :language: python

Running the Examples
--------------------

Most examples require you to have your own MEG/EEG data and video/audio files. 
However, ``video_sample_meg_sync.py`` and ``multiple_meg_video_sync.py`` use MNE-Python's 
sample dataset and create fake video files, so they can be run immediately after 
installation.

To run an example:

.. code-block:: bash

   python examples/video_sample_meg_sync.py

Make sure you have activated your conda environment with mne-videobrowser installed 
before running the examples.

Note
----

Examples that require custom data files will need you to modify the file paths in 
the scripts to point to your own MEG/EEG, video, and audio files.
