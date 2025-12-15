Examples
========

This page provides examples of how to use ``mne-videobrowser`` package.
All examples require you to have your own MEG/EEG data and video/audio files and modify
the file paths in the scripts accordingly. See
`examples/ <https://github.com/ttaiv/mne-videobrowser/tree/main/examples>`_
directory in the GitHub repository for these examples as Python files and
also two examples that can be run without your own data (they are not included here
as they have boilerplate code for creating fake data).

Standalone video and audio browsing
--------------------------------------

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

Video with MEG data
~~~~~~~~~~~~~~~~~~~

Browse MEG data with synchronized video:

.. literalinclude:: ../../examples/video_meg_sync.py
   :language: python

Audio with MEG data
~~~~~~~~~~~~~~~~~~~

Browse MEG data with synchronized audio:

.. literalinclude:: ../../examples/audio_meg_sync.py
   :language: python


Video and Audio with MEG Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Browse MEG data with both video and audio synchronized:

.. literalinclude:: ../../examples/video_audio_meg_sync.py
   :language: python


Two Videos with MEG Data
~~~~~~~~~~~~~~~~~~~~~~~~

Browse MEG data with two synchronized videos:

.. literalinclude:: ../../examples/two_video_meg_sync.py
   :language: python

Two Videos and Audio with MEG Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Browse MEG data with two videos and audio all synchronized:

.. literalinclude:: ../../examples/two_video_audio_meg_sync.py
   :language: python

