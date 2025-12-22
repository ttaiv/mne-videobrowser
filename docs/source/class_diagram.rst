Class Diagram
=============

Here you can find a class diagram representing the main classes and their relationships in the package.
The diagram is not perfectly up-to-date with the codebase (most notably audio related parts are missing),
but it still gives an overview of the architecture.
The main functions such as :func:`~mne_videobrowser.browse_raw_with_video` first create a :class:`~mne_videobrowser.browsers.SyncableBrowser` instance, and
pass it to the main controller ``SyncedRawMediaBrowser``, which handles setting up the GUI and synchronization
updates between the media browser(s) and MNE-Python's raw data browser. The audio browser is very similar to the video browser, it also inherits
from :class:`~mne_videobrowser.browsers.SyncableBrowserWidget` and fetches audio data using a separate interface (:class:`~mne_videobrowser.media.AudioFile`).

.. image:: _static/mne_videobrowser_class_diagram.png
   :alt: Class Diagram
