Synchronization Logic
=====================

This section explains how ``mne-videobrowser`` synchronizes different data streams (MEG/EEG, video, audio) that may have different sampling rates and time bases. The synchronization mechanism relies on mapping **sample indices** rather than the time values displayed in the individual browsers. This approach decouples the visualization (which often assumes evenly spaced samples) from the underlying temporal alignment (which accounts for jitter, clock drift, and gaps).

Key Components
--------------

1. **Individual Browsers (UI)**:
   Browsers like the MNE-Qt-Browser **raw data browser** (for MEG/EEG) and :class:`~mne_videobrowser.browsers.AudioBrowser` use an **evenly spaced time axis** (Nominal Time). This is derived from the sampling rate:

   .. math::

      \text{time} = \frac{\text{index}}{\text{sampling rate}}

   However, this may not perfectly reflect the physical recording time if there is clock drift or irregular sampling.

2. **TimestampAligner (Logic)**:
   The :class:`~mne_videobrowser.TimestampAligner` class handles the mapping between different streams using **True Time** (measured timestamps). It accounts for:
   
   * Jitter in sample arrival times.
   * Clock drift between devices.
   * Gaps or missing data.

   It maintains a bidirectional mapping between the timestamps of two streams.

3. **BrowserSynchronizer**:
   The ``BrowserSynchronizer`` coordinates the updates between browsers using ``TimestampAligner`` instances. When a user interacts with one browser, it calculates the corresponding index in the other browser based on true timestamps.

Synchronization Process
-----------------------

When a user interacts with a browser (e.g., the primary MEG/EEG data browser), the following steps occur:

1. **User Action**: The user selects a time point in the browser.
2. **Index Conversion**: The browser converts this visual position (Nominal Time) to a **sample index**.
3. **Alignment**: The ``BrowserSynchronizer`` passes this index to the ``TimestampAligner``.
4. **Timestamp Lookup**: The ``TimestampAligner`` looks up the **true timestamp** for that index in the source stream.
5. **Mapping**: It finds the index in the target stream (e.g., video) that has the closest timestamp to the source timestamp.
6. **Update**: The target browser is updated to show the frame or sample corresponding to that specific index.
   The target browser converts the index back to its own Nominal Time for display.

This architecture ensures that synchronization is **accurate with respect to the measured timestamps**, even if the individual browsers display a simplified, evenly spaced time axis. Note that because the browsers use Nominal Time for display, there may be slight discrepancies in the displayed times across different browsers, even though they are synchronized.
