from importlib.metadata import version

import mne_videobrowser


def test_version():
    """Test that __version__ is accessible and matches real package version."""
    assert hasattr(mne_videobrowser, "__version__")
    assert isinstance(mne_videobrowser.__version__, str)

    package_version = version("mne-videobrowser")
    assert mne_videobrowser.__version__ == package_version
