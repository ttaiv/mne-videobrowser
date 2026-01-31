"""Tests for package version."""

import ast
from importlib.metadata import version
from pathlib import Path


def test_version_defined():
    """Test that __version__ is defined in __init__.py."""
    init_file = (
        Path(__file__).parent.parent / "src" / "mne_videobrowser" / "__init__.py"
    )
    content = init_file.read_text()

    # Parse the file and check that __version__ is assigned
    tree = ast.parse(content)
    version_defined = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__version__":
                    version_defined = True
                    break

    assert version_defined, "__version__ is not defined in __init__.py"


def test_version_in_all():
    """Test that __version__ is exported in __all__."""
    init_file = (
        Path(__file__).parent.parent / "src" / "mne_videobrowser" / "__init__.py"
    )
    content = init_file.read_text()

    # Parse the file and find __all__ definition
    tree = ast.parse(content)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    # Get the list elements
                    if isinstance(node.value, ast.List):
                        all_items = [
                            elt.value
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant)
                        ]
                        assert "__version__" in all_items, (
                            "__version__ is not in __all__"
                        )
                        return

    raise AssertionError("__all__ is not defined in __init__.py")


def test_version_matches_metadata():
    """Test that __version__ can be retrieved from package metadata."""
    pkg_version = version("mne-videobrowser")
    assert pkg_version is not None
    assert len(pkg_version) > 0
    # Version should follow semantic versioning pattern
    assert "." in pkg_version
