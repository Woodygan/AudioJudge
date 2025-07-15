import pytest


def test_import():
    """Test that the package can be imported correctly."""
    from audiojudge import AudioJudge, AudioExample, AudioExamplePointwise, __version__

    # Check that the version is a string
    assert isinstance(__version__, str)

    # Check that the classes are available
    assert AudioJudge
    assert AudioExample
    assert AudioExamplePointwise
