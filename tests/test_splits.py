import pytest

from anuraset_dl.splits import recording_id


def test_recording_id_removes_segment_interval() -> None:
    filename = "INCT20955_20190909_050000_0_3.wav"
    assert recording_id(filename) == "INCT20955_20190909_050000"


def test_recording_id_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        recording_id("audio.wav")
