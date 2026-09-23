import pytest
from pathlib import Path
from audio_drama.tts.edge_engine import EdgeTTSEngine, EDGE_VOICE_CATALOG

def test_edge_tts_catalog():
    engine = EdgeTTSEngine()
    voices = engine.get_available_voices()
    assert "pl-PL-ZofiaNeural" in voices
    assert "pl-PL-MarekNeural" in voices
    assert EDGE_VOICE_CATALOG["pl-PL-ZofiaNeural"]["gender"] == "female"
    assert EDGE_VOICE_CATALOG["pl-PL-MarekNeural"]["gender"] == "male"

def test_edge_tts_formatters():
    assert EdgeTTSEngine._format_rate(1.0) == "+0%"
    assert EdgeTTSEngine._format_rate(1.1) == "+10%"
    assert EdgeTTSEngine._format_rate(0.9) == "-10%"

    assert EdgeTTSEngine._format_pitch(0.0) == "+0Hz"
    assert EdgeTTSEngine._format_pitch(10.0) == "+10Hz"
    assert EdgeTTSEngine._format_pitch(-15.0) == "-15Hz"

    assert EdgeTTSEngine._format_volume(1.0) == "+0%"
    assert EdgeTTSEngine._format_volume(0.8) == "-20%"
