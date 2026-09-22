import pytest
from pathlib import Path
from audio_drama.tts.piper_engine import PiperEngine, PIPER_VOICE_CATALOG
from audio_drama.tts.benchmark import TTSBenchmark

def test_piper_voice_catalog():
    assert "pl_PL-gosia-medium" in PIPER_VOICE_CATALOG
    assert "pl_PL-darkman-medium" in PIPER_VOICE_CATALOG
    assert PIPER_VOICE_CATALOG["pl_PL-gosia-medium"]["gender"] == "female"
    assert PIPER_VOICE_CATALOG["pl_PL-darkman-medium"]["gender"] == "male"

def test_piper_engine_initialization(tmp_path):
    engine = PiperEngine(voices_dir=tmp_path)
    assert engine.default_voice == "pl_PL-darkman-medium"
    assert engine.get_available_voices() == list(PIPER_VOICE_CATALOG.keys())

def test_tts_benchmark_init(tmp_path):
    bench = TTSBenchmark(output_dir=tmp_path)
    assert bench.output_dir.exists()
