import pytest
import soundfile as sf
from pathlib import Path
from audio_drama.sfx.stable_audio import StableAudioEngine

def test_stable_audio_engine_procedural_generation(tmp_path):
    engine = StableAudioEngine(device="cpu", sample_rate=44100)
    out_file = tmp_path / "ambient_test.wav"
    
    res = engine.generate_ambient(
        prompt="hum of convenience store fluorescent lights",
        duration_seconds=1.5,
        output_path=out_file,
        volume=0.3
    )
    
    assert res.exists()
    info = sf.info(str(res))
    assert info.samplerate == 44100
    assert info.channels == 2
    assert 1.4 <= info.duration <= 1.6
