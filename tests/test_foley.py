import pytest
import numpy as np

from audio_drama.sfx.library import FoleyLibrary
from audio_drama.sfx.foley_matcher import FoleyMatcher
from audio_drama.core.models import AudioCue, DeliveryHints

def test_foley_library_procedural_generation(tmp_path):
    lib = FoleyLibrary(sfx_dir=tmp_path)
    
    chime = lib.get_sound("konbini_chime")
    assert isinstance(chime, np.ndarray)
    assert len(chime) > 44100 # > 1 sekunda
    assert np.max(np.abs(chime)) > 0.05

    beep = lib.get_sound("scanner_beep")
    assert isinstance(beep, np.ndarray)
    assert 1000 < len(beep) < 44100 # krótki bip < 1s

    rustle = lib.get_sound("plastic_rustle")
    assert isinstance(rustle, np.ndarray)
    assert len(rustle) > 20000

    spray = lib.get_sound("alcohol_spray")
    assert isinstance(spray, np.ndarray)
    assert len(spray) > 10000

    ambience = lib.generate_store_ambience(duration_s=2.0)
    assert isinstance(ambience, np.ndarray)
    assert len(ambience) == int(2.0 * 44100)

def test_foley_matcher_scene_events(tmp_path):
    lib = FoleyLibrary(sfx_dir=tmp_path)
    matcher = FoleyMatcher(lib)

    cues = [
        AudioCue(
            cue_id="c01",
            cue_order=1,
            cue_type="narration",
            speaker_id="narrator",
            text="Sklep wypełniały dźwięki. Rozbrzmiewała melodyjka informująca, że ktoś wchodzi."
        ),
        AudioCue(
            cue_id="c02",
            cue_order=2,
            cue_type="dialogue",
            speaker_id="keiko",
            text="Dzień dobry! Zapakować onigiri w osobną folię?"
        ),
        AudioCue(
            cue_id="c03",
            cue_order=3,
            cue_type="narration",
            speaker_id="narrator",
            text="Zdezynfekowałam ręce sprejem z alkoholem, słysząc piknięć skanera przy kasie."
        )
    ]

    timing = [
        {"audio": np.zeros(44100, dtype=np.float32), "sample_rate": 44100, "pause_after_ms": 300},
        {"audio": np.zeros(44100, dtype=np.float32), "sample_rate": 44100, "pause_after_ms": 300},
        {"audio": np.zeros(44100, dtype=np.float32), "sample_rate": 44100, "pause_after_ms": 300},
    ]

    events = matcher.match_scene_sfx(cues, timing)
    event_names = [e["name"] for e in events]

    assert "konbini_chime" in event_names
    assert "plastic_rustle" in event_names
    assert "alcohol_spray" in event_names
    assert "scanner_beep" in event_names
