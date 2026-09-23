import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from audio_drama.tts.xtts_engine import XTTSEngine, XTTS_PRESET_MAP

def test_xtts_available_voices():
    with patch("TTS.api.TTS"):
        engine = XTTSEngine(gpu=False)
        voices = engine.get_available_voices()
        assert "Ana Florence" in voices
        assert "Claribel Dervla" in voices
        assert "Damian Black" in voices

def test_xtts_speaker_resolution():
    with patch("TTS.api.TTS"):
        engine = XTTSEngine(gpu=False)
        # Test preset name
        speaker, wav = engine._resolve_speaker_and_wav("Damian Black")
        assert speaker == "Damian Black"
        assert wav is None

        # Test default fallback
        speaker_def, wav_def = engine._resolve_speaker_and_wav("NonExistentVoice")
        assert speaker_def == "Ana Florence"
        assert wav_def is None
