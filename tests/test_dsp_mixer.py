import numpy as np
import pytest
from audio_drama.dsp.mixer import SceneMixer

def test_sidechain_ducking_attenuation():
    sr = 44100
    duration_s = 2.0
    num_samples = int(sr * duration_s)
    
    # 1 sekunda ciszy, 1 sekunda głośnego głosu (sinusoida 440Hz, 0 dBFS)
    voice = np.zeros(num_samples, dtype=np.float32)
    t = np.linspace(0, 1, sr, endpoint=False)
    voice[sr:] = 0.8 * np.sin(2 * np.pi * 440 * t)
    
    # Ciągłe tło muzyczne o stałej głośności
    bgm = 0.5 * np.ones(num_samples, dtype=np.float32)
    
    ducked_bgm = SceneMixer.apply_sidechain_ducking(
        voice_signal=voice,
        bgm_signal=bgm,
        sample_rate=sr,
        ducking_db=-14.0,
        attack_ms=15.0,
        release_ms=150.0
    )
    
    # W pierwszej sekundzie (cisza) BGM powinno być niemal nienaruszone (~0.5)
    assert np.mean(np.abs(ducked_bgm[:int(sr * 0.8)])) > 0.45
    
    # W drugiej sekundzie (głos) BGM powinno zostać stłumione o co najmniej 10 dB
    mean_loud_bgm = np.mean(np.abs(ducked_bgm[int(sr * 1.2):int(sr * 1.8)]))
    assert mean_loud_bgm < 0.20

def test_scene_mix_timeline():
    sr = 44100
    mixer = SceneMixer(sample_rate=sr)
    
    # 2 krótkie kwestie głosowe po 0.5s
    v1 = 0.5 * np.sin(2 * np.pi * 300 * np.linspace(0, 0.5, int(sr * 0.5)))
    v2 = 0.5 * np.sin(2 * np.pi * 500 * np.linspace(0, 0.5, int(sr * 0.5)))
    
    voice_cues = [
        {"audio": v1, "pause_after_ms": 200},
        {"audio": v2, "pause_after_ms": 200}
    ]
    
    mixed_voice, duration_s = mixer.assemble_voice_track(voice_cues)
    expected_duration = 0.5 + 0.2 + 0.5 + 0.2
    assert pytest.approx(duration_s, 0.05) == expected_duration
    assert len(mixed_voice) == round(duration_s * sr)

def test_resampling_prevents_chipmunk_effect():
    """
    Test regresyjny: upewnia się, że sygnał 22050 Hz (z Piper TTS)
    połączony w SceneMixerze (44100 Hz) zachowuje 100% czasu trwania
    i nie zostaje 2x skompresowany czasowo (efekt chomika).
    """
    from audio_drama.dsp.mixer import resample_audio
    mixer = SceneMixer(sample_rate=44100)
    
    # 1.0 sekunda głosu o częstotliwości 22050 Hz
    voice_22k = np.sin(2 * np.pi * 200 * np.linspace(0, 1.0, 22050, endpoint=False)).astype(np.float32)
    
    # Bezpośredni test funkcji resample_audio
    resampled_44k = resample_audio(voice_22k, src_sr=22050, dst_sr=44100)
    assert len(resampled_44k) == 44100
    
    # Test w assemble_voice_track z podaniem sample_rate
    cues = [
        {"audio": voice_22k, "sample_rate": 22050, "pause_after_ms": 0}
    ]
    track, duration_s = mixer.assemble_voice_track(cues)
    assert pytest.approx(duration_s, 0.01) == 1.0
    assert len(track) == 44100
