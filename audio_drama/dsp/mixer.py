import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import soundfile as sf
from pedalboard import Pedalboard, Limiter, HighpassFilter, Gain

def resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """
    Wielofazowy resampling sygnału audio z zachowaniem fazy i wysokości tonu.
    Eliminuje efekt chipmunka przy łączeniu sygnałów o różnych częstotliwościach próbkowania.
    """
    if src_sr == dst_sr or len(audio) == 0:
        return audio.astype(np.float32)
    from math import gcd
    import scipy.signal
    g = gcd(src_sr, dst_sr)
    return scipy.signal.resample_poly(
        audio,
        up=dst_sr // g,
        down=src_sr // g,
        axis=0
    ).astype(np.float32)

class SceneMixer:
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    @staticmethod
    def apply_sidechain_ducking(
        voice_signal: np.ndarray,
        bgm_signal: np.ndarray,
        sample_rate: int = 44100,
        threshold_db: float = -28.0,
        ducking_db: float = -14.0,
        attack_ms: float = 15.0,
        release_ms: float = 350.0
    ) -> np.ndarray:
        """
        Płynny kompresor sidechain oparty na detektorze obwiedni AR (Attack/Release).
        Gdy głos przekracza próg głośności, sygnał tła BGM jest płynnie tłumiony o ducking_db.
        """
        # Obliczenie mono obwiedni głosu
        if voice_signal.ndim > 1:
            voice_mono = np.mean(voice_signal, axis=0 if voice_signal.shape[0] < voice_signal.shape[1] else 1)
        else:
            voice_mono = voice_signal

        voice_rect = np.abs(voice_mono)
        num_samples = len(voice_rect)

        # Współczynniki czasowe filtru IIR
        alpha_attack = np.exp(-1.0 / (max(attack_ms, 1.0) * 0.001 * sample_rate))
        alpha_release = np.exp(-1.0 / (max(release_ms, 1.0) * 0.001 * sample_rate))

        # Detekcja obwiedni w pętli jednobiegunowej
        envelope = np.zeros(num_samples, dtype=np.float32)
        current_env = 0.0
        for i in range(num_samples):
            v = voice_rect[i]
            if v > current_env:
                current_env = (1.0 - alpha_attack) * v + alpha_attack * current_env
            else:
                current_env = (1.0 - alpha_release) * v + alpha_release * current_env
            envelope[i] = current_env

        # Wyliczenie redukcji wzmocnienia (Gain Reduction)
        threshold_linear = 10.0 ** (threshold_db / 20.0)
        max_attenuation = 10.0 ** (ducking_db / 20.0) # np. -14 dB -> ~0.20

        # Wektor mnożników głośności
        gain_curve = np.ones(num_samples, dtype=np.float32)
        mask = envelope > threshold_linear
        
        # Płynne skalowanie powyżej progu
        ratio_scale = np.clip((envelope[mask] - threshold_linear) / (threshold_linear * 3.0), 0.0, 1.0)
        gain_curve[mask] = 1.0 - ratio_scale * (1.0 - max_attenuation)

        # Zastosowanie krzywej do BGM
        if bgm_signal.ndim > 1:
            if bgm_signal.shape[0] == 2: # stereo (2, N)
                return bgm_signal * gain_curve[np.newaxis, :]
            else: # stereo (N, 2)
                return bgm_signal * gain_curve[:, np.newaxis]
        return bgm_signal * gain_curve

    def assemble_voice_track(
        self,
        cues_audio: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, float]:
        """
        Łączy kwestie głosowe sekwencyjnie z naturalnymi przerwami między wypowiedziami,
        automatycznie resamplując każdy fragment do self.sample_rate.
        """
        track_parts = []
        total_samples = 0

        for cue in cues_audio:
            audio = cue["audio"].astype(np.float32)
            src_sr = cue.get("sample_rate", self.sample_rate)
            if src_sr != self.sample_rate:
                audio = resample_audio(audio, src_sr=src_sr, dst_sr=self.sample_rate)

            pause_ms = cue.get("pause_after_ms", 300)
            pause_samples = int(self.sample_rate * (pause_ms / 1000.0))

            track_parts.append(audio)
            total_samples += len(audio)

            if pause_samples > 0:
                silence = np.zeros(pause_samples, dtype=np.float32)
                track_parts.append(silence)
                total_samples += pause_samples

        if not track_parts:
            return np.zeros(0, dtype=np.float32), 0.0

        full_track = np.concatenate(track_parts)
        duration_s = len(full_track) / self.sample_rate
        return full_track, duration_s

    def mix_and_master_scene(
        self,
        voice_track: np.ndarray,
        sfx_events: List[Dict[str, Any]],
        bgm_track: Optional[np.ndarray] = None,
        output_path: Optional[str | Path] = None
    ) -> np.ndarray:
        """
        Miksuje stemy sceny z sidechain duckingiem i limiterem szczytowym (Mastering).
        """
        scene_len = len(voice_track)
        if scene_len == 0:
            return np.zeros((2, 0), dtype=np.float32)

        # Dopasowanie BGM (zapętlenie lub przycięcie do długości sceny)
        if bgm_track is not None and len(bgm_track) > 0:
            if len(bgm_track) < scene_len:
                repeats = int(np.ceil(scene_len / len(bgm_track)))
                bgm_aligned = np.tile(bgm_track, repeats)[:scene_len]
            else:
                bgm_aligned = bgm_track[:scene_len]

            # Dynamiczny sidechain ducking
            bgm_processed = self.apply_sidechain_ducking(
                voice_signal=voice_track,
                bgm_signal=bgm_aligned,
                sample_rate=self.sample_rate
            )
        else:
            bgm_processed = np.zeros(scene_len, dtype=np.float32)

        # Przygotowanie ścieżki SFX
        sfx_track = np.zeros(scene_len, dtype=np.float32)
        for event in sfx_events:
            offset_sample = int(event.get("offset_s", 0.0) * self.sample_rate)
            sfx_audio = event.get("audio", np.zeros(0, dtype=np.float32))
            vol_db = event.get("volume_db", -12.0)
            sfx_gain = 10.0 ** (vol_db / 20.0)

            end_sample = min(offset_sample + len(sfx_audio), scene_len)
            insert_len = end_sample - offset_sample
            if insert_len > 0:
                sfx_track[offset_sample:end_sample] += sfx_audio[:insert_len] * sfx_gain

        # Konwersja do stereo (2, N)
        master_left = voice_track + sfx_track + bgm_processed
        master_right = voice_track + sfx_track + bgm_processed
        master_stereo = np.stack([master_left, master_right], axis=0)

        # Mastering chain: Highpass 30Hz + PeakLimiter -1.0 dB
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=30.0),
            Limiter(threshold_db=-1.0)
        ])
        master_final = board(master_stereo, self.sample_rate)

        if output_path:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            # soundfile oczekuje (N, 2)
            sf.write(str(out_file), master_final.T, self.sample_rate, subtype="PCM_16")

        return master_final
