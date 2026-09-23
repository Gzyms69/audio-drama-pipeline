import math
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import soundfile as sf
import scipy.signal

class FoleyLibrary:
    """
    Proceduralny bank autentycznych efektów dźwiękowych (Foley) i tła konbini
    generowanych cyfrowo w wysokiej rozdzielczości (44.1 kHz, 32-bit float).
    """

    def __init__(self, sfx_dir: str | Path = "data/sfx", sample_rate: int = 44100):
        self.sfx_dir = Path(sfx_dir)
        self.sfx_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self._cache: Dict[str, np.ndarray] = {}

    def get_sound(self, sound_name: str) -> np.ndarray:
        """Zwraca sygnał audio (mono float32) dla danego efektu dźwiękowego."""
        if sound_name in self._cache:
            return self._cache[sound_name]

        file_path = self.sfx_dir / f"{sound_name}.wav"
        if file_path.exists():
            data, sr = sf.read(str(file_path))
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if sr != self.sample_rate:
                from audio_drama.dsp.mixer import resample_audio
                data = resample_audio(data, sr, self.sample_rate)
            self._cache[sound_name] = data.astype(np.float32)
            return self._cache[sound_name]

        # Generacja proceduralna
        generator = getattr(self, f"generate_{sound_name}", None)
        if generator is not None:
            audio = generator()
            sf.write(str(file_path), audio, self.sample_rate)
            self._cache[sound_name] = audio
            return audio

        # Domyślny krótki klik jeśli dźwięk nieznany
        return np.zeros(int(0.1 * self.sample_rate), dtype=np.float32)

    def generate_konbini_chime(self) -> np.ndarray:
        """
        Kultowa melodyjka powitalna japońskiego sklepu convenience store (Panasonic EBG888).
        Sekwencja 11 tonów dzwonków rurowych:
        F#4 -> D4 -> A3 -> D4 -> E4 -> A4 -> E4 -> F#4 -> E4 -> A3 -> D4
        """
        sr = self.sample_rate
        # Częstotliwości nut (Hz)
        notes = [
            (370.0, 0.17),  # F#4
            (293.7, 0.17),  # D4
            (220.0, 0.17),  # A3
            (293.7, 0.17),  # D4
            (329.6, 0.17),  # E4
            (440.0, 0.34),  # A4 (dłuższa)
            (329.6, 0.17),  # E4
            (370.0, 0.17),  # F#4
            (329.6, 0.17),  # E4
            (220.0, 0.17),  # A3
            (293.7, 0.60),  # D4 (wybrzmienie końcowe)
        ]

        total_duration = sum(dur for _, dur in notes) + 1.2
        total_samples = int(total_duration * sr)
        audio = np.zeros(total_samples, dtype=np.float32)

        current_time = 0.0
        for freq, dur in notes:
            start_idx = int(current_time * sr)
            # Długość wybrzmienia każdego dzwonka (chime decay)
            decay_dur = 1.0
            n_samples = int(decay_dur * sr)
            t = np.linspace(0, decay_dur, n_samples, endpoint=False)

            # Obwiednia tłumiona wykładniczo
            envelope = np.exp(-t / 0.28) * (1.0 - np.exp(-t / 0.005))

            # Synteza addytywna metalicznego dzwonka (ton podstawowy + alikwoty dzwonu)
            tone = (
                1.00 * np.sin(2 * np.pi * freq * t) +
                0.40 * np.sin(2 * np.pi * freq * 2.0 * t) +
                0.20 * np.sin(2 * np.pi * freq * 3.0 * t) +
                0.15 * np.sin(2 * np.pi * freq * 4.2 * t)
            )
            chime = (tone * envelope * 0.25).astype(np.float32)

            end_idx = min(start_idx + n_samples, total_samples)
            actual_len = end_idx - start_idx
            audio[start_idx:end_idx] += chime[:actual_len]

            current_time += dur

        # Normalizacja do -3 dB
        peak = np.max(np.abs(audio)) + 1e-6
        if peak > 0:
            audio = (audio / peak) * 0.7
        return audio

    def generate_scanner_beep(self) -> np.ndarray:
        """
        Charakterystyczne podwójne piknięcie laserowego czytnika kodów kreskowych przy kasie.
        Częstotliwości 1850 Hz i 2400 Hz, czas trwania 70 ms.
        """
        sr = self.sample_rate
        duration = 0.075
        n_samples = int(duration * sr)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        # Szybka obwiednia trapezowa z zaokrąglonym atakiem i zanikiem
        attack_s = 0.005
        release_s = 0.020
        att_samples = int(attack_s * sr)
        rel_samples = int(release_s * sr)
        sustain_samples = n_samples - att_samples - rel_samples

        env = np.concatenate([
            np.linspace(0.0, 1.0, att_samples),
            np.ones(max(sustain_samples, 0)),
            np.linspace(1.0, 0.0, rel_samples)
        ])[:n_samples]

        tone = 0.7 * np.sin(2 * np.pi * 1850 * t) + 0.3 * np.sin(2 * np.pi * 2400 * t)
        beep = (tone * env * 0.45).astype(np.float32)
        return beep

    def generate_plastic_rustle(self) -> np.ndarray:
        """
        Szelest folii celofanowej onigiri i torebki foliowej konbini.
        Filtrowany szum pasmowoprzepustowy (3 kHz - 10 kHz) z serią losowych mikro-impulsów.
        """
        sr = self.sample_rate
        duration = 1.1
        n_samples = int(duration * sr)
        audio = np.zeros(n_samples, dtype=np.float32)

        # Generowanie losowych zgnieceń celofanu
        np.random.seed(42)
        num_bursts = 18
        burst_times = np.sort(np.random.uniform(0.05, duration - 0.15, num_bursts))

        for bt in burst_times:
            b_dur = np.random.uniform(0.02, 0.07)
            b_len = int(b_dur * sr)
            noise = np.random.normal(0, 1, b_len).astype(np.float32)
            t = np.linspace(0, b_dur, b_len, endpoint=False)
            env = np.exp(-t / (b_dur * 0.35)) * np.sin(np.pi * np.clip(t / b_dur, 0, 1))

            burst = noise * env * np.random.uniform(0.3, 0.7)
            idx = int(bt * sr)
            end_idx = min(idx + b_len, n_samples)
            audio[idx:end_idx] += burst[:end_idx - idx]

        # Filtracja pasmowoprzepustowa (charakterystyka cienkiej, szeleszczącej folii)
        sos = scipy.signal.butter(4, [2500, 9500], btype="bandpass", fs=sr, output="sos")
        filtered = scipy.signal.sosfilt(sos, audio).astype(np.float32)

        peak = np.max(np.abs(filtered)) + 1e-6
        return (filtered / peak) * 0.35

    def generate_alcohol_spray(self) -> np.ndarray:
        """
        Podwójne psiknięcie atomizera do dezynfekcji rąk na wejściu/przy ladzie.
        """
        sr = self.sample_rate
        duration = 0.5
        n_samples = int(duration * sr)
        audio = np.zeros(n_samples, dtype=np.float32)

        # Dwa impulsy aerozolu
        for spray_start in [0.03, 0.22]:
            s_dur = 0.12
            s_len = int(s_dur * sr)
            t = np.linspace(0, s_dur, s_len, endpoint=False)
            env = (t / 0.01) * np.exp(-t / 0.025)
            env = env / (np.max(env) + 1e-6)

            noise = np.random.normal(0, 1, s_len).astype(np.float32)
            # Syczący aerozol (filtr górnoprzepustowy 2.5 kHz)
            sos = scipy.signal.butter(3, 2500, btype="highpass", fs=sr, output="sos")
            spray = scipy.signal.sosfilt(sos, noise * env)

            idx = int(spray_start * sr)
            end_idx = min(idx + s_len, n_samples)
            audio[idx:end_idx] += spray[:end_idx - idx] * 0.4

        return audio.astype(np.float32)

    def generate_cash_register(self) -> np.ndarray:
        """
        Dźwięk otwarcia kasy fiskalnej i brzęku monet.
        """
        sr = self.sample_rate
        duration = 0.8
        n_samples = int(duration * sr)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        # Klik zamka elektromagnetycznego (500 Hz + impuls)
        click_dur = 0.03
        click_samples = int(click_dur * sr)
        t_click = t[:click_samples]
        click = np.sin(2 * np.pi * 520 * t_click) * np.exp(-t_click / 0.008)

        # Szum przesuwającej się metalowej szuflady
        drawer_noise = np.random.normal(0, 0.2, n_samples).astype(np.float32)
        sos = scipy.signal.butter(3, [300, 1800], btype="bandpass", fs=sr, output="sos")
        slide = scipy.signal.sosfilt(sos, drawer_noise) * np.exp(-t / 0.35)

        # Cichy brzęk monet (dzwonki 3200 Hz i 4100 Hz)
        coin_ring = 0.15 * np.sin(2 * np.pi * 3200 * t) * np.exp(-t / 0.2) + \
                    0.10 * np.sin(2 * np.pi * 4100 * t) * np.exp(-t / 0.15)

        audio = np.zeros(n_samples, dtype=np.float32)
        audio[:click_samples] += click * 0.5
        audio += (slide * 0.4 + coin_ring * 0.25).astype(np.float32)

        peak = np.max(np.abs(audio)) + 1e-6
        return (audio / peak) * 0.5

    def generate_store_ambience(self, duration_s: float = 15.0) -> np.ndarray:
        """
        Słyszalny, gęsty miks atmosferyczny wnętrza sklepu:
        Cichy szum kompresora lodówek z napojami (50/100/240 Hz),
        klimatyzator sklepów Smile Mart oraz subtelny szmer tła (-20 dBFS).
        """
        sr = self.sample_rate
        n_samples = int(duration_s * sr)
        t = np.linspace(0, duration_s, n_samples, endpoint=False)

        # 1. Harmoniczny dron chłodziarek (50 Hz zasilanie, 100 Hz podwojenie, 240 Hz kompresor)
        fridge_drone = (
            0.18 * np.sin(2 * np.pi * 50.0 * t) +
            0.10 * np.sin(2 * np.pi * 100.0 * t) +
            0.05 * np.sin(2 * np.pi * 240.0 * t)
        ).astype(np.float32)

        # 2. Szum przepływu powietrza klimatyzacji (różowy szum filtrowany 150-1800 Hz)
        white = np.random.normal(0, 1, n_samples).astype(np.float32)
        sos_air = scipy.signal.butter(3, [120, 1600], btype="bandpass", fs=sr, output="sos")
        air_hum = scipy.signal.sosfilt(sos_air, white) * 0.25

        # 3. Subtelny brzęk transformatora świetlówek jarzeniowych (120 Hz i 1200 Hz na poziomie -30 dB)
        light_hum = (0.02 * np.sin(2 * np.pi * 120 * t) + 0.008 * np.sin(2 * np.pi * 1200 * t)).astype(np.float32)

        miks = fridge_drone * 0.4 + air_hum * 0.5 + light_hum
        # Normalizacja do odczuwalnego poziomu tła (-20 dBFS)
        peak = np.max(np.abs(miks)) + 1e-6
        target_gain = 0.20 # stały, wyraźny, ale niezagłuszający poziom
        return (miks / peak * target_gain).astype(np.float32)
