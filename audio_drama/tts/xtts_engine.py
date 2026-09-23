import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import soundfile as sf

from audio_drama.tts.base import BaseTTSEngine
from audio_drama.dsp.mixer import resample_audio

# Automatyczna akceptacja licencji Coqui CPML
os.environ["COQUI_TOS_AGREED"] = "1"

# Zestaw predefiniowanych profili dla słuchowiska
XTTS_PRESET_MAP: Dict[str, Dict[str, str]] = {
    "Ana Florence": {
        "gender": "female",
        "role": "narrator",
        "description": "Ciepły, zbalansowany, intymny głos kobiecy (idealny dla narratorki Keiko)"
    },
    "Claribel Dervla": {
        "gender": "female",
        "role": "dialogue",
        "description": "Żywy, wyrazisty kobiecy głos (Keiko przy kasie / Izumi)"
    },
    "Damian Black": {
        "gender": "male",
        "role": "character",
        "description": "Głęboki, męski głos narracyjny i dialogowy (klient / kierownik sklepu)"
    },
    "Andrew Chipper": {
        "gender": "male",
        "role": "dialogue",
        "description": "Zwykły męski tembr dialogowy (klienci konbini)"
    },
    "Daisy Studious": {
        "gender": "female",
        "role": "dialogue",
        "description": "Młodszy, czysty kobiecy głos (współpracowniczki)"
    },
    "Gracie Wise": {
        "gender": "female",
        "role": "dialogue",
        "description": "Dojrzały kobiecy głos (starsze klientki)"
    }
}


class XTTSEngine(BaseTTSEngine):
    """
    Silnik syntezy mowy oparty o HuggingFace Coqui XTTS-v2.
    Oferuje autentyczny, ludzki rezonans, oddechy i intonację
    poprzez syntezę wielojęzyczną i klonowanie głosu (Zero-Shot Voice Cloning).
    """

    _tts_instance = None  # Singleton instancji modelu

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        default_speaker: str = "Ana Florence",
        cache_dir: str | Path = "data/cache/tts_xtts",
        voices_dir: str | Path = "assets/voices",
        gpu: Optional[bool] = None
    ):
        self.model_name = model_name
        self.default_speaker = default_speaker
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.voices_dir = Path(voices_dir)
        self.voices_dir.mkdir(parents=True, exist_ok=True)

        if gpu is None:
            import torch
            self.use_gpu = torch.cuda.is_available()
        else:
            self.use_gpu = gpu

        self._ensure_model_loaded()

    def _ensure_model_loaded(self) -> None:
        if XTTSEngine._tts_instance is None:
            from TTS.api import TTS
            XTTSEngine._tts_instance = TTS(self.model_name, gpu=self.use_gpu)
        self.tts = XTTSEngine._tts_instance

    def get_available_voices(self) -> List[str]:
        voices = list(XTTS_PRESET_MAP.keys())
        if self.voices_dir.exists():
            for wav_file in self.voices_dir.glob("*.wav"):
                voices.append(wav_file.stem)
        return voices

    def _resolve_speaker_and_wav(self, voice: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        """Określa czy głos to preset nazwy czy plik referencyjny wav."""
        if not voice:
            return self.default_speaker, None

        wav_candidate = Path(voice)
        if wav_candidate.exists() and wav_candidate.suffix.lower() == ".wav":
            return None, str(wav_candidate.resolve())

        named_wav = self.voices_dir / f"{voice}.wav"
        if named_wav.exists():
            return None, str(named_wav.resolve())

        available_speakers = getattr(self.tts, "speakers", []) or []
        if voice in available_speakers:
            return voice, None

        if voice in XTTS_PRESET_MAP:
            return voice, None

        return self.default_speaker, None

    def synthesize(
        self,
        text: str,
        output_path: str | Path,
        voice: Optional[str] = None,
        speed: float = 1.0,
        pitch: float = 0.0,
        volume: float = 1.0,
        target_sample_rate: Optional[int] = 44100
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        speaker_name, speaker_wav = self._resolve_speaker_and_wav(voice)
        temp_wav = self.cache_dir / f"temp_{output_path.stem}.wav"

        kwargs = {
            "text": text,
            "file_path": str(temp_wav),
            "language": "pl",
            "split_sentences": False
        }

        if speaker_wav:
            kwargs["speaker_wav"] = speaker_wav
        else:
            kwargs["speaker"] = speaker_name

        self.tts.tts_to_file(**kwargs)

        audio_data, sr = sf.read(str(temp_wav))
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        audio_data = audio_data.astype(np.float32)

        if volume != 1.0:
            audio_data *= volume

        if target_sample_rate and sr != target_sample_rate:
            audio_data = resample_audio(audio_data, sr, target_sample_rate)
            sr = target_sample_rate

        sf.write(str(output_path), audio_data, sr)

        if temp_wav.exists():
            try:
                temp_wav.unlink()
            except OSError:
                pass

        return output_path
