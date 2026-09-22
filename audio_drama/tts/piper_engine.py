import sys
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import requests

from audio_drama.tts.base import BaseTTSEngine

PIPER_VOICE_CATALOG: Dict[str, Dict[str, str]] = {
    "pl_PL-gosia-medium": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/gosia/medium/pl_PL-gosia-medium.onnx.json",
        "gender": "female",
        "description": "Czysty, profesjonalny głos lektorski żeński"
    },
    "pl_PL-darkman-medium": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/darkman/medium/pl_PL-darkman-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/darkman/medium/pl_PL-darkman-medium.onnx.json",
        "gender": "male",
        "description": "Głęboki, wyrazisty głos lektorski męski"
    },
    "pl_PL-mc_speech-medium": {
        "onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/mc_speech/medium/pl_PL-mc_speech-medium.onnx",
        "json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/pl/pl_PL/mc_speech/medium/pl_PL-mc_speech-medium.onnx.json",
        "gender": "male",
        "description": "Neutralny męski głos radiowo-narracyjny"
    }
}

class PiperEngine(BaseTTSEngine):
    """Silnik syntezy mowy Piper TTS z obsługą modeli ONNX dla języka polskiego."""

    def __init__(
        self,
        voices_dir: str | Path = "data/voices",
        default_voice: str = "pl_PL-darkman-medium"
    ):
        self.voices_dir = Path(voices_dir)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.default_voice = default_voice

    def get_available_voices(self) -> List[str]:
        return list(PIPER_VOICE_CATALOG.keys())

    def ensure_voice(self, voice_name: str) -> Tuple[Path, Path]:
        """Upewnia się, że model ONNX i plik konfiguracyjny są pobrane."""
        if voice_name not in PIPER_VOICE_CATALOG:
            # Sprawdzenie czy użytkownik nie podał bezpośredniej ścieżki
            custom_onnx = self.voices_dir / f"{voice_name}.onnx"
            custom_json = self.voices_dir / f"{voice_name}.onnx.json"
            if custom_onnx.exists() and custom_json.exists():
                return custom_onnx, custom_json
            # Jeśli brak w katalogu, użyj domyślnego
            voice_name = self.default_voice

        info = PIPER_VOICE_CATALOG[voice_name]
        onnx_path = self.voices_dir / f"{voice_name}.onnx"
        json_path = self.voices_dir / f"{voice_name}.onnx.json"

        if not onnx_path.exists():
            self._download_file(info["onnx"], onnx_path)
        if not json_path.exists():
            self._download_file(info["json"], json_path)

        return onnx_path, json_path

    @staticmethod
    def _download_file(url: str, dest_path: Path) -> None:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = dest_path.with_suffix(".tmp")
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        temp_path.replace(dest_path)

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
        """
        Syntetyzuje podany tekst do pliku WAV za pomocą Piper TTS.
        Dla Pipera: mniejszy length_scale = szybsza mowa (length_scale = 1.0 / speed).
        Jeśli podano target_sample_rate (domyślnie 44100 Hz), plik jest bezstratnie
        resamplowany z natywnego 22050 Hz do studyjnego standardu 44.1 kHz.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        voice_name = voice or self.default_voice
        onnx_file, json_file = self.ensure_voice(voice_name)

        # Bezpieczne ograniczenie mnożnika długości fonemów
        safe_speed = max(0.5, min(speed, 2.0))
        length_scale = round(1.0 / safe_speed, 3)

        cmd = [
            sys.executable,
            "-m", "piper",
            "-m", str(onnx_file),
            "-c", str(json_file),
            "-f", str(out_p),
            "--length-scale", str(length_scale),
            "--volume", str(max(0.1, min(volume, 2.0)))
        ]

        # Uruchomienie procesu syntezy z podaniem tekstu przez stdin
        clean_text = text.strip()
        if not clean_text:
            clean_text = "..."

        proc = subprocess.run(
            cmd,
            input=clean_text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )

        if proc.returncode != 0:
            err_msg = proc.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"Błąd syntezy Piper TTS (kod {proc.returncode}): {err_msg}")

        # Automatyczny resampling do target_sample_rate (np. 44100 Hz)
        if target_sample_rate is not None and out_p.exists():
            import soundfile as sf
            from audio_drama.dsp.mixer import resample_audio
            raw_audio, sr = sf.read(str(out_p))
            if sr != target_sample_rate:
                resampled = resample_audio(raw_audio, src_sr=sr, dst_sr=target_sample_rate)
                sf.write(str(out_p), resampled, target_sample_rate)

        return out_p
