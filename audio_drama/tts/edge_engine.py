import os
import asyncio
import tempfile
import concurrent.futures
from pathlib import Path
from typing import Optional, List, Dict
import soundfile as sf

from audio_drama.tts.base import BaseTTSEngine
from audio_drama.dsp.mixer import resample_audio
import edge_tts

EDGE_VOICE_CATALOG: Dict[str, Dict[str, str]] = {
    "pl-PL-ZofiaNeural": {
        "gender": "female",
        "description": "Naturalny, zbalansowany, studyjny głos lektorski żeński (idealny dla Keiko Furukury i narratorki)"
    },
    "pl-PL-MarekNeural": {
        "gender": "male",
        "description": "Ciepły, wyrazisty, naturalny głos lektorski męski (idealny dla klientów i postaci męskich)"
    }
}

class EdgeTTSEngine(BaseTTSEngine):
    """
    Silnik syntezy mowy oparty o Edge-TTS (Microsoft Azure Neural Voices)
    zapewniający studyjną jakość oddechu i intonacji dla języka polskiego bez obciążenia GPU.
    """

    def __init__(
        self,
        default_voice: str = "pl-PL-ZofiaNeural",
        cache_dir: str | Path = "data/cache/tts_edge"
    ):
        self.default_voice = default_voice
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_available_voices(self) -> List[str]:
        return list(EDGE_VOICE_CATALOG.keys())

    @staticmethod
    def _format_rate(speed: float) -> str:
        # speed = 1.0 -> "+0%", 1.1 -> "+10%", 0.9 -> "-10%"
        pct = int(round((speed - 1.0) * 100))
        return f"+{pct}%" if pct >= 0 else f"{pct}%"

    @staticmethod
    def _format_pitch(pitch_hz: float) -> str:
        # pitch_hz = 0 -> "+0Hz", 10 -> "+10Hz", -15 -> "-15Hz"
        hz = int(round(pitch_hz))
        return f"+{hz}Hz" if hz >= 0 else f"{hz}Hz"

    @staticmethod
    def _format_volume(volume: float) -> str:
        # volume = 1.0 -> "+0%", 0.8 -> "-20%"
        pct = int(round((volume - 1.0) * 100))
        return f"+{pct}%" if pct >= 0 else f"{pct}%"

    async def _generate_audio_file(
        self,
        text: str,
        dest_mp3: Path,
        voice: str,
        rate_str: str,
        pitch_str: str,
        volume_str: str
    ) -> None:
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate_str,
            pitch=pitch_str,
            volume=volume_str
        )
        await communicate.save(str(dest_mp3))

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
        Syntetyzuje tekst do pliku WAV za pomocą Edge-TTS z zachowaniem
        studyjnego próbkowania (np. 44.1 kHz).
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        clean_text = text.strip()
        if not clean_text:
            clean_text = "..."

        chosen_voice = voice or self.default_voice
        # Jeśli podano głos piperowy (np. pl_PL-gosia-medium), zmapuj go na odpowiednik Edge-TTS
        if "gosia" in chosen_voice.lower():
            chosen_voice = "pl-PL-ZofiaNeural"
        elif "darkman" in chosen_voice.lower() or "mc_speech" in chosen_voice.lower():
            chosen_voice = "pl-PL-MarekNeural"

        if chosen_voice not in EDGE_VOICE_CATALOG:
            chosen_voice = self.default_voice

        rate_str = self._format_rate(speed)
        pitch_str = self._format_pitch(pitch)
        volume_str = self._format_volume(volume)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
            tmp_mp3_path = Path(tmp_mp3.name)

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(
                        asyncio.run,
                        self._generate_audio_file(
                            clean_text, tmp_mp3_path, chosen_voice, rate_str, pitch_str, volume_str
                        )
                    ).result()
            else:
                asyncio.run(
                    self._generate_audio_file(
                        clean_text, tmp_mp3_path, chosen_voice, rate_str, pitch_str, volume_str
                    )
                )

            # Odczyt MP3 i konwersja do WAV z opcjonalnym resamplingiem
            raw_audio, sr = sf.read(str(tmp_mp3_path))
            target_sr = target_sample_rate or 44100
            if sr != target_sr:
                processed_audio = resample_audio(raw_audio, src_sr=sr, dst_sr=target_sr)
            else:
                processed_audio = raw_audio

            sf.write(str(out_p), processed_audio, target_sr)

        finally:
            if tmp_mp3_path.exists():
                tmp_mp3_path.unlink()

        return out_p
