import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import soundfile as sf

from audio_drama.tts.piper_engine import PiperEngine, PIPER_VOICE_CATALOG

class TTSBenchmark:
    """Narzędzie do porównywania jakości, barwy i wydajności modeli TTS (A/B testing)."""

    def __init__(self, tts_engine: Optional[PiperEngine] = None, output_dir: str | Path = "output/benchmark"):
        self.engine = tts_engine or PiperEngine()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compare_voices(
        self,
        text: str,
        voices: Optional[List[str]] = None,
        speed: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Syntetyzuje ten sam tekst różnymi głosami i zbiera metryki wydajnościowe oraz ścieżki do odsłuchu.
        """
        selected_voices = voices or list(PIPER_VOICE_CATALOG.keys())
        results = []

        for v_name in selected_voices:
            out_file = self.output_dir / f"sample_{v_name}.wav"
            t0 = time.perf_counter()
            self.engine.synthesize(text=text, output_path=out_file, voice=v_name, speed=speed)
            t1 = time.perf_counter()
            elapsed_s = t1 - t0

            # Odczyt długości wygenerowanego audio
            audio_info = sf.info(str(out_file))
            audio_duration_s = audio_info.duration
            rtf = round(audio_duration_s / max(elapsed_s, 0.001), 1)

            results.append({
                "voice": v_name,
                "file_path": str(out_file),
                "elapsed_seconds": round(elapsed_s, 2),
                "duration_seconds": round(audio_duration_s, 2),
                "speedup_factor": f"{rtf}x realtime",
                "sample_rate": audio_info.samplerate
            })

        return results
