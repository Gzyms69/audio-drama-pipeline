import os
from pathlib import Path
from typing import Optional
import numpy as np
import soundfile as sf

class StableAudioEngine:
    """
    Silnik generacji tła atmosferycznego i efektów dźwiękowych przy użyciu Stable Audio Open 1.0.
    Posiada mechanizm proceduralnego generowania tła awaryjnego (fallback) przy braku wag GPU.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-audio-open-1.0",
        device: str = "cuda" if "HIP_VISIBLE_DEVICES" in os.environ or "CUDA_VISIBLE_DEVICES" in os.environ else "cpu",
        sample_rate: int = 44100
    ):
        self.model_id = model_id
        self.device = device
        self.sample_rate = sample_rate
        self._pipe = None

    def _lazy_load_pipeline(self):
        """Ładuje pipeline dyfuzyjny diffusers na żądanie."""
        if self._pipe is None:
            try:
                import torch
                from diffusers import StableAudioPipeline
                self._pipe = StableAudioPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
                )
                self._pipe = self._pipe.to(self.device)
            except Exception as e:
                # Brak diffusers lub wag - przejście do trybu proceduralnego
                self._pipe = "fallback"

    def generate_ambient(
        self,
        prompt: str,
        duration_seconds: float = 30.0,
        output_path: str | Path = "stems/ambient.wav",
        volume: float = 0.5
    ) -> Path:
        """
        Generuje stereofoniczne tło atmosferyczne dla sceny.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        self._lazy_load_pipeline()

        if self._pipe != "fallback" and self._pipe is not None:
            try:
                import torch
                audio = self._pipe(
                    prompt,
                    negative_prompt="Low quality, noisy, distorted, speech, vocals",
                    num_inference_steps=50,
                    audio_end_in_s=duration_seconds,
                    num_waveforms_per_prompt=1
                ).audios[0] # (channels, samples)
                
                audio_np = audio.cpu().numpy().T # (samples, 2)
                sf.write(str(out_p), audio_np * volume, self.sample_rate)
                return out_p
            except Exception:
                pass

        # Proceduralny generator ambientu (syntetyczny dron/pokój)
        samples = int(duration_seconds * self.sample_rate)
        # Generowanie brązowego szumu (ciche tło klimatyzacji / pokoju)
        white_noise = np.random.normal(0, 1, (samples, 2)).astype(np.float32)
        # Filtracja akumulacyjna dla profilu 1/f^2 (ciepły szum)
        brown_noise = np.cumsum(white_noise, axis=0)
        brown_noise /= np.max(np.abs(brown_noise) + 1e-6)

        # Dodanie subtelnego drona 60 Hz (klimatyzatory konbini)
        t = np.linspace(0, duration_seconds, samples, endpoint=False)
        drone = (np.sin(2 * np.pi * 60 * t) * 0.15 + np.sin(2 * np.pi * 120 * t) * 0.05).astype(np.float32)
        drone_stereo = np.column_stack([drone, drone])

        final_ambient = (brown_noise * 0.4 + drone_stereo * 0.3) * volume
        sf.write(str(out_p), final_ambient.astype(np.float32), self.sample_rate)
        return out_p
