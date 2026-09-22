from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Any

class BaseTTSEngine(ABC):
    """Abstrakcyjny interfejs bazowy dla silników syntezy mowy TTS."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        output_path: str | Path,
        voice: Optional[str] = None,
        speed: float = 1.0,
        pitch: float = 0.0,
        volume: float = 1.0,
        target_sample_rate: Optional[int] = None
    ) -> Path:
        """Syntetyzuje tekst do pliku audio WAV i zwraca ścieżkę do pliku."""
        pass

    @abstractmethod
    def get_available_voices(self) -> List[str]:
        """Zwraca listę identyfikatorów dostępnych modeli głosowych."""
        pass
