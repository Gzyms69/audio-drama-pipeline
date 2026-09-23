import re
from typing import List, Dict, Any, Optional
import numpy as np

from audio_drama.sfx.library import FoleyLibrary
from audio_drama.core.models import AudioCue

class FoleyMatcher:
    """
    Analizator semantyczny przypisujący efekty dźwiękowe Foley
    do osi czasu scenariusza na podstawie słów kluczowych i kontekstu.
    """

    def __init__(self, library: Optional[FoleyLibrary] = None):
        self.library = library or FoleyLibrary()

    def match_scene_sfx(
        self,
        cues: List[AudioCue],
        cues_timing: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Dopasowuje efekty dźwiękowe do kwestii w scenie.
        cues_timing: lista dictów z kluczami:
          - 'audio': np.ndarray
          - 'sample_rate': int
          - 'pause_after_ms': int
        Zwraca listę zdarzeń SFX w formacie miksera:
          [{'name': str, 'offset_s': float, 'audio': np.ndarray, 'volume_db': float}]
        """
        sfx_events: List[Dict[str, Any]] = []

        # Obliczenie dokładnych punktów startowych (w sekundach) dla każdej kwestii
        current_time_s = 0.0
        cue_offsets_s: List[float] = []
        for timing in cues_timing:
            cue_offsets_s.append(current_time_s)
            sr = timing.get("sample_rate", 44100)
            audio = timing.get("audio")
            duration_s = (len(audio) / sr) if audio is not None else 0.0
            pause_s = timing.get("pause_after_ms", 300) / 1000.0
            current_time_s += duration_s + pause_s

        # Reguła 1: Melodyjka wejściowa konbini (jeśli tekst sceny odnosi się do sklepu/wejścia)
        first_cues_text = " ".join([c.text.lower() for c in cues[:4]])
        if any(k in first_cues_text for k in ["melodyjka", "wchodzi", "sklep", "smile mart", "otwar"]):
            chime_audio = self.library.get_sound("konbini_chime")
            sfx_events.append({
                "name": "konbini_chime",
                "offset_s": 0.0,
                "audio": chime_audio,
                "volume_db": -6.0
            })

        # Reguła 2: Analiza poszczególnych kwestii
        for i, cue in enumerate(cues):
            if i >= len(cue_offsets_s):
                break

            cue_start_s = cue_offsets_s[i]
            text_lower = cue.text.lower()

            # Skaner kodów kreskowych
            if any(k in text_lower for k in ["skaner", "piknięć", "potwierdzenie", "czytnik", "piknięcia"]):
                beep_audio = self.library.get_sound("scanner_beep")
                sfx_events.append({
                    "name": "scanner_beep",
                    "offset_s": cue_start_s + 0.25,
                    "audio": beep_audio,
                    "volume_db": -7.0
                })

            # Szelest folii / opakowań onigiri
            if any(k in text_lower for k in ["onigiri", "foli", "szelest", "torebk", "zapakować", "papierosy"]):
                rustle_audio = self.library.get_sound("plastic_rustle")
                sfx_events.append({
                    "name": "plastic_rustle",
                    "offset_s": max(0.0, cue_start_s - 0.1),
                    "audio": rustle_audio,
                    "volume_db": -11.0
                })

            # Psiknięcie płynu do dezynfekcji rąk
            if any(k in text_lower for k in ["zdezynfekowałam", "sprej", "alkohol", "spray", "atomizer"]):
                spray_audio = self.library.get_sound("alcohol_spray")
                sfx_events.append({
                    "name": "alcohol_spray",
                    "offset_s": cue_start_s + 0.1,
                    "audio": spray_audio,
                    "volume_db": -9.0
                })

            # Kasa fiskalna / szuflada / płatność
            if any(k in text_lower for k in ["paragon", "rachunek", "suica", "banknot", "reszta", "kasa"]):
                cash_audio = self.library.get_sound("cash_register")
                sfx_events.append({
                    "name": "cash_register",
                    "offset_s": cue_start_s + 0.3,
                    "audio": cash_audio,
                    "volume_db": -10.0
                })

        return sfx_events
