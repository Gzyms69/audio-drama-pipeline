from typing import Optional, Dict, Tuple
from audio_drama.core.models import Character
from audio_drama.storage.db import DatabaseManager

class CastBibleManager:
    """Zarządza dynamiczną Księgą Głosów w SQLite i przypisuje profile mowy."""

    def __init__(self, db: DatabaseManager, engine_type: str = "edge"):
        self.db = db
        self.engine_type = engine_type
        self._ensure_narrator()

    def _get_engine_voices(self, gender: str) -> str:
        """Zwraca identyfikator modelu w zależności od aktywnego silnika TTS."""
        if self.engine_type == "edge":
            return "pl-PL-ZofiaNeural" if gender == "female" else "pl-PL-MarekNeural"
        else: # piper
            return "pl_PL-gosia-medium" if gender == "female" else "pl_PL-darkman-medium"

    def _ensure_narrator(self) -> None:
        """Dla 'Dziewczyny z konbini' narrator jest postacią pierwszoosobową: 36-letnią Keiko."""
        existing = self.db.get_character("narrator")
        expected_voice = self._get_engine_voices("female")
        if not existing or existing.voice_name != expected_voice or existing.gender != "female":
            self.db.upsert_character(
                Character(
                    id="narrator",
                    name="Keiko Furukura (Narratorka)",
                    aliases=["Narrator", "Lektor"],
                    voice_type=self.engine_type,
                    voice_name=expected_voice,
                    gender="female",
                    description="Główny głos narracyjny Keiko Furukury, spokojny i introspektywny",
                    speed_factor=0.94,
                    pitch_offset=0.0
                )
            )

    def resolve_character(self, char_id: str, suggested_name: Optional[str] = None) -> Character:
        existing = self.db.get_character(char_id)
        expected_voice_female = self._get_engine_voices("female")
        expected_voice_male = self._get_engine_voices("male")

        # Słownik postaci ze specyfikacji 'Dziewczyna z konbini'
        # Format: (gender, name, speed_factor, pitch_offset, description)
        known: Dict[str, Tuple[str, str, float, float, str]] = {
            "narrator": ("female", "Keiko (Narratorka)", 0.94, 0.0, "Narracja 1. osoby Keiko"),
            "keiko": ("female", "Keiko Furukura", 1.05, 0.1, "Keiko przy kasie sklepu konbini"),
            "furukura": ("female", "Keiko Furukura", 1.05, 0.1, "Keiko Furukura"),
            "klient": ("male", "Klient w sklepie", 1.0, -0.05, "Męski klient konbini"),
            "klientka": ("female", "Klientka w sklepie", 1.0, 0.0, "Kobieca klientka konbini"),
            "shiraha": ("male", "Shiraha", 0.95, -0.15, "Shiraha, cyniczny były pracownik"),
            "sugawara": ("female", "Sugawara", 1.0, 0.0, "Sugawara, pracownica sklepu"),
            "izumi": ("female", "Pani Izumi", 1.02, 0.0, "Pani Izumi, koordynatorka dorywcza"),
            "menedzer": ("male", "Kierownik sklepu", 0.98, -0.08, "Kierownik sklepu Smile Mart"),
            "manager": ("male", "Kierownik sklepu", 0.98, -0.08, "Kierownik sklepu Smile Mart"),
            "kierownik": ("male", "Kierownik sklepu", 0.98, -0.08, "Kierownik sklepu Smile Mart"),
            "mami": ("female", "Mami", 1.04, 0.05, "Mami, młodsza siostra Keiko")
        }

        lower_id = char_id.lower()
        if lower_id in known:
            gender, name, speed, pitch, desc = known[lower_id]
        else:
            name = suggested_name or char_id.capitalize()
            is_female = lower_id.endswith(("ko", "ka")) or any(k in lower_id for k in ["pani", "kobieta", "dziewczyna", "siostra"])
            gender = "female" if is_female else "male"
            speed = 1.0
            pitch = 0.0
            desc = f"Obsadzona postać: {name}"

        voice_name = expected_voice_female if gender == "female" else expected_voice_male

        char = Character(
            id=char_id,
            name=name,
            aliases=[name],
            voice_type=self.engine_type,
            voice_name=voice_name,
            gender=gender,
            description=desc,
            speed_factor=speed,
            pitch_offset=pitch
        )
        self.db.upsert_character(char)
        return char
