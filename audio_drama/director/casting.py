from typing import Optional
from audio_drama.core.models import Character
from audio_drama.storage.db import DatabaseManager

class CastBibleManager:
    """Zarządza dynamiczną Księgą Głosów w SQLite i przypisuje profile mowy."""

    def __init__(self, db: DatabaseManager):
        self.db = db
        self._ensure_narrator()

    def _ensure_narrator(self) -> None:
        if not self.db.get_character("narrator"):
            self.db.upsert_character(
                Character(
                    id="narrator",
                    name="Lektor Główny",
                    aliases=["Narrator"],
                    voice_type="piper",
                    voice_name="pl_PL-darkman-medium",
                    gender="male",
                    description="Główny głos narracyjny, wyrazisty i spokojny",
                    speed_factor=1.0,
                    pitch_offset=0.0
                )
            )

    def resolve_character(self, char_id: str, suggested_name: Optional[str] = None) -> Character:
        existing = self.db.get_character(char_id)
        if existing:
            return existing

        # Wykrywanie płci i profilu głosu na podstawie imienia/ID
        name = suggested_name or char_id.capitalize()
        lower_id = char_id.lower()
        # Słownik znanych postaci z Dziewczyny z konbini (Sayaka Murata)
        known = {
            "keiko": ("female", "pl_PL-gosia-medium", "Keiko Furukura, 36-letnia pracownica konbini"),
            "furukura": ("female", "pl_PL-gosia-medium", "Keiko Furukura"),
            "shiraha": ("male", "pl_PL-darkman-medium", "Shiraha, cyniczny, neurotyczny były pracownik"),
            "sugawara": ("female", "pl_PL-gosia-medium", "Sugawara, pracownica sklepu"),
            "izumi": ("female", "pl_PL-gosia-medium", "Pani Izumi, współpracowniczka"),
            "menedzer": ("male", "pl_PL-darkman-medium", "Kierownik sklepu Smile Mart"),
            "manager": ("male", "pl_PL-darkman-medium", "Kierownik sklepu"),
            "mami": ("female", "pl_PL-gosia-medium", "Mami, młodsza siostra Keiko")
        }

        if lower_id in known:
            gender, voice_name, desc = known[lower_id]
        else:
            is_female = lower_id.endswith(("ko", "ka")) or any(k in lower_id for k in ["pani", "kobieta", "dziewczyna", "siostra"])
            gender = "female" if is_female else "male"
            voice_name = "pl_PL-gosia-medium" if is_female else "pl_PL-darkman-medium"
            desc = f"Automatycznie obsadzona postać: {name}"

        char = Character(
            id=char_id,
            name=name,
            aliases=[name],
            voice_type="piper",
            voice_name=voice_name,
            gender=gender,
            description=f"Automatycznie obsadzona postać: {name}",
            speed_factor=1.0,
            pitch_offset=0.0
        )
        self.db.upsert_character(char)
        return char
