import json
from typing import List, Optional
from audio_drama.core.models import Character, AudioCue, StateUpdate

def build_macro_cast_prompt(book_title: str, text_sample: str) -> str:
    return f"""Jesteś reżyserem wielogłosowego słuchowiska radiowego.
Twoim zadaniem jest przeanalizowanie fragmentu książki "{book_title}" i stworzenie globalnego rejestru postaci (Cast Registry) oraz profilu akustycznego.

Zidentyfikuj wszystkich występujących bohaterów. Dla każdego określ:
1. id (unikalny identyfikator małymi literami, np. geralt, jaskier, innkeeper)
2. name (pełne imię / nazwa)
3. aliases (inne formy, zaimki lub tytuły)
4. voice_type ('f5_clone' dla postaci dialogowych, 'kokoro_narrator' dla lektora)
5. pitch_offset (sugerowana modulacja wysokości tonu od -0.2 do +0.2)
6. speed_factor (tempo mówienia od 0.8 do 1.2)

Zwróć odpowiedź WYŁĄCZNIE jako listę JSON obiektów postaci:
```json
[
  {{
    "id": "narrator",
    "name": "Lektor Główny",
    "aliases": ["Narrator"],
    "voice_type": "kokoro_narrator",
    "pitch_offset": 0.0,
    "speed_factor": 1.0
  }},
  {{
    "id": "geralt",
    "name": "Geralt z Rivii",
    "aliases": ["Wiedźmin", "Biały Wilk"],
    "voice_type": "f5_clone",
    "pitch_offset": -0.1,
    "speed_factor": 0.95
  }}
]
```

FRAGMENT KSIĄŻKI:
\"\"\"
{text_sample[:10000]}
\"\"\"
"""

def build_micro_screenplay_prompt(
    cast_registry: List[Character],
    lookback_cues: List[AudioCue],
    current_state: Optional[StateUpdate],
    target_chunk: str,
    lookahead_chunk: str
) -> str:
    cast_summary = "\n".join([f"- {c.id} ({c.name}, aliasy: {', '.join(c.aliases)})" for c in cast_registry])
    
    lookback_text = "Brak wcześniejszego kontekstu (początek sceny)."
    if lookback_cues:
        lines = []
        for cue in lookback_cues[-5:]:
            lines.append(f"[{cue.speaker_id}]: {cue.text}")
        lookback_text = "\n".join(lines)

    state_json = json.dumps(current_state.model_dump(), ensure_ascii=False) if current_state else "{}"

    return f"""Jesteś reżyserem słuchowiska audio. Przekształć poniższy fragment tekstu [TARGET CHUNK] w ustrukturyzowany scenariusz audio (JSON).

ZASADY REŻYSERII:
1. Rozdziel partię opisową od kwestii wypowiadanych przez bohaterów (cue_type: 'dialogue' lub 'narration').
2. Dla narracji użyj speaker_id: 'narrator'.
3. Rozwiąż atrybucje opóźnione (np. '— rzekł Jan') i anafory na podstawie [LOOKAHEAD BUFFER].
4. Dodaj efekty dźwiękowe (SFX) dla wyraźnych akcji fizycznych (kroki, uderzenia, trzaskanie drzwiami, brzęk miecza).
5. Dobierz precyzyjne emocje do delivery: neutral, whisper, angry, nervous, sad, commanding, stoic, joyful, trembling.
6. Nie generuj wpisów dla tekstu z [LOOKAHEAD BUFFER]! Służy on wyłącznie do odczytu podmiotu i atrybucji.

ZAREJESTROWANA OBSADA (CAST REGISTRY):
{cast_summary}

POPRZEDNI STAN I OSTATNIE DIALOGI [LOOKBACK BUFFER]:
Aktualny stan: {state_json}
Ostatnie kwestie:
{lookback_text}

TEKST DO PRZETWORZENIA [TARGET CHUNK]:
\"\"\"
{target_chunk}
\"\"\"

KONTEKST WYPRZEDZAJĄCY [LOOKAHEAD BUFFER] (TYLKO DO ODCZYTU ATRYBUCJI):
\"\"\"
{lookahead_chunk}
\"\"\"
"""
