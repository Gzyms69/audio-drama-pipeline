import re
from typing import List, Tuple, Optional
from audio_drama.core.models import (
    Scene,
    AudioCue,
    DeliveryHints,
    SfxEvent,
    StateUpdate,
    ScreenplayResponse,
    AcousticEnvironment
)

# Słowa kluczowe kasjerki konbini (Keiko Furukura)
CASHIER_KEYWORDS = [
    "dzień dobry", "dziękuję", "do widzenia", "witamy",
    "oczywiście", "już podaję", "zapakować", "zbliżyć",
    "rachunek", "paragon", "potwierdzenie", "tylko dwa", "dobrze"
]

# Słowa kluczowe klienta konbini
CUSTOMER_KEYWORDS = [
    "papierosy", "piątkę", "corn dog", "poproszę",
    "suicą", "suica", "proszę razem", "yyy", "ile zostało"
]

def _attribute_dialogue_speaker(
    spoken_text: str,
    narrative_tag: Optional[str],
    last_narrative_context: str,
    recent_speaker: Optional[str]
) -> Tuple[str, DeliveryHints]:
    """
    Inteligentna atrybucja postaci w scenariuszu radiowym na podstawie
    treści dialogu, wtrącenia narracyjnego oraz kontekstu poprzedzającego.
    """
    text_lower = spoken_text.lower()
    tag_lower = (narrative_tag or "").lower()
    ctx_lower = last_narrative_context.lower()

    # 1. Sprawdzenie wtrącenia narracyjnego pod kątem Keiko / 1. osoby narratorki (-łam, np. ukłoniłam się, wzięłam)
    if re.search(r'\b\w+łam\b', tag_lower) or "keiko" in tag_lower or "furukura" in tag_lower:
        # Kwestia wypowiedziana przez Keiko (narratorkę)
        return "keiko", DeliveryHints(emotion="cheerful", speed=1.05, pitch_shift=0.1)

    # 2. Sprawdzenie czy wtrącenie narracyjne wskazuje na inną postać
    if any(k in tag_lower for k in ["klient", "mężczyzna", "powiedział", "odparł", "rzucił"]):
        return "klient", DeliveryHints(emotion="neutral", speed=1.0, pitch_shift=-0.05)
    if "izumi" in tag_lower or "kobieta" in tag_lower or "odpowiedziała" in tag_lower:
        return "izumi", DeliveryHints(emotion="neutral", speed=1.0, pitch_shift=0.0)
    if "kierownik" in tag_lower or "menedżer" in tag_lower or "menedzer" in tag_lower:
        return "menedzer", DeliveryHints(emotion="stoic", speed=0.98, pitch_shift=-0.08)

    # 3. Sprawdzenie kontekstu poprzedzającego akapitu
    if "izumi" in ctx_lower and any(k in ctx_lower for k in ["zagadnęła", "spytała", "mówi"]):
        return "izumi", DeliveryHints(emotion="neutral", speed=1.0, pitch_shift=0.0)
    if ("kierownik" in ctx_lower or "menedżer" in ctx_lower) and any(k in ctx_lower for k in ["został", "powiedział", "mówi"]):
        return "menedzer", DeliveryHints(emotion="stoic", speed=0.98, pitch_shift=-0.08)

    # 4. Sprawdzenie słów kluczowych w samej wypowiedzi
    if any(k in text_lower for k in CASHIER_KEYWORDS):
        return "keiko", DeliveryHints(emotion="cheerful", speed=1.05, pitch_shift=0.1)

    if any(k in text_lower for k in CUSTOMER_KEYWORDS):
        return "klient", DeliveryHints(emotion="neutral", speed=1.0, pitch_shift=-0.05)

    # 5. Płynna naprzemienność dialogu przy ladzie (Klient <-> Kasjerka)
    if recent_speaker == "keiko":
        return "klient", DeliveryHints(emotion="neutral", speed=1.0, pitch_shift=-0.05)
    elif recent_speaker == "klient":
        return "keiko", DeliveryHints(emotion="cheerful", speed=1.05, pitch_shift=0.1)

    # Domyślny fallback: klient sklepu
    return "klient", DeliveryHints(emotion="neutral", speed=1.0, pitch_shift=-0.05)


from audio_drama.extractor.text_sanitizer import TextSanitizer

def parse_scene_heuristically(scene: Scene) -> ScreenplayResponse:
    """
    Deterministyczny, wielogłosowy parser scenariusza dla języka polskiego
    z Sentence-Level Phrasing Engine:
    - Oczyszcza tekst z przypisów i odnośników książkowych,
    - Dzieli narrację na pojedyncze jednostki myślowe ze studyjnymi pauzami oddechowymi,
    - Precyzyjnie rozdziela kwestie mówione od wtrąceń narracyjnych.
    """
    cleaned_raw = TextSanitizer.clean_footnotes(scene.raw_text)
    paragraphs = [p.strip() for p in cleaned_raw.split("\n") if p.strip()]
    cues: List[AudioCue] = []
    active_characters = {"narrator"}
    order = 1

    raw_lower = cleaned_raw.lower()
    bgm_prompt = "quiet indoor room tone, subtle air conditioning hum"
    if "konbini" in raw_lower or "sklep" in raw_lower or "smile mart" in raw_lower:
        bgm_prompt = "convenience store ambient hum, quiet neon buzz, distant refrigeration drone"
    elif "deszcz" in raw_lower or "pada" in raw_lower:
        bgm_prompt = "soft rain against glass window, distant street traffic"
    elif "ulic" in raw_lower or "chodnik" in raw_lower or "stacj" in raw_lower:
        bgm_prompt = "subtle city street ambience, distant urban drone"

    last_narrative_context = ""
    recent_speaker = None

    for p in paragraphs:
        # Dialogi w języku polskim rozpoczynają się od myślnika / pauzy dialogowej
        if p.startswith(("—", "–", "-")):
            raw_dialogue = p.lstrip("—–- ").strip()

            # Podział na część mówioną oraz wtrącenie narracyjne (np. "Dzień dobry! – Ukłoniłam się lekko")
            parts = re.split(r'\s+[—–-]\s+', raw_dialogue, maxsplit=1)
            spoken_text = parts[0].strip()
            narrative_tag = parts[1].strip() if len(parts) > 1 else None

            speaker_id, delivery = _attribute_dialogue_speaker(
                spoken_text=spoken_text,
                narrative_tag=narrative_tag,
                last_narrative_context=last_narrative_context,
                recent_speaker=recent_speaker
            )

            active_characters.add(speaker_id)
            recent_speaker = speaker_id

            # Kwestia wypowiadana przez postać (krótka, naturalna pauza po odpowiedzi)
            cues.append(
                AudioCue(
                    cue_id=f"{scene.scene_id}_{order:03d}",
                    scene_id=scene.scene_id,
                    cue_order=order,
                    cue_type="dialogue",
                    speaker_id=speaker_id,
                    text=spoken_text,
                    delivery=delivery,
                    pause_after_ms=220
                )
            )
            order += 1

            # Jeśli w dialogu było wtrącenie narracyjne (np. "Ukłoniłam się lekko..."), rozbij na zdania i dodaj do narracji
            if narrative_tag:
                tag_sentences = TextSanitizer.split_into_sentences(narrative_tag)
                for s_idx, s_text in enumerate(tag_sentences):
                    is_last = (s_idx == len(tag_sentences) - 1)
                    cues.append(
                        AudioCue(
                            cue_id=f"{scene.scene_id}_{order:03d}",
                            scene_id=scene.scene_id,
                            cue_order=order,
                            cue_type="narration",
                            speaker_id="narrator",
                            text=s_text,
                            delivery=DeliveryHints(emotion="neutral", speed=0.94),
                            pause_after_ms=650 if is_last else 420
                        )
                    )
                    order += 1
                last_narrative_context = narrative_tag

        else:
            # Akapit narracji opisowej - dekompozycja na pojedyncze zdania ze studyjnymi przerwami na oddech
            last_narrative_context = p
            sentences = TextSanitizer.split_into_sentences(p)
            for s_idx, s_text in enumerate(sentences):
                is_last = (s_idx == len(sentences) - 1)
                cues.append(
                    AudioCue(
                        cue_id=f"{scene.scene_id}_{order:03d}",
                        scene_id=scene.scene_id,
                        cue_order=order,
                        cue_type="narration",
                        speaker_id="narrator",
                        text=s_text,
                        delivery=DeliveryHints(emotion="neutral", speed=0.94),
                        pause_after_ms=750 if is_last else 420
                    )
                )
                order += 1

    state_update = StateUpdate(
        location="konbini" if "sklep" in raw_lower else "room",
        active_characters=list(active_characters),
        acoustic_environment=AcousticEnvironment.ROOM_SMALL,
        current_bgm_track=bgm_prompt
    )

    return ScreenplayResponse(state_update=state_update, cues=cues)
