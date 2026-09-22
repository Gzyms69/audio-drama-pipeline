import re
from typing import List
from audio_drama.core.models import (
    Scene,
    AudioCue,
    DeliveryHints,
    SfxEvent,
    StateUpdate,
    ScreenplayResponse,
    AcousticEnvironment
)

def parse_scene_heuristically(scene: Scene) -> ScreenplayResponse:
    """
    Deterministyczny parser scenariusza dla języka polskiego,
    używany jako bezpieczny fallback przy braku aktywnego serwera LLM.
    """
    paragraphs = [p.strip() for p in scene.raw_text.split("\n") if p.strip()]
    cues: List[AudioCue] = []
    active_characters = set()
    order = 1

    # Heurystyczne wykrywanie tła atmosferycznego
    raw_lower = scene.raw_text.lower()
    bgm_prompt = "quiet indoor room tone, subtle air conditioning hum"
    if "konbini" in raw_lower or "sklep" in raw_lower or "smile mart" in raw_lower:
        bgm_prompt = "convenience store ambient hum, quiet neon buzz, distant refrigeration drone"
    elif "deszcz" in raw_lower or "pada" in raw_lower:
        bgm_prompt = "soft rain against glass window, distant street traffic"
    elif "ulic" in raw_lower or "chodnik" in raw_lower or "stacj" in raw_lower:
        bgm_prompt = "subtle city street ambience, distant urban drone"

    for p in paragraphs:
        # Dialogi w języku polskim zaczynają się od myślnika / pauzy
        if p.startswith(("—", "–", "-")):
            dialogue_text = p.lstrip("—–- ").strip()
            # Sprawdzenie prostej atrybucji dialogowej
            speaker_id = "postac"
            if "shiraha" in dialogue_text.lower():
                speaker_id = "shiraha"
            elif "keiko" in dialogue_text.lower() or "furukura" in dialogue_text.lower():
                speaker_id = "keiko"

            active_characters.add(speaker_id)
            cues.append(
                AudioCue(
                    cue_id=f"{scene.scene_id}_{order:03d}",
                    scene_id=scene.scene_id,
                    cue_order=order,
                    cue_type="dialogue",
                    speaker_id=speaker_id,
                    text=dialogue_text,
                    delivery=DeliveryHints(emotion="neutral", speed=1.0)
                )
            )
        else:
            # Narracja opisowa
            cues.append(
                AudioCue(
                    cue_id=f"{scene.scene_id}_{order:03d}",
                    scene_id=scene.scene_id,
                    cue_order=order,
                    cue_type="narration",
                    speaker_id="narrator",
                    text=p,
                    delivery=DeliveryHints(emotion="neutral", speed=1.0)
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
