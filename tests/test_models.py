import pytest
from pydantic import ValidationError
from audio_drama.core.models import (
    AcousticEnvironment,
    DeliveryHints,
    SfxEvent,
    AudioCue,
    StateUpdate,
    ScreenplayResponse,
    Character,
    Scene
)

def test_sfx_event_validation():
    sfx = SfxEvent(
        timing="before_speech",
        prompt="door creaking slowly",
        volume_db=-12.0
    )
    assert sfx.timing == "before_speech"
    assert sfx.volume_db == -12.0

def test_delivery_hints_validation():
    delivery = DeliveryHints(
        emotion="whisper",
        speed=1.1,
        pitch_shift=-0.05
    )
    assert delivery.emotion == "whisper"
    assert delivery.speed == 1.1

def test_audio_cue_creation():
    cue = AudioCue(
        cue_id="c_001",
        scene_id="s_ch01_01",
        cue_order=1,
        cue_type="dialogue",
        speaker_id="geralt",
        text="Nie teraz.",
        delivery=DeliveryHints(emotion="stoic", speed=0.95),
        sfx=[
            SfxEvent(timing="simultaneous", prompt="swords clashing in distance", volume_db=-18.0)
        ]
    )
    assert cue.speaker_id == "geralt"
    assert len(cue.sfx) == 1
    assert cue.sfx[0].timing == "simultaneous"

def test_state_update_validation():
    state = StateUpdate(
        location="tavern_dark",
        active_characters=["geralt", "jaskier"],
        acoustic_environment=AcousticEnvironment.TAVERN_WOOD,
        current_bgm_track="bgm_tavern_lively"
    )
    assert state.acoustic_environment == AcousticEnvironment.TAVERN_WOOD
    assert "geralt" in state.active_characters

def test_screenplay_response_parsing():
    raw_data = {
        "state_update": {
            "location": "stone_crypt",
            "active_characters": ["geralt"],
            "acoustic_environment": "hall_stone",
            "current_bgm_track": "bgm_dark_drone"
        },
        "cues": [
            {
                "cue_id": "c_001",
                "scene_id": "s_01",
                "cue_order": 1,
                "cue_type": "narration",
                "speaker_id": "narrator",
                "delivery": {
                    "emotion": "neutral",
                    "speed": 1.0,
                    "pitch_shift": 0.0
                },
                "text": "W kryptach panował chłód.",
                "sfx": [
                    {
                        "timing": "before_speech",
                        "prompt": "cold wind howl in underground tunnel",
                        "volume_db": -10.0
                    }
                ]
            }
        ]
    }
    response = ScreenplayResponse.model_validate(raw_data)
    assert response.state_update.location == "stone_crypt"
    assert len(response.cues) == 1
    assert response.cues[0].cue_type == "narration"
