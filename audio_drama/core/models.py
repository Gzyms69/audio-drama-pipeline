from enum import Enum
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class AcousticEnvironment(str, Enum):
    ROOM_SMALL = "room_small"
    HALL_STONE = "hall_stone"
    FOREST_OPEN = "forest_open"
    CAVE_WET = "cave_wet"
    TAVERN_WOOD = "tavern_wood"
    DUNGEON = "dungeon"
    STREET_CITY = "street_city"

class DeliveryHints(BaseModel):
    emotion: str = Field(
        default="neutral",
        description="Emocjonalna intonacja: neutral, whisper, angry, nervous, sad, commanding, stoic, joyful, trembling"
    )
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Mnożnik tempa wypowiedzi (np. 1.15)")
    pitch_shift: float = Field(default=0.0, ge=-0.5, le=0.5, description="Względna modulacja tonu głosu")

class SfxEvent(BaseModel):
    timing: Literal["before_speech", "simultaneous", "after_speech"] = Field(
        default="simultaneous",
        description="Punkt w czasie wyzwolenia dźwięku względem mowy"
    )
    prompt: str = Field(..., description="Opis semantyczny dźwięku, np. 'heavy iron gate closing'")
    volume_db: float = Field(default=-12.0, le=0.0, ge=-40.0, description="Względna głośność efektu w dB")

class AudioCue(BaseModel):
    cue_id: str = Field(..., description="Identyfikator kwestii, np. c_ch01_001")
    scene_id: Optional[str] = Field(default=None, description="Identyfikator sceny nadrzędnej")
    cue_order: int = Field(default=1, ge=1, description="Kolejność w ramach sceny")
    cue_type: Literal["dialogue", "narration"] = Field(default="dialogue")
    speaker_id: str = Field(..., description="ID postaci z Cast Registry lub 'narrator'")
    text: str = Field(..., description="Treść wypowiedzi do zsyntezowania przez TTS")
    delivery: DeliveryHints = Field(default_factory=DeliveryHints)
    sfx: List[SfxEvent] = Field(default_factory=list)
    voice_wav_path: Optional[str] = None
    duration_ms: Optional[float] = None
    pause_after_ms: int = Field(default=350, description="Długość naturalnej pauzy po kwestii w milisekundach")
    status: str = "pending"

class StateUpdate(BaseModel):
    location: str = Field(..., description="Identyfikator lokacji w scenie")
    active_characters: List[str] = Field(default_factory=list, description="Lista postaci fizycznie obecnych w scenie")
    acoustic_environment: AcousticEnvironment = Field(default=AcousticEnvironment.ROOM_SMALL)
    current_bgm_track: Optional[str] = Field(default=None, description="Identyfikator lub prompt pętli tła muzycznego")

class ScreenplayResponse(BaseModel):
    state_update: StateUpdate
    cues: List[AudioCue]

class Character(BaseModel):
    id: str = Field(..., description="Identyfikator postaci (np. geralt, keiko)")
    name: str = Field(..., description="Imię lub miano postaci")
    aliases: List[str] = Field(default_factory=list, description="Alternatywne formy zwracania się do postaci")
    voice_type: str = Field(default="piper", description="Silnik głosu: piper, f5_clone, xtts")
    voice_name: Optional[str] = Field(default=None, description="Identyfikator modelu głosu, np. pl_PL-darkman-medium")
    gender: str = Field(default="unknown", description="Płeć postaci: female, male, unknown")
    description: Optional[str] = Field(default=None, description="Profil psychofizyczny i cechy głosu")
    reference_wav_path: Optional[str] = None
    pitch_offset: float = 0.0
    speed_factor: float = 1.0

class Scene(BaseModel):
    scene_id: str
    chapter_idx: int
    scene_idx: int
    raw_text: str
    acoustic_env: Optional[str] = "room_small"
    bgm_prompt: Optional[str] = None
    status: str = "pending"
