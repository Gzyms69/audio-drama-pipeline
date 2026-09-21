import pytest
from pathlib import Path
from audio_drama.storage.db import DatabaseManager
from audio_drama.core.models import Character, Scene, AudioCue, DeliveryHints, SfxEvent

@pytest.fixture
def test_db(tmp_path: Path):
    db_file = tmp_path / "test_project.db"
    manager = DatabaseManager(db_file)
    manager.initialize_schema()
    return manager

def test_initialize_and_meta(test_db: DatabaseManager):
    test_db.set_meta("book_title", "Wiedźmin: Ostatnie Życzenie")
    assert test_db.get_meta("book_title") == "Wiedźmin: Ostatnie Życzenie"
    assert test_db.get_meta("non_existing") is None

def test_character_crud(test_db: DatabaseManager):
    char = Character(
        id="geralt",
        name="Geralt z Rivii",
        aliases=["Wiedźmin", "Biały Wilk"],
        voice_type="f5_clone",
        reference_wav_path="voices/geralt_ref.wav",
        pitch_offset=-0.05,
        speed_factor=0.95
    )
    test_db.upsert_character(char)
    fetched = test_db.get_character("geralt")
    assert fetched is not None
    assert fetched.name == "Geralt z Rivii"
    assert "Biały Wilk" in fetched.aliases
    assert fetched.pitch_offset == -0.05

def test_scene_and_cue_lifecycle(test_db: DatabaseManager):
    scene = Scene(
        scene_id="s_ch01_001",
        chapter_idx=1,
        scene_idx=1,
        raw_text="Geralt wszedł do karczmy. Drzwi zatrzasnęły się z hukiem.",
        acoustic_env="tavern_wood",
        bgm_prompt="quiet medieval tavern ambient",
        status="pending"
    )
    test_db.upsert_scene(scene)
    
    cue = AudioCue(
        cue_id="cue_001",
        scene_id="s_ch01_001",
        cue_order=1,
        cue_type="dialogue",
        speaker_id="geralt",
        text="Piwa.",
        delivery=DeliveryHints(emotion="stoic", speed=0.9),
        sfx=[SfxEvent(timing="after_speech", prompt="wooden mug slamming on counter", volume_db=-6.0)]
    )
    test_db.insert_cue(cue)

    cues = test_db.get_cues_for_scene("s_ch01_001")
    assert len(cues) == 1
    assert cues[0].speaker_id == "geralt"
    assert cues[0].text == "Piwa."
    assert len(cues[0].sfx) == 1
    assert cues[0].sfx[0].volume_db == -6.0
