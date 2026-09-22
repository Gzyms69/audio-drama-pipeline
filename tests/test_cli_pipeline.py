import pytest
from pathlib import Path
from click.testing import CliRunner

from audio_drama.cli import cli
from audio_drama.storage.db import DatabaseManager
from audio_drama.core.models import Scene
from audio_drama.director.heuristic import parse_scene_heuristically
from audio_drama.director.casting import CastBibleManager

def test_heuristic_parser():
    sample_text = (
        "W małym konbini na rogu panował spokój.\n"
        "— Dzień dobry, czy są świeże onigiri? — zapytał klient.\n"
        "— Tak, właśnie dostarczono — odparła Keiko."
    )
    scene = Scene(
        scene_id="s_ch01_001",
        chapter_idx=1,
        scene_idx=1,
        raw_text=sample_text
    )
    screenplay = parse_scene_heuristically(scene)
    assert len(screenplay.cues) == 3
    assert screenplay.cues[0].cue_type == "narration"
    assert screenplay.cues[0].speaker_id == "narrator"
    assert screenplay.cues[1].cue_type == "dialogue"
    assert screenplay.cues[2].cue_type == "dialogue"
    assert "convenience store" in screenplay.state_update.current_bgm_track.lower()

def test_cast_bible_manager(tmp_path):
    db_path = tmp_path / "test.db"
    db_mgr = DatabaseManager(db_path)
    db_mgr.initialize_schema()

    cast_mgr = CastBibleManager(db_mgr)
    narrator = cast_mgr.resolve_character("narrator")
    assert narrator.id == "narrator"
    assert narrator.voice_name == "pl_PL-darkman-medium"

    keiko = cast_mgr.resolve_character("keiko")
    assert keiko.gender == "female"
    assert keiko.voice_name == "pl_PL-gosia-medium"

    shiraha = cast_mgr.resolve_character("shiraha")
    assert shiraha.gender == "male"
    assert shiraha.voice_name == "pl_PL-darkman-medium"

def test_cli_status(tmp_path):
    runner = CliRunner()
    db_path = tmp_path / "empty.db"
    result = runner.invoke(cli, ["init", "--db", str(db_path)])
    assert result.exit_code == 0

    res_status = runner.invoke(cli, ["status", "--db", str(db_path)])
    assert res_status.exit_code == 0
    assert "Liczba scen w bazie: 0" in res_status.output
