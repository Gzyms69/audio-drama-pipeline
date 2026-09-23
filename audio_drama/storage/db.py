import json
import sqlite3
from pathlib import Path
from typing import List, Optional
from audio_drama.core.models import Character, Scene, AudioCue, DeliveryHints, SfxEvent

class DatabaseManager:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def initialize_schema(self) -> None:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executescript("""
            CREATE TABLE IF NOT EXISTS project_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS characters (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                aliases TEXT NOT NULL,
                voice_type TEXT NOT NULL,
                voice_name TEXT,
                gender TEXT DEFAULT 'unknown',
                description TEXT,
                reference_wav_path TEXT,
                pitch_offset REAL DEFAULT 0.0,
                speed_factor REAL DEFAULT 1.0
            );

            CREATE TABLE IF NOT EXISTS scenes (
                scene_id TEXT PRIMARY KEY,
                chapter_idx INTEGER NOT NULL,
                scene_idx INTEGER NOT NULL,
                raw_text TEXT NOT NULL,
                acoustic_env TEXT,
                bgm_prompt TEXT,
                status TEXT DEFAULT 'pending'
            );

            CREATE TABLE IF NOT EXISTS audio_cues (
                cue_id TEXT PRIMARY KEY,
                scene_id TEXT NOT NULL,
                cue_order INTEGER NOT NULL,
                cue_type TEXT NOT NULL,
                speaker_id TEXT NOT NULL,
                text_content TEXT NOT NULL,
                emotion TEXT,
                speed REAL,
                pitch_shift REAL,
                sfx_json TEXT,
                voice_wav_path TEXT,
                duration_ms REAL,
                status TEXT DEFAULT 'pending',
                FOREIGN KEY (scene_id) REFERENCES scenes(scene_id),
                FOREIGN KEY (speaker_id) REFERENCES characters(id)
            );

            CREATE INDEX IF NOT EXISTS idx_cues_scene ON audio_cues(scene_id);
            CREATE INDEX IF NOT EXISTS idx_cues_status ON audio_cues(status);
            CREATE INDEX IF NOT EXISTS idx_scenes_chapter ON scenes(chapter_idx, scene_idx);
            """)

            # Bezpieczne migracje kolumn w characters
            cursor.execute("PRAGMA table_info(characters)")
            cols = [c[1] for c in cursor.fetchall()]
            if "voice_name" not in cols:
                cursor.execute("ALTER TABLE characters ADD COLUMN voice_name TEXT")
            if "gender" not in cols:
                cursor.execute("ALTER TABLE characters ADD COLUMN gender TEXT DEFAULT 'unknown'")
            if "description" not in cols:
                cursor.execute("ALTER TABLE characters ADD COLUMN description TEXT")
            conn.commit()

    def set_meta(self, key: str, value: str) -> None:
        with self.get_connection() as conn:
            conn.execute(
                "INSERT INTO project_meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value)
            )
            conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT value FROM project_meta WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row["value"] if row else None

    def upsert_character(self, character: Character) -> None:
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO characters (id, name, aliases, voice_type, voice_name, gender, description, reference_wav_path, pitch_offset, speed_factor)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    aliases = excluded.aliases,
                    voice_type = excluded.voice_type,
                    voice_name = excluded.voice_name,
                    gender = excluded.gender,
                    description = excluded.description,
                    reference_wav_path = excluded.reference_wav_path,
                    pitch_offset = excluded.pitch_offset,
                    speed_factor = excluded.speed_factor
            """, (
                character.id,
                character.name,
                json.dumps(character.aliases, ensure_ascii=False),
                character.voice_type,
                character.voice_name,
                character.gender,
                character.description,
                character.reference_wav_path,
                character.pitch_offset,
                character.speed_factor
            ))
            conn.commit()

    def get_character(self, char_id: str) -> Optional[Character]:
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM characters WHERE id = ?", (char_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return Character(
                id=row["id"],
                name=row["name"],
                aliases=json.loads(row["aliases"]),
                voice_type=row["voice_type"],
                voice_name=row["voice_name"] if "voice_name" in row.keys() else None,
                gender=row["gender"] if "gender" in row.keys() and row["gender"] else "unknown",
                description=row["description"] if "description" in row.keys() else None,
                reference_wav_path=row["reference_wav_path"],
                pitch_offset=row["pitch_offset"],
                speed_factor=row["speed_factor"]
            )

    def list_characters(self) -> List[Character]:
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM characters")
            rows = cursor.fetchall()
            return [
                Character(
                    id=row["id"],
                    name=row["name"],
                    aliases=json.loads(row["aliases"]),
                    voice_type=row["voice_type"],
                    voice_name=row["voice_name"] if "voice_name" in row.keys() else None,
                    gender=row["gender"] if "gender" in row.keys() and row["gender"] else "unknown",
                    description=row["description"] if "description" in row.keys() else None,
                    reference_wav_path=row["reference_wav_path"],
                    pitch_offset=row["pitch_offset"],
                    speed_factor=row["speed_factor"]
                )
                for row in rows
            ]

    def upsert_scene(self, scene: Scene) -> None:
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO scenes (scene_id, chapter_idx, scene_idx, raw_text, acoustic_env, bgm_prompt, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scene_id) DO UPDATE SET
                    chapter_idx = excluded.chapter_idx,
                    scene_idx = excluded.scene_idx,
                    raw_text = excluded.raw_text,
                    acoustic_env = excluded.acoustic_env,
                    bgm_prompt = excluded.bgm_prompt,
                    status = excluded.status
            """, (
                scene.scene_id,
                scene.chapter_idx,
                scene.scene_idx,
                scene.raw_text,
                scene.acoustic_env,
                scene.bgm_prompt,
                scene.status
            ))
            conn.commit()

    def list_scenes(self) -> List[Scene]:
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM scenes ORDER BY chapter_idx, scene_idx")
            return [
                Scene(
                    scene_id=row["scene_id"],
                    chapter_idx=row["chapter_idx"],
                    scene_idx=row["scene_idx"],
                    raw_text=row["raw_text"],
                    acoustic_env=row["acoustic_env"],
                    bgm_prompt=row["bgm_prompt"],
                    status=row["status"]
                )
                for row in cursor.fetchall()
            ]

    def insert_cue(self, cue: AudioCue) -> None:
        sfx_dicts = [s.model_dump() for s in cue.sfx]
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO audio_cues (
                    cue_id, scene_id, cue_order, cue_type, speaker_id,
                    text_content, emotion, speed, pitch_shift, sfx_json,
                    voice_wav_path, duration_ms, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cue_id) DO UPDATE SET
                    scene_id = excluded.scene_id,
                    cue_order = excluded.cue_order,
                    cue_type = excluded.cue_type,
                    speaker_id = excluded.speaker_id,
                    text_content = excluded.text_content,
                    emotion = excluded.emotion,
                    speed = excluded.speed,
                    pitch_shift = excluded.pitch_shift,
                    sfx_json = excluded.sfx_json,
                    voice_wav_path = excluded.voice_wav_path,
                    duration_ms = excluded.duration_ms,
                    status = excluded.status
            """, (
                cue.cue_id,
                cue.scene_id,
                cue.cue_order,
                cue.cue_type,
                cue.speaker_id,
                cue.text,
                cue.delivery.emotion,
                cue.delivery.speed,
                cue.delivery.pitch_shift,
                json.dumps(sfx_dicts, ensure_ascii=False),
                cue.voice_wav_path,
                cue.duration_ms,
                cue.status
            ))
            conn.commit()

    def get_cues_for_scene(self, scene_id: str) -> List[AudioCue]:
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM audio_cues WHERE scene_id = ? ORDER BY cue_order",
                (scene_id,)
            )
            cues = []
            for row in cursor.fetchall():
                raw_sfx = json.loads(row["sfx_json"]) if row["sfx_json"] else []
                sfx_list = [SfxEvent(**item) for item in raw_sfx]
                cues.append(
                    AudioCue(
                        cue_id=row["cue_id"],
                        scene_id=row["scene_id"],
                        cue_order=row["cue_order"],
                        cue_type=row["cue_type"],
                        speaker_id=row["speaker_id"],
                        text=row["text_content"],
                        delivery=DeliveryHints(
                            emotion=row["emotion"] or "neutral",
                            speed=row["speed"] or 1.0,
                            pitch_shift=row["pitch_shift"] or 0.0
                        ),
                        sfx=sfx_list,
                        voice_wav_path=row["voice_wav_path"],
                        duration_ms=row["duration_ms"],
                        status=row["status"]
                    )
                )
            return cues

    def clear_cues_for_scene(self, scene_id: str) -> None:
        """Usuwa wszystkie kwestie dla danej sceny (np. przy ponownej reżyserii)."""
        with self.get_connection() as conn:
            conn.execute("DELETE FROM audio_cues WHERE scene_id = ?", (scene_id,))
            conn.commit()
