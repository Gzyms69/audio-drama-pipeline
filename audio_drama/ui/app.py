import os
import sys
import glob
from pathlib import Path
from typing import Optional, List, Tuple
import soundfile as sf
import numpy as np
import gradio as gr

from audio_drama.storage.db import DatabaseManager
from audio_drama.core.models import Scene, Character
from audio_drama.extractor.epub_parser import EpubExtractor
from audio_drama.director.heuristic import parse_scene_heuristically
from audio_drama.director.casting import CastBibleManager
from audio_drama.tts.edge_engine import EdgeTTSEngine
from audio_drama.tts.piper_engine import PiperEngine
from audio_drama.tts.xtts_engine import XTTSEngine, XTTS_PRESET_MAP
from audio_drama.sfx.library import FoleyLibrary
from audio_drama.sfx.foley_matcher import FoleyMatcher
from audio_drama.sfx.stable_audio import StableAudioEngine
from audio_drama.dsp.mixer import SceneMixer

DEFAULT_EPUB = "/home/gzyms/Downloads/[Sayaka Murata] Dziewczyna z konbini.epub"

TTS_MODEL_OPTIONS = {
    "HuggingFace XTTS-v2 (Naturalny Lektor / Klonowanie Głosu)": "xtts",
    "Edge-TTS (Szybki lektor chmurowy Azure)": "edge",
    "Piper TTS (Lekki lokalny model ONNX)": "piper"
}

DIRECTOR_MODEL_OPTIONS = {
    "Heurystyczny (Reguły, analiza dialogów i frazowanie)": "heuristic",
    "Ollama (Lokalny model LLM)": "ollama",
    "Llama Server (GBNF grammar)": "llama"
}


def find_available_epubs() -> List[str]:
    """Wyszukuje pliki EPUB w katalogu projektu, Downloads i data."""
    paths = []
    if os.path.exists(DEFAULT_EPUB):
        paths.append(DEFAULT_EPUB)
    search_dirs = [Path("."), Path("data"), Path("/home/gzyms/Downloads")]
    for sdir in search_dirs:
        if sdir.exists():
            for f in sdir.glob("*.epub"):
                abs_f = str(f.resolve())
                if abs_f not in paths:
                    paths.append(abs_f)
    return paths or ["Brak plików EPUB"]


def get_existing_scene_files() -> List[str]:
    """Zwraca listę wygenerowanych plików wav w output/scenes."""
    out_dir = Path("output/scenes")
    if not out_dir.exists():
        return []
    wavs = sorted(out_dir.glob("*.wav"))
    return [w.name for w in wavs]


def run_pipeline_action(
    epub_path: str,
    uploaded_file,
    tts_model_label: str,
    director_model_label: str,
    target_scene: str,
    sentence_pause_ms: int,
    bgm_gain_dbfs: float,
    force_regen: bool,
    progress=gr.Progress(track_tqdm=True)
) -> Tuple[Optional[str], str, str]:
    """Główna funkcja wykonawcza potoku wywoływana z poziomu GUI."""
    logs: List[str] = []

    def log(msg: str):
        logs.append(msg)

    # 1. Ustalenie pliku EPUB
    actual_epub = None
    if uploaded_file is not None:
        actual_epub = uploaded_file.name if hasattr(uploaded_file, "name") else str(uploaded_file)
    elif epub_path and os.path.exists(epub_path):
        actual_epub = epub_path
    elif os.path.exists(DEFAULT_EPUB):
        actual_epub = DEFAULT_EPUB

    if not actual_epub or not os.path.exists(actual_epub):
        return None, "❌ Błąd: Nie wskazano prawidłowego pliku EPUB.", "\n".join(logs)

    tts_type = TTS_MODEL_OPTIONS.get(tts_model_label, "xtts")
    director_type = DIRECTOR_MODEL_OPTIONS.get(director_model_label, "heuristic")

    log(f"Rozpoczęto zadanie dla: {Path(actual_epub).name}")
    log(f"Wybrany silnik TTS: {tts_type.upper()}")
    log(f"Wybrany reżyser: {director_type.upper()}")
    log(f"Pauza międzyzdaniowa: {sentence_pause_ms} ms, Tło BGM: {bgm_gain_dbfs} dBFS")

    progress(0.1, desc="Inicjalizacja bazy i ekstrakcja EPUB...")
    db_mgr = DatabaseManager("project.db")
    db_mgr.initialize_schema()

    # Ekstrakcja jeśli baza pusta
    existing_scenes = db_mgr.list_scenes()
    if not existing_scenes:
        extractor = EpubExtractor(actual_epub)
        meta = extractor.get_metadata()
        db_mgr.set_meta("title", meta.get("title", "Audio Drama"))
        db_mgr.set_meta("author", meta.get("author", "Autor"))
        chapters = extractor.extract_chapters()
        for chap in chapters:
            chunks = EpubExtractor.chunk_text(chap["content"], max_tokens=1500, overlap_tokens=200)
            for s_idx, chunk in enumerate(chunks, 1):
                s_id = f"s_ch{chap['chapter_idx']:02d}_{s_idx:03d}"
                db_mgr.upsert_scene(Scene(
                    scene_id=s_id,
                    chapter_idx=chap["chapter_idx"],
                    scene_idx=s_idx,
                    raw_text=chunk,
                    status="pending"
                ))
        all_scenes = db_mgr.list_scenes()
    else:
        all_scenes = existing_scenes

    # Dobór sceny docelowej
    scene_filter = target_scene.strip() if target_scene else "s_ch02_001"
    target_scenes = [s for s in all_scenes if s.scene_id == scene_filter]
    if not target_scenes:
        target_scenes = [all_scenes[0]] if all_scenes else []

    if not target_scenes:
        return None, "❌ Brak scen do przetworzenia w bazie.", "\n".join(logs)

    # Inicjalizacja silników
    progress(0.2, desc="Ładowanie modeli i obsady...")
    cast_mgr = CastBibleManager(db_mgr, engine_type=tts_type)

    if tts_type == "xtts":
        log("Ładowanie modelu HuggingFace Coqui XTTS-v2...")
        tts_engine = XTTSEngine()
    elif tts_type == "edge":
        log("Ładowanie silnika Edge-TTS...")
        tts_engine = EdgeTTSEngine()
    else:
        log("Ładowanie silnika Piper TTS...")
        tts_engine = PiperEngine()

    foley_lib = FoleyLibrary()
    foley_matcher = FoleyMatcher(foley_lib)
    sfx_engine = StableAudioEngine()
    mixer = SceneMixer(sample_rate=44100)

    out_scenes_dir = Path("output/scenes")
    stems_v_dir = Path("stems/voices")
    stems_a_dir = Path("stems/ambient")
    out_scenes_dir.mkdir(parents=True, exist_ok=True)
    stems_v_dir.mkdir(parents=True, exist_ok=True)
    stems_a_dir.mkdir(parents=True, exist_ok=True)

    generated_master_path = None

    for s_i, scene in enumerate(target_scenes):
        if force_regen:
            db_mgr.clear_cues_for_scene(scene.scene_id)
            scene.status = "pending"
            db_mgr.upsert_scene(scene)

        # Faza 1: Reżyseria
        progress(0.3, desc=f"Reżyseria sceny {scene.scene_id}...")
        log(f"Scena {scene.scene_id}: Analiza scenariusza...")
        cues = db_mgr.get_cues_for_scene(scene.scene_id)
        if not cues:
            script_data = parse_scene_heuristically(scene.raw_text, scene.scene_id, cast_mgr)
            scene.bgm_prompt = script_data.bgm_prompt
            db_mgr.upsert_scene(scene)
            for cue in script_data.cues:
                db_mgr.insert_cue(cue)
            cues = db_mgr.get_cues_for_scene(scene.scene_id)

        # Faza 2: Synteza TTS
        log(f"Scena {scene.scene_id}: Synteza {len(cues)} kwestii ({tts_type.upper()})...")
        cues_audio = []
        for c_idx, cue in enumerate(cues):
            pct = 0.3 + (c_idx / max(len(cues), 1)) * 0.4
            progress(pct, desc=f"Synteza kwestii {c_idx+1}/{len(cues)}...")

            wav_name = f"{scene.scene_id}_{cue.cue_order:03d}_{cue.speaker_id}.wav"
            wav_path = stems_v_dir / wav_name

            char = cast_mgr.resolve_character(cue.speaker_id)
            speed = getattr(cue, "speed", 1.0) or 1.0
            if cue.cue_type == "narration":
                speed = 0.94

            if not wav_path.exists() or force_regen:
                tts_engine.synthesize(
                    text=cue.text_content,
                    output_path=wav_path,
                    voice=char.voice_name,
                    speed=speed,
                    target_sample_rate=44100
                )
                cue.voice_wav_path = str(wav_path)
                db_mgr.insert_cue(cue)

            audio_data, sr = sf.read(str(wav_path))
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)

            cue.duration_ms = round((len(audio_data) / max(sr, 1)) * 1000.0, 1)
            db_mgr.insert_cue(cue)

            pause = getattr(cue, "pause_after_ms", None)
            if pause is None:
                pause = sentence_pause_ms if cue.cue_type == "narration" else 220

            cues_audio.append({
                "audio": audio_data,
                "sample_rate": sr,
                "pause_after_ms": pause
            })

        # Montaż głosu
        progress(0.75, desc="Montaż ścieżki wokalnej...")
        voice_track, voice_dur_s = mixer.assemble_voice_track(cues_audio)

        # Faza 3: Tło dźwiękowe
        progress(0.85, desc="Generacja i miks tła audio...")
        bgm_file = stems_a_dir / f"{scene.scene_id}_bgm.wav"
        if "konbini" in (scene.bgm_prompt or "").lower() or "sklep" in (scene.raw_text or "").lower():
            bgm_audio = foley_lib.generate_store_ambience(duration_s=max(voice_dur_s, 5.0))
            sf.write(str(bgm_file), bgm_audio, 44100)
        else:
            sfx_engine.generate_ambient(
                prompt=scene.bgm_prompt or "quiet convenience store room tone",
                duration_seconds=max(voice_dur_s, 5.0),
                output_path=bgm_file,
                sample_rate=44100
            )

        # Faza 4: Efekty Foley i Mastering
        progress(0.92, desc="Mastering DSP i miksowanie...")
        foley_events = foley_matcher.match_events(cues, cues_audio)
        out_scene_file = out_scenes_dir / f"{scene.scene_id}.wav"
        mixer.mix_scene(
            scene_id=scene.scene_id,
            cues_audio=cues_audio,
            bgm_path=bgm_file,
            foley_events=foley_events,
            output_path=out_scene_file
        )
        scene.status = "mixed"
        db_mgr.upsert_scene(scene)

        generated_master_path = str(out_scene_file.resolve())
        log(f"Scena {scene.scene_id} pomyślnie wygenerowana: {out_scene_file.name} ({voice_dur_s:.1f}s)")

    progress(1.0, desc="Zakończono pomyślnie!")
    status_summary = f"✅ Sukces: Wygenerowano słuchowisko dla sceny {target_scenes[0].scene_id}!"
    return generated_master_path, status_summary, "\n".join(logs)


def load_stem_files(scene_name: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], str]:
    """Wczytuje plik master oraz stemy wokalne, ambient i metryki dla wybranej sceny."""
    if not scene_name:
        return None, None, None, None, "Wybierz scenę z listy."

    scene_id = Path(scene_name).stem
    master_path = Path("output/scenes") / f"{scene_id}.wav"
    bgm_path = Path("stems/ambient") / f"{scene_id}_bgm.wav"

    master_res = str(master_path) if master_path.exists() else None
    bgm_res = str(bgm_path) if bgm_path.exists() else None

    # Znajdź pierwszą kwestię wokalną dla podglądu lub zmontowaną ścieżkę
    voice_files = sorted(Path("stems/voices").glob(f"{scene_id}_*.wav"))
    sample_voice = str(voice_files[0]) if voice_files else None

    # Oblicz metryki
    metrics_text = "Brak danych audio."
    if master_path.exists():
        data, sr = sf.read(str(master_path))
        dur = len(data) / sr
        peak = np.max(np.abs(data))
        rms = np.sqrt(np.mean(data**2))
        rms_db = 20 * np.log10(max(rms, 1e-9))
        metrics_text = f"**Czas trwania:** {dur:.1f} s | **Szczyt (Peak):** {peak:.4f} | **RMS Loudness:** {rms_db:.2f} dBFS | **Sample Rate:** {sr} Hz"

    return master_res, sample_voice, bgm_res, master_res, metrics_text


def get_characters_summary() -> str:
    """Zwraca podsumowanie obsady w formacie Markdown."""
    if not os.path.exists("project.db"):
        return "Baza danych nie została jeszcze zainicjalizowana."
    db_mgr = DatabaseManager("project.db")
    chars = db_mgr.list_characters()
    if not chars:
        return "Brak zapisanych postaci w bazie."
    
    md = "| Identyfikator | Postać | Płeć | Silnik | Model / Barwa Głosu | Opis |\n|---|---|---|---|---|---|\n"
    for c in chars:
        md += f"| `{c.id}` | **{c.name}** | {c.gender} | `{c.voice_type}` | `{c.voice_name}` | {c.description} |\n"
    return md


def update_character_voice(char_id: str, new_voice: str) -> str:
    """Aktualizuje przypisaną barwę głosu dla postaci w SQLite."""
    if not os.path.exists("project.db"):
        return "Baza danych nie istnieje."
    db_mgr = DatabaseManager("project.db")
    char = db_mgr.get_character(char_id)
    if not char:
        return f"Nie znaleziono postaci: {char_id}"
    char.voice_name = new_voice
    db_mgr.upsert_character(char)
    return f"✅ Zaktualizowano barwę dla {char.name}: {new_voice}"


def create_ui() -> gr.Blocks:
    """Tworzy i konfiguruje graficzny interfejs Audio Drama Studio."""
    epubs = find_available_epubs()
    default_epub_val = epubs[0] if epubs else ""

    with gr.Blocks(title="Audio Drama Studio") as demo:
        gr.Markdown(
            """
            # 🎧 Audio Drama Studio — Generator Słuchowisk & Dubbingu
            ### Naturalny rytm czytania, zero-shot voice cloning (XTTS-v2), pauzy oddechowe i oprawa dźwiękowa Foley
            """
        )

        with gr.Tabs():
            # ZAKŁADKA 1: STUDIO GENEROWANIA
            with gr.TabItem("🎬 Studio Produkcji (Scene Generator)"):
                with gr.Row():
                    with gr.Column(scale=5):
                        gr.Markdown("### 1. Źródło i Pliki")
                        epub_dropdown = gr.Dropdown(
                            choices=epubs,
                            value=default_epub_val,
                            label="Wybierz plik EPUB z dysku",
                            allow_custom_value=True
                        )
                        upload_box = gr.File(
                            label="...lub wgraj nowy plik EPUB (Drag & Drop)",
                            file_types=[".epub"]
                        )

                        gr.Markdown("### 2. Wybór Modeli")
                        tts_selector = gr.Dropdown(
                            choices=list(TTS_MODEL_OPTIONS.keys()),
                            value="HuggingFace XTTS-v2 (Naturalny Lektor / Klonowanie Głosu)",
                            label="🎙️ Silnik Syntezy Mowy (TTS)"
                        )
                        director_selector = gr.Dropdown(
                            choices=list(DIRECTOR_MODEL_OPTIONS.keys()),
                            value="Heurystyczny (Reguły, analiza dialogów i frazowanie)",
                            label="🎬 Model Reżysera (Director)"
                        )

                        gr.Markdown("### 3. Konfiguracja Sceny i Audio")
                        scene_input = gr.Textbox(
                            value="s_ch02_001",
                            label="Identyfikator Sceny (np. s_ch02_001)",
                            placeholder="s_ch02_001"
                        )
                        pause_slider = gr.Slider(
                            minimum=200, maximum=1000, value=420, step=20,
                            label="Pauza Oddechowa Międzyzdaniowa (ms)"
                        )
                        bgm_slider = gr.Slider(
                            minimum=-45.0, maximum=-15.0, value=-33.0, step=1.0,
                            label="Głośność Tła Otoczenia (dBFS)"
                        )
                        force_check = gr.Checkbox(
                            value=True,
                            label="Wymuś pełną regenerację sceny (--force)"
                        )

                        btn_run = gr.Button("🚀 Generuj Słuchowisko", variant="primary", size="lg")

                    with gr.Column(scale=7):
                        gr.Markdown("### 4. Wygenerowane Słuchowisko")
                        main_player = gr.Audio(
                            label="Odtwarzacz Główny (Master Audio)",
                            type="filepath"
                        )
                        status_display = gr.Markdown("Gotowy do generowania. Wybierz model i kliknij przycisk.")
                        live_logs = gr.Textbox(
                            label="Dziennik Zdarzeń Potoku (Live Log)",
                            lines=14,
                            max_lines=25
                        )

                btn_run.click(
                    fn=run_pipeline_action,
                    inputs=[
                        epub_dropdown,
                        upload_box,
                        tts_selector,
                        director_selector,
                        scene_input,
                        pause_slider,
                        bgm_slider,
                        force_check
                    ],
                    outputs=[main_player, status_display, live_logs]
                )

            # ZAKŁADKA 2: MIKSER I STEMY AUDIO
            with gr.TabItem("🎚️ Mikser & Stemy Audio"):
                gr.Markdown("### Inspekcja Ścieżek Dźwiękowych i Balansu")
                scene_list = get_existing_scene_files()
                initial_scene = scene_list[0] if scene_list else None

                with gr.Row():
                    scene_picker = gr.Dropdown(
                        choices=scene_list,
                        value=initial_scene,
                        label="Wybierz wygenerowaną scenę do odsłuchu"
                    )
                    btn_refresh_scenes = gr.Button("🔄 Odśwież Listę Scen")

                metrics_display = gr.Markdown("Wybierz scenę, aby zobaczyć metryki.")

                with gr.Row():
                    with gr.Column():
                        stem_master = gr.Audio(label="Ścieżka Główna (Master Mix)", type="filepath")
                    with gr.Column():
                        stem_voice = gr.Audio(label="Ścieżka Wokalna (Vocals / Kwestia)", type="filepath")

                with gr.Row():
                    with gr.Column():
                        stem_bgm = gr.Audio(label="Tło Dźwiękowe (Ambience / Chłodziarki)", type="filepath")
                    with gr.Column():
                        stem_foley = gr.Audio(label="Ścieżka Podglądowa", type="filepath")

                scene_picker.change(
                    fn=load_stem_files,
                    inputs=[scene_picker],
                    outputs=[stem_master, stem_voice, stem_bgm, stem_foley, metrics_display]
                )

                def refresh_scenes_action():
                    scenes = get_existing_scene_files()
                    return gr.Dropdown(choices=scenes, value=scenes[0] if scenes else None)

                btn_refresh_scenes.click(
                    fn=refresh_scenes_action,
                    outputs=[scene_picker]
                )

            # ZAKŁADKA 3: OBSADA I KLONOWANIE GŁOSU
            with gr.TabItem("🎭 Obsada & Profile Głosu (Casting)"):
                gr.Markdown("### Dynamiczna Księga Postaci i Przypisanie Barw Głosu")
                cast_table = gr.Markdown(value=get_characters_summary())
                btn_refresh_cast = gr.Button("🔄 Odśwież Obsadę")

                gr.Markdown("#### Zmień Barwę Głosu dla Postaci:")
                with gr.Row():
                    char_id_input = gr.Dropdown(
                        choices=["narrator", "keiko", "klient", "izumi", "menedzer"],
                        value="narrator",
                        label="Identyfikator Postaci"
                    )
                    voice_preset_input = gr.Dropdown(
                        choices=list(XTTS_PRESET_MAP.keys()) + ["pl-PL-ZofiaNeural", "pl-PL-MarekNeural", "pl_PL-gosia-medium"],
                        value="Ana Florence",
                        label="Nowa Barwa / Preset Głosu",
                        allow_custom_value=True
                    )
                    btn_update_voice = gr.Button("Zapisz Zmianę Głosu", variant="secondary")

                update_status = gr.Markdown("")

                btn_update_voice.click(
                    fn=update_character_voice,
                    inputs=[char_id_input, voice_preset_input],
                    outputs=[update_status]
                )

                btn_refresh_cast.click(
                    fn=get_characters_summary,
                    outputs=[cast_table]
                )

        gr.Markdown(
            """
            ---
            *Audio Drama Pipeline v5.0 | Projekt: Dziewczyna z konbini | Architektura: Hexagonal Clean Audio Engine*
            """
        )

    return demo


def launch_gui(port: int = 7860, share: bool = False):
    """Startuje serwer Gradio na wskazanym porcie."""
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=port, share=share)


if __name__ == "__main__":
    launch_gui()
