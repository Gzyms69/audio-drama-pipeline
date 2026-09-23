import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any
import click
import numpy as np
import soundfile as sf
from rich.table import Table
from rich.console import Console

from audio_drama.storage.db import DatabaseManager
from audio_drama.extractor.epub_parser import EpubExtractor
from audio_drama.core.models import Scene, Character, AudioCue, ScreenplayResponse
from audio_drama.director.gbnf_grammar import save_grammar_file
from audio_drama.director.ollama_engine import OllamaEngine
from audio_drama.director.llama_engine import LlamaServerEngine
from audio_drama.director.prompts import build_micro_screenplay_prompt
from audio_drama.director.heuristic import parse_scene_heuristically
from audio_drama.director.casting import CastBibleManager
from audio_drama.tts.piper_engine import PiperEngine, PIPER_VOICE_CATALOG
from audio_drama.tts.edge_engine import EdgeTTSEngine, EDGE_VOICE_CATALOG
from audio_drama.tts.xtts_engine import XTTSEngine
from audio_drama.tts.benchmark import TTSBenchmark
from audio_drama.sfx.stable_audio import StableAudioEngine
from audio_drama.sfx.library import FoleyLibrary
from audio_drama.sfx.foley_matcher import FoleyMatcher
from audio_drama.dsp.mixer import SceneMixer
from audio_drama.telemetry.hardware import get_hardware_telemetry
from audio_drama.telemetry.eta import EtaCalculator
from audio_drama.telemetry.logger import EventLogger
from audio_drama.telemetry.dashboard import PipelineDashboard

@click.group()
def cli():
    """Audio Drama Pipeline - Narzędzie CLI do tworzenia słuchowisk z plików EPUB."""
    pass

@cli.command("init")
@click.option("--db", default="project.db", help="Ścieżka do pliku bazy SQLite.")
def init_db(db: str):
    """Inicjalizuje bazę danych project.db z pełnym schematem."""
    db_mgr = DatabaseManager(db)
    db_mgr.initialize_schema()
    click.echo(f"Zainicjalizowano bazę danych: {Path(db).resolve()}")

@cli.command("import-epub")
@click.argument("epub_file", type=click.Path(exists=True))
@click.option("--db", default="project.db", help="Ścieżka do bazy SQLite.")
@click.option("--chunk-size", default=1500, help="Maksymalny rozmiar sceny w słowach.")
@click.option("--overlap", default=200, help="Rozmiar nakładania w słowach.")
def import_epub(epub_file: str, db: str, chunk_size: int, overlap: int):
    """Importuje książkę EPUB, normalizuje tekst i dzieli na sceny."""
    db_mgr = DatabaseManager(db)
    db_mgr.initialize_schema()

    extractor = EpubExtractor(epub_file)
    meta = extractor.get_metadata()
    db_mgr.set_meta("title", meta["title"])
    db_mgr.set_meta("author", meta["author"])
    db_mgr.set_meta("language", meta["language"])

    click.echo(f"Książka: '{meta['title']}' autorstwa {meta['author']} ({meta['language']})")

    chapters = extractor.extract_chapters()
    click.echo(f"Wyodrębniono {len(chapters)} rozdziałów.")

    total_scenes = 0
    for chap in chapters:
        chunks = EpubExtractor.chunk_text(chap["content"], max_tokens=chunk_size, overlap_tokens=overlap)
        for s_idx, chunk in enumerate(chunks, 1):
            scene_id = f"s_ch{chap['chapter_idx']:02d}_{s_idx:03d}"
            scene = Scene(
                scene_id=scene_id,
                chapter_idx=chap["chapter_idx"],
                scene_idx=s_idx,
                raw_text=chunk,
                status="pending"
            )
            db_mgr.upsert_scene(scene)
            total_scenes += 1

    click.echo(f"Pomyślnie zaimportowano {total_scenes} scen do bazy {db}.")

@cli.command("status")
@click.option("--db", default="project.db", help="Ścieżka do bazy SQLite.")
def status(db: str):
    """Wyświetla status projektu i postęp przetwarzania scen."""
    db_path = Path(db)
    if not db_path.exists():
        click.echo(f"Baza danych {db} nie istnieje. Uruchom najpierw: uv run audio-drama import-epub <plik.epub>")
        return

    db_mgr = DatabaseManager(db)
    title = db_mgr.get_meta("title") or "Nieznany"
    author = db_mgr.get_meta("author") or "Nieznany"
    scenes = db_mgr.list_scenes()
    characters = db_mgr.list_characters()

    click.echo("=" * 60)
    click.echo(f"Projekt Słuchowiska: {title} - {author}")
    click.echo(f"Liczba scen w bazie: {len(scenes)}")
    click.echo(f"Zarejestrowane postacie: {len(characters)}")

    status_counts = {}
    for s in scenes:
        status_counts[s.status] = status_counts.get(s.status, 0) + 1

    for st, count in status_counts.items():
        click.echo(f"  • {st.upper()}: {count} scen")
    click.echo("=" * 60)

@cli.command("export-grammar")
@click.option("--out", default="grammar/audio_screenplay.gbnf", help="Ścieżka docelowa gramatyki GBNF.")
def export_grammar(out: str):
    """Eksportuje reguły gramatyki GBNF dla llama-server."""
    p = save_grammar_file(out)
    click.echo(f"Zapisano gramatykę GBNF w: {p.resolve()}")

@cli.command("benchmark-tts")
@click.option("--text", default="Dzień dobry! Witamy w konbini Smile Mart. Życzymy udanych zakupów!", help="Tekst próbny do syntezy.")
@click.option("--voices", default="pl_PL-darkman-medium,pl_PL-gosia-medium", help="Przecinkowa lista głosów Pipera.")
@click.option("--speed", default=1.0, type=float, help="Tempo mowy.")
@click.option("--out", default="output/benchmark", help="Katalog wyjściowy próbek.")
def benchmark_tts(text: str, voices: str, speed: float, out: str):
    """Uruchamia benchmark porównawczy głosów TTS (A/B testing) dla polskiego tekstu."""
    console = Console()
    console.print(f"[bold cyan]Uruchamianie benchmarku TTS...[/bold cyan]")
    console.print(f"Tekst testowy: [italic]'{text}'[/italic]\n")

    voice_list = [v.strip() for v in voices.split(",") if v.strip()]
    bench = TTSBenchmark(output_dir=out)
    results = bench.compare_voices(text=text, voices=voice_list, speed=speed)

    table = Table(title="Wyniki Porównania Głosów TTS (Piper ONNX)")
    table.add_column("Model Głosu", style="cyan")
    table.add_column("Czas Generacji", justify="right")
    table.add_column("Długość Audio", justify="right")
    table.add_column("Współczynnik RTF", justify="right", style="green")
    table.add_column("Plik Wynikowy", style="yellow")

    for r in results:
        table.add_row(
            r["voice"],
            f"{r['elapsed_seconds']}s",
            f"{r['duration_seconds']}s",
            r["speedup_factor"],
            r["file_path"]
        )

    console.print(table)
    console.print(f"\n[bold green]Próbki gotowe do odsłuchu w:[/bold green] {Path(out).resolve()}")

@cli.command("export-audio")
@click.option("--db", default="project.db", help="Ścieżka do bazy SQLite.")
@click.option("--format", "audio_fmt", type=click.Choice(["m4b", "mp3"]), default="m4b", help="Format wyjściowy.")
@click.option("--out", default=None, help="Ścieżka do pliku wynikowego.")
def export_audio(db: str, audio_fmt: str, out: Optional[str]):
    """Łączy wszystkie wygenerowane sceny w jeden zintegrowany plik audiobooka (.m4b / .mp3)."""
    db_mgr = DatabaseManager(db)
    title = db_mgr.get_meta("title") or "Audiobook"
    author = db_mgr.get_meta("author") or "Nieznany"

    scenes_dir = Path("output/scenes")
    if not scenes_dir.exists():
        click.echo("Katalog output/scenes nie istnieje. Najpierw wygeneruj sceny.")
        return

    scenes = db_mgr.list_scenes()
    mixed_scenes = [s for s in scenes if (scenes_dir / f"{s.scene_id}.wav").exists()]

    if not mixed_scenes:
        click.echo("Brak zmiksowanych plików scen w output/scenes/.")
        return

    click.echo(f"Znaleziono {len(mixed_scenes)} zmiksowanych scen. Przygotowywanie eksportu...")

    concat_list_file = Path("output/concat_list.txt")
    concat_list_file.parent.mkdir(parents=True, exist_ok=True)

    with open(concat_list_file, "w", encoding="utf-8") as f:
        for s in mixed_scenes:
            wav_path = (scenes_dir / f"{s.scene_id}.wav").resolve()
            f.write(f"file '{wav_path}'\n")

    out_file = Path(out) if out else Path(f"output/{title.replace(' ', '_')}.{audio_fmt}")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list_file),
        "-c:a", "aac" if audio_fmt == "m4b" else "libmp3lame",
        "-b:a", "192k",
        "-metadata", f"title={title}",
        "-metadata", f"artist={author}",
        str(out_file)
    ]

    subprocess.run(cmd, check=True)
    click.echo(f"Wyeksportowano zintegrowany audiobook: {out_file.resolve()}")

@cli.command("monitor")
@click.option("--db", default="project.db", help="Ścieżka do bazy SQLite.")
def monitor_cmd(db: str):
    """Uruchamia pełnoekranowy dashboard TUI monitorujący postęp, sprzęt i logi."""
    from rich.live import Live

    db_mgr = DatabaseManager(db)
    title = db_mgr.get_meta("title") or "Audio Drama"
    dashboard = PipelineDashboard(book_title=title)
    logger = EventLogger()
    eta_calc = EtaCalculator(total_items=1)

    with Live(dashboard.generate_view({}, {}, "00:00:00", "--:--:--", []), refresh_per_second=2) as live:
        try:
            while True:
                telemetry = get_hardware_telemetry(db_path=db, stems_dir="stems")
                scenes = db_mgr.list_scenes()
                status_counts = {}
                for s in scenes:
                    status_counts[s.status] = status_counts.get(s.status, 0) + 1

                total_scenes = len(scenes)
                scripted = status_counts.get("scripted", 0) + status_counts.get("synthesized", 0) + status_counts.get("mixed", 0)
                stage_progress = {
                    "step0": f"{total_scenes} scen",
                    "stage1": f"{scripted}/{total_scenes} scen",
                    "stage2": f"{status_counts.get('synthesized', 0)}/{total_scenes} scen",
                    "stage3": f"{status_counts.get('sfx_done', 0)}/{total_scenes} scen",
                    "stage4": f"{status_counts.get('mixed', 0)}/{total_scenes} scen"
                }
                recent_logs = logger.get_recent_logs()
                view = dashboard.generate_view(
                    telemetry=telemetry,
                    stage_progress=stage_progress,
                    elapsed_str=eta_calc.get_formatted_elapsed(),
                    eta_str=eta_calc.get_formatted_eta() if scripted > 0 else "--:--:--",
                    recent_logs=recent_logs
                )
                live.update(view)
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass

@cli.command("run")
@click.argument("epub_file", type=click.Path(exists=True))
@click.option("--db", default="project.db", help="Ścieżka do bazy SQLite.")
@click.option("--chapter", type=int, default=None, help="Przetwórz tylko wskazany numer rozdziału.")
@click.option("--scenes", default=None, help="Zakres scen do przetworzenia, np. s_ch01_001..s_ch01_005.")
@click.option("--director", "director_type", type=click.Choice(["ollama", "llama", "heuristic"]), default="ollama", help="Silnik reżysera.")
@click.option("--tts", "tts_type", type=click.Choice(["xtts", "edge", "piper"]), default="xtts", help="Silnik syntezy mowy TTS (xtts = HuggingFace klonowanie głosu, edge = naturalne głosy studyjne, piper = offline ONNX).")
@click.option("--force", is_flag=True, default=False, help="Wymuś ponowną reżyserię i syntezę scen.")
@click.option("--tmux", is_flag=True, default=False, help="Uruchom proces w tle w trwałej sesji tmux.")
def run_pipeline(epub_file: str, db: str, chapter: Optional[int], scenes: Optional[str], director_type: str, tts_type: str, force: bool, tmux: bool):
    """Uruchamia pełny potok czasu rzeczywistego ze zintegrowanym dashboardem i odsłuchem per-scena."""
    if tmux and "TMUX" not in os.environ:
        session_name = "audio-drama"
        cmd = f"uv run audio-drama run '{Path(epub_file).resolve()}' --db '{Path(db).resolve()}'"
        if chapter:
            cmd += f" --chapter {chapter}"
        if scenes:
            cmd += f" --scenes {scenes}"
        cmd += f" --director {director_type} --tts {tts_type}"
        if force:
            cmd += " --force"

        click.echo(f"Uruchamianie w trwałej sesji tmux: {session_name}")
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name, cmd], check=True)
        click.echo(f"Sesja tmux została utworzona.")
        click.echo(f"Aby się podpiąć, wykonaj: tmux attach -t {session_name}")
        subprocess.run(["tmux", "attach", "-t", session_name])
        return

    logger = EventLogger()
    logger.log("PIPELINE", f"Rozpoczęto zadanie dla: {Path(epub_file).name}")

    db_mgr = DatabaseManager(db)
    db_mgr.initialize_schema()

    # Import metadanych i rozdziałów, jeśli baza jest pusta
    existing_scenes = db_mgr.list_scenes()
    if not existing_scenes:
        extractor = EpubExtractor(epub_file)
        meta = extractor.get_metadata()
        db_mgr.set_meta("title", meta["title"])
        db_mgr.set_meta("author", meta["author"])
        db_mgr.set_meta("language", meta["language"])

        chapters = extractor.extract_chapters()
        for chap in chapters:
            chunks = EpubExtractor.chunk_text(chap["content"], max_tokens=1500, overlap_tokens=200)
            for s_idx, chunk in enumerate(chunks, 1):
                scene_id = f"s_ch{chap['chapter_idx']:02d}_{s_idx:03d}"
                scene = Scene(
                    scene_id=scene_id,
                    chapter_idx=chap["chapter_idx"],
                    scene_idx=s_idx,
                    raw_text=chunk,
                    status="pending"
                )
                db_mgr.upsert_scene(scene)

    all_scenes = db_mgr.list_scenes()
    # Filtrowanie według zakresu lub rozdziału
    if chapter is not None:
        target_scenes = [s for s in all_scenes if s.chapter_idx == chapter]
    elif scenes is not None:
        if ".." in scenes:
            start_s, end_s = scenes.split("..")
            target_scenes = [s for s in all_scenes if start_s <= s.scene_id <= end_s]
        else:
            target_scenes = [s for s in all_scenes if s.scene_id == scenes]
    else:
        target_scenes = all_scenes

    if force:
        for s in target_scenes:
            db_mgr.clear_cues_for_scene(s.scene_id)
            s.status = "pending"
            db_mgr.upsert_scene(s)

    pending_scenes = [s for s in target_scenes if s.status != "mixed"]
    logger.log("PIPELINE", f"Do przetworzenia: {len(pending_scenes)} scen (z {len(target_scenes)} w wybranym zakresie)")

    # Inicjalizacja silników
    cast_mgr = CastBibleManager(db_mgr, engine_type=tts_type)
    if tts_type == "xtts":
        tts_engine = XTTSEngine()
    elif tts_type == "edge":
        tts_engine = EdgeTTSEngine()
    else:
        tts_engine = PiperEngine()
    foley_lib = FoleyLibrary()
    foley_matcher = FoleyMatcher(foley_lib)
    sfx_engine = StableAudioEngine()
    mixer = SceneMixer(sample_rate=44100)

    # Inicjalizacja reżysera
    ollama_engine = OllamaEngine() if director_type == "ollama" else None
    llama_engine = LlamaServerEngine() if director_type == "llama" else None

    # Katalogi wyjściowe
    output_scenes_dir = Path("output/scenes")
    stems_voices_dir = Path("stems/voices")
    stems_ambient_dir = Path("stems/ambient")
    output_scenes_dir.mkdir(parents=True, exist_ok=True)
    stems_voices_dir.mkdir(parents=True, exist_ok=True)
    stems_ambient_dir.mkdir(parents=True, exist_ok=True)

    title = db_mgr.get_meta("title") or "Audio Drama"
    dashboard = PipelineDashboard(book_title=title)
    eta_calc = EtaCalculator(total_items=len(pending_scenes))

    from rich.live import Live
    with Live(dashboard.generate_view({}, {}, "00:00:00", "--:--:--", []), refresh_per_second=2) as live:
        for idx, scene in enumerate(pending_scenes, 1):
            t_start = time.perf_counter()
            logger.log("DIRECTOR", f"Scena {scene.scene_id} (Rozdz. {scene.chapter_idx}): Analiza scenariusza...")

            # FAZA 1: Reżyseria
            cues = db_mgr.get_cues_for_scene(scene.scene_id)
            if not cues or force:
                screenplay = None
                if director_type == "ollama" and ollama_engine:
                    try:
                        cast_reg = db_mgr.list_characters()
                        prompt = build_micro_screenplay_prompt(
                            cast_registry=cast_reg,
                            lookback_cues=[],
                            current_state=None,
                            target_chunk=scene.raw_text,
                            lookahead_chunk=""
                        )
                        screenplay = ollama_engine.generate(prompt)
                    except Exception as e:
                        logger.log("DIRECTOR", f"Ollama niedostępna ({e}). Przełączenie na parser heurystyczny.")
                elif director_type == "llama" and llama_engine:
                    try:
                        cast_reg = db_mgr.list_characters()
                        prompt = build_micro_screenplay_prompt(
                            cast_registry=cast_reg,
                            lookback_cues=[],
                            current_state=None,
                            target_chunk=scene.raw_text,
                            lookahead_chunk=""
                        )
                        screenplay = llama_engine.generate(prompt)
                    except Exception as e:
                        logger.log("DIRECTOR", f"llama-server niedostępny ({e}). Przełączenie na parser heurystyczny.")

                if screenplay is None:
                    screenplay = parse_scene_heuristically(scene)

                db_mgr.clear_cues_for_scene(scene.scene_id)
                # Zapis cues do bazy
                for cue in screenplay.cues:
                    cue.scene_id = scene.scene_id
                    db_mgr.insert_cue(cue)
                    # Dynamiczna obsada postaci
                    cast_mgr.resolve_character(cue.speaker_id)

                scene.bgm_prompt = screenplay.state_update.current_bgm_track
                scene.status = "scripted"
                db_mgr.upsert_scene(scene)
                cues = db_mgr.get_cues_for_scene(scene.scene_id)

            logger.log("TTS", f"Scena {scene.scene_id}: Synteza {len(cues)} kwestii ({tts_type.upper()})...")

            # FAZA 2: Synteza Mowy
            cues_audio = []
            for cue in cues:
                char = cast_mgr.resolve_character(cue.speaker_id)
                voice_name = char.voice_name
                wav_path = stems_voices_dir / f"{cue.cue_id}.wav"

                # Generowanie pliku WAV jeśli nie istnieje lub wymuszono
                if not wav_path.exists() or cue.status != "synthesized" or force:
                    speed = char.speed_factor * cue.delivery.speed
                    pitch = getattr(char, "pitch_offset", 0.0) * 50.0 + (cue.delivery.pitch_shift * 50.0)
                    tts_engine.synthesize(
                        text=cue.text,
                        output_path=wav_path,
                        voice=voice_name,
                        speed=speed,
                        pitch=pitch,
                        volume=1.0
                    )
                    cue.voice_wav_path = str(wav_path)
                    cue.status = "synthesized"
                    db_mgr.insert_cue(cue)

                audio_data, sr = sf.read(str(wav_path))
                # Konwersja do mono float32
                if audio_data.ndim > 1:
                    audio_data = np.mean(audio_data, axis=1)

                cue.duration_ms = round((len(audio_data) / max(sr, 1)) * 1000.0, 1)
                db_mgr.insert_cue(cue)

                pause_after = cue.pause_after_ms if getattr(cue, "pause_after_ms", None) is not None else (220 if cue.cue_type == "dialogue" else 420)
                cues_audio.append({
                    "audio": audio_data,
                    "sample_rate": sr,
                    "pause_after_ms": pause_after
                })

            # Montaż ścieżki lektorskiej
            voice_track, voice_dur_s = mixer.assemble_voice_track(cues_audio)

            # FAZA 3: Atmosfera / BGM
            logger.log("SFX", f"Scena {scene.scene_id}: Generacja tła audio...")
            bgm_file = stems_ambient_dir / f"{scene.scene_id}_bgm.wav"
            if "konbini" in (scene.bgm_prompt or "").lower() or "sklep" in (scene.raw_text or "").lower():
                bgm_audio = foley_lib.generate_store_ambience(duration_s=max(voice_dur_s, 5.0))
                sf.write(str(bgm_file), bgm_audio, 44100)
            else:
                sfx_engine.generate_ambient(
                    prompt=scene.bgm_prompt or "quiet convenience store room tone",
                    duration_seconds=max(voice_dur_s, 5.0),
                    output_path=bgm_file,
                    volume=0.35
                )
                bgm_audio, _ = sf.read(str(bgm_file))
                if bgm_audio.ndim > 1:
                    bgm_audio = np.mean(bgm_audio, axis=1)

            # FAZA 4: Miks i Mastering Foley
            sfx_events = foley_matcher.match_scene_sfx(cues, cues_audio)
            logger.log("MIXER", f"Scena {scene.scene_id}: Mastering DSP z {len(sfx_events)} efektami Foley...")
            out_scene_file = output_scenes_dir / f"{scene.scene_id}.wav"
            mixer.mix_and_master_scene(
                voice_track=voice_track,
                sfx_events=sfx_events,
                bgm_track=bgm_audio,
                output_path=out_scene_file
            )

            scene.status = "mixed"
            db_mgr.upsert_scene(scene)

            t_elapsed = time.perf_counter() - t_start
            eta_calc.update(t_elapsed)
            logger.log("COMPLETE", f"Scena {scene.scene_id} gotowa: {out_scene_file.name} ({voice_dur_s:.1f}s)")

            # Aktualizacja telemetrii na żywo
            telemetry = get_hardware_telemetry(db_path=db, stems_dir="stems")
            scenes_list = db_mgr.list_scenes()
            status_counts = {}
            for s in scenes_list:
                status_counts[s.status] = status_counts.get(s.status, 0) + 1

            total = len(scenes_list)
            scripted = status_counts.get("scripted", 0) + status_counts.get("synthesized", 0) + status_counts.get("mixed", 0)
            stage_progress = {
                "step0": f"{total} scen",
                "stage1": f"{scripted}/{total} scen",
                "stage2": f"{status_counts.get('synthesized', 0)}/{total} scen",
                "stage3": f"{status_counts.get('sfx_done', 0)}/{total} scen",
                "stage4": f"{status_counts.get('mixed', 0)}/{total} scen"
            }
            view = dashboard.generate_view(
                telemetry=telemetry,
                stage_progress=stage_progress,
                elapsed_str=eta_calc.get_formatted_elapsed(),
                eta_str=eta_calc.get_formatted_eta(),
                recent_logs=logger.get_recent_logs()
            )
            live.update(view)

    click.echo(f"\nUkończono przetwarzanie! Zmiksowane sceny znajdują się w: {output_scenes_dir.resolve()}")

@cli.command("gui")
@click.option("--port", default=7860, help="Port serwera Web GUI (Gradio).")
@click.option("--share", is_flag=True, default=False, help="Utwórz publiczny link Gradio Share.")
def gui_cmd(port: int, share: bool):
    """Uruchamia nowoczesny graficzny interfejs Web GUI (Gradio) w przeglądarce."""
    from audio_drama.ui.app import launch_gui
    click.echo(f"Uruchamianie Audio Drama Studio GUI na http://localhost:{port}...")
    launch_gui(port=port, share=share)

@cli.command("ui")
@click.option("--port", default=7860, help="Port serwera Web GUI (Gradio).")
@click.option("--share", is_flag=True, default=False, help="Utwórz publiczny link Gradio Share.")
def ui_cmd(port: int, share: bool):
    """Alias dla komendy gui."""
    from audio_drama.ui.app import launch_gui
    click.echo(f"Uruchamianie Audio Drama Studio GUI na http://localhost:{port}...")
    launch_gui(port=port, share=share)

def main():
    cli()

if __name__ == "__main__":
    main()

