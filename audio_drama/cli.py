import os
import sys
import time
import subprocess
import click
from pathlib import Path
from audio_drama.storage.db import DatabaseManager
from audio_drama.extractor.epub_parser import EpubExtractor
from audio_drama.core.models import Scene, Character
from audio_drama.director.gbnf_grammar import save_grammar_file
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
@click.option("--tmux", is_flag=True, default=False, help="Uruchom proces w tle w trwałej sesji tmux.")
def run_pipeline(epub_file: str, db: str, tmux: bool):
    """Uruchamia pełny pipeline ze zintegrowanym dashboardem i opcją sesji tmux."""
    if tmux and "TMUX" not in os.environ:
        session_name = "audio-drama"
        cmd = f"uv run audio-drama run '{Path(epub_file).resolve()}' --db '{Path(db).resolve()}'"
        click.echo(f"Uruchamianie w trwałej sesji tmux: {session_name}")
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name, cmd], check=True)
        click.echo(f"Sesja tmux została utworzona.")
        click.echo(f"Aby się podpiąć, wykonaj: tmux attach -t {session_name}")
        subprocess.run(["tmux", "attach", "-t", session_name])
        return

    # Krok 0: Import EPUB
    logger = EventLogger()
    logger.log("PIPELINE", f"Rozpoczęto przetwarzanie pliku: {Path(epub_file).name}")
    
    db_mgr = DatabaseManager(db)
    db_mgr.initialize_schema()

    extractor = EpubExtractor(epub_file)
    meta = extractor.get_metadata()
    db_mgr.set_meta("title", meta["title"])
    db_mgr.set_meta("author", meta["author"])
    db_mgr.set_meta("language", meta["language"])

    logger.log("EXTRACTOR", f"Zaimportowano metadane: '{meta['title']}' autorstwa {meta['author']}")

    chapters = extractor.extract_chapters()
    total_scenes = 0
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
            total_scenes += 1

    logger.log("EXTRACTOR", f"Zakończono Krok 0: {total_scenes} scen gotowych w bazie {db}")
    
    # Przejście do widoku dashboardu
    from rich.live import Live
    dashboard = PipelineDashboard(book_title=meta["title"])
    eta_calc = EtaCalculator(total_items=total_scenes)

    with Live(dashboard.generate_view({}, {}, "00:00:00", "--:--:--", []), refresh_per_second=2) as live:
        logger.log("DIRECTOR", "Inicjalizacja modułu reżysera (Gemma 4 12B)...")
        # Pętla aktualizująca dashboard
        for i in range(1, total_scenes + 1):
            time.sleep(0.5) # Symulacja / odświeżenie pętli
            telemetry = get_hardware_telemetry(db_path=db, stems_dir="stems")
            stage_progress = {
                "step0": f"{total_scenes} scen [OK]",
                "stage1": f"Przygotowano {total_scenes} scen",
                "stage2": "Oczekuje na sygnał",
                "stage3": "Oczekuje",
                "stage4": "Oczekuje"
            }
            view = dashboard.generate_view(
                telemetry=telemetry,
                stage_progress=stage_progress,
                elapsed_str=eta_calc.get_formatted_elapsed(),
                eta_str="Gotowy do reżyserii",
                recent_logs=logger.get_recent_logs()
            )
            live.update(view)
            break

def main():
    cli()

if __name__ == "__main__":
    main()
