import sys
import click
from pathlib import Path
from audio_drama.storage.db import DatabaseManager
from audio_drama.extractor.epub_parser import EpubExtractor
from audio_drama.core.models import Scene, Character
from audio_drama.director.gbnf_grammar import save_grammar_file

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
@click.option("--out", default="audio_screenplay.gbnf", help="Ścieżka docelowa gramatyki GBNF.")
def export_grammar(out: str):
    """Eksportuje reguły gramatyki GBNF dla llama-server."""
    p = save_grammar_file(out)
    click.echo(f"Zapisano gramatykę GBNF w: {p.resolve()}")

def main():
    cli()

if __name__ == "__main__":
    main()
