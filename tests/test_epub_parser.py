import pytest
from pathlib import Path
import ebooklib
from ebooklib import epub
from audio_drama.extractor.epub_parser import EpubExtractor

@pytest.fixture
def sample_epub_path(tmp_path: Path) -> Path:
    book = epub.EpubBook()
    book.set_identifier("id_test_123")
    book.set_title("Testowa Opowieść")
    book.set_language("pl")
    book.add_author("Andrzej Testowy")

    # Rozdział 1
    c1 = epub.EpubHtml(title="Rozdział I", file_name="chap_01.xhtml", lang="pl")
    c1.content = """
    <html>
      <head><title>Rozdział I</title></head>
      <body>
        <h1>Rozdział I</h1>
        <p>Geralt wszedł do karczmy. W środku śmierdziało piwem i mokrym psem.</p>
        <p>— Czego tu? — zapytał karczmarz, wycierając brudną szmatą blat.</p>
        <p>— Piwa — rzekł wiedźmin. — I spokoju.</p>
      </body>
    </html>
    """
    book.add_item(c1)
    
    # Dodaj NCX i Nav wymagane przez specyfikację EPUB
    ncx = epub.EpubNcx()
    book.add_item(ncx)
    nav = epub.EpubNav()
    book.add_item(nav)

    book.toc = (epub.Link("chap_01.xhtml", "Rozdział I", "chap_01"),)
    book.spine = ["nav", c1]

    epub_file = tmp_path / "test_book.epub"
    epub.write_epub(str(epub_file), book, {})
    return epub_file

def test_epub_metadata_and_extraction(sample_epub_path: Path):
    extractor = EpubExtractor(sample_epub_path)
    metadata = extractor.get_metadata()
    assert metadata["title"] == "Testowa Opowieść"
    assert "Andrzej Testowy" in metadata["author"]

    chapters = extractor.extract_chapters()
    assert len(chapters) == 1
    assert chapters[0]["chapter_idx"] == 1
    assert "Geralt wszedł do karczmy" in chapters[0]["content"]
    assert "Rozdział I" in chapters[0]["title"]

def test_chunking_narrative():
    long_text = "\n\n".join([f"To jest akapit numer {i}. Zawiera słowa opisujące scenę w lesie." for i in range(100)])
    chunks = EpubExtractor.chunk_text(long_text, max_tokens=200, overlap_tokens=30)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 250 for c in chunks)
