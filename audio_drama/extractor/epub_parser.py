import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

class EpubExtractor:
    def __init__(self, epub_path: str | Path):
        self.epub_path = Path(epub_path)
        if not self.epub_path.exists():
            raise FileNotFoundError(f"Plik EPUB nie istnieje: {self.epub_path}")
        self.book = epub.read_epub(str(self.epub_path), options={"ignore_ncx": True})

    def get_metadata(self) -> Dict[str, Any]:
        title = "Unknown Title"
        author = "Unknown Author"
        language = "pl"

        title_meta = self.book.get_metadata("DC", "title")
        if title_meta:
            title = title_meta[0][0]

        creator_meta = self.book.get_metadata("DC", "creator")
        if creator_meta:
            author = creator_meta[0][0]

        lang_meta = self.book.get_metadata("DC", "language")
        if lang_meta:
            language = lang_meta[0][0]

        return {
            "title": title,
            "author": author,
            "language": language
        }

    @staticmethod
    def clean_html(raw_html: str | bytes) -> str:
        soup = BeautifulSoup(raw_html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "head"]):
            tag.decompose()

        # Zachowaj podział na akapity
        paragraphs = []
        for p in soup.find_all(["p", "h1", "h2", "h3", "h4", "div", "blockquote"]):
            text = p.get_text(separator=" ", strip=True)
            if text:
                paragraphs.append(text)

        if not paragraphs:
            text = soup.get_text(separator="\n\n", strip=True)
            return text

        cleaned_text = "\n\n".join(paragraphs)
        # Normalizacja wielokrotnych spacji i nowych linii
        cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
        return cleaned_text.strip()

    def extract_chapters(self) -> List[Dict[str, Any]]:
        chapters = []
        idx = 1
        for item in self.book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                content = self.clean_html(item.get_content())
                # Ignoruj puste lub skrajnie krótkie pliki (np. okładki, puste strony)
                if len(content.strip()) < 40:
                    continue

                # Spróbuj wyciągnąć tytuł rozdziału z pierwszego nagłówka
                soup = BeautifulSoup(item.get_content(), "html.parser")
                header = soup.find(["h1", "h2", "h3"])
                title = header.get_text(strip=True) if header else f"Rozdział {idx}"

                chapters.append({
                    "chapter_idx": idx,
                    "title": title,
                    "content": content,
                    "item_id": item.get_id()
                })
                idx += 1
        return chapters

    @staticmethod
    def chunk_text(text: str, max_tokens: int = 1500, overlap_tokens: int = 200) -> List[str]:
        """
        Dzieli tekst na kawałki o wielkości max_tokens słów/tokenów,
        respektując granice akapitów (nie tnąc zdań w połowie).
        """
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_count = 0

        for p in paragraphs:
            words = p.split()
            p_len = len(words)

            if current_count + p_len > max_tokens and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                # Zachowaj overlap z ostatnich akapitów
                overlap_chunk = []
                overlap_count = 0
                for prev in reversed(current_chunk):
                    prev_words = prev.split()
                    if overlap_count + len(prev_words) <= overlap_tokens:
                        overlap_chunk.insert(0, prev)
                        overlap_count += len(prev_words)
                    else:
                        break
                current_chunk = overlap_chunk
                current_count = overlap_count

            current_chunk.append(p)
            current_count += p_len

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks
