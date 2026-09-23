import re
from typing import List

class TextSanitizer:
    """
    Normalizator i czyściciel tekstu literackiego przed syntezą TTS.
    Usuwa odnośniki do przypisów, glosariusz tłumacza oraz koryguje interpunkcję.
    """

    @staticmethod
    def clean_footnotes(text: str) -> str:
        """
        Usuwa odnośniki do przypisów z tekstu (np. 'konbini 1 ,' -> 'konbini,')
        oraz wycina całe bloki przypisów tłumacza z końca tekstu/rozdziału.
        """
        lines = text.split("\n")
        filtered_lines: List[str] = []

        for line in lines:
            line_str = line.strip()
            # Wykrywanie linii przypisów tłumacza/autora (np. '1 Konbini... (przyp. tłum.)')
            if re.match(r'^\s*\d+\s+[A-ZĄĆĘŁŃÓŚŹŻ]', line_str):
                if any(kw in line_str.lower() for kw in ["przyp. tłum", "przyp. aut", "przyp. red", "skrót od", "popularna", "banknot", "karta płatnicza"]):
                    continue
            if re.search(r'\(przyp\.\s*(tłum|aut|red)\.?\)', line_str, re.IGNORECASE):
                continue
            filtered_lines.append(line)

        cleaned = "\n".join(filtered_lines)

        # Zamiana twardych spacji na zwykłe
        cleaned = cleaned.replace("\xa0", " ")

        # Usunięcie numerów przypisów występujących bezpośrednio przed znakami interpunkcyjnymi:
        # np. "konbini 1 ," -> "konbini," | "piątkę 3 ." -> "piątkę." | "potwierdzenie 4 ?" -> "potwierdzenie?"
        cleaned = re.sub(r'(\w+)\s+\d+\s*([,\.\?!…])', r'\1\2', cleaned)

        # Usunięcie numerów przypisów stojących po słowie przed spacją:
        # np. "Suicą 5 poproszę" -> "Suicą poproszę"
        cleaned = re.sub(r'(\w+)\s+\d+\s+(?=[a-ząćęłńóśźżA-ZĄĆĘŁŃÓŚŹŻ])', r'\1 ', cleaned)

        # Usunięcie wiszących pojedynczych myślników
        cleaned = re.sub(r'\s+–\s*$', '', cleaned, flags=re.MULTILINE)

        # Usunięcie wielokrotnych spacji
        cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)

        return cleaned.strip()

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """
        Dzieli akapit tekstu na pojedyncze jednostki zdaniowe/myśli.
        """
        cleaned = text.strip()
        if not cleaned:
            return []

        # Dzielenie po kropkach, wykrzyknikach, znakach zapytania i wielokropkach
        # z zachowaniem integralności skrótów (np. itp., np., itd.)
        parts = re.split(r'(?<=[.!?…])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ—–„"\'\d])', cleaned)
        sentences = [p.strip() for p in parts if p.strip()]
        return sentences if sentences else [cleaned]
