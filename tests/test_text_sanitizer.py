import pytest
from audio_drama.extractor.text_sanitizer import TextSanitizer

def test_clean_footnotes():
    raw = (
        "Wszystkie mieszały się ze sobą, stały się jednym odgłosem konbini 1 , który ciągle uderzał w bębenki.\n"
        "Wykładałam onigiri 2 , które dopiero co przyjechały.\n"
        "— A, i jeszcze papierosy, piątkę 3 .\n"
        "— Suicą 5 poproszę.\n"
        "1 Konbini to sklep (przyp. tłum.).\n"
        "2 Onigiri to kulka ryżowa (przyp. tłum.)."
    )

    cleaned = TextSanitizer.clean_footnotes(raw)

    assert "1 ," not in cleaned
    assert "konbini, który" in cleaned
    assert "2 ," not in cleaned
    assert "onigiri, które" in cleaned
    assert "piątkę." in cleaned
    assert "Suicą poproszę" in cleaned
    assert "przyp. tłum" not in cleaned
    assert "Konbini to sklep" not in cleaned

def test_split_into_sentences():
    para = "Sklep wypełniały dźwięki. Melodyjka informująca, że ktoś wchodzi do środka. Wszystkie mieszały się ze sobą!"
    sentences = TextSanitizer.split_into_sentences(para)

    assert len(sentences) == 3
    assert sentences[0] == "Sklep wypełniały dźwięki."
    assert sentences[1] == "Melodyjka informująca, że ktoś wchodzi do środka."
    assert sentences[2] == "Wszystkie mieszały się ze sobą!"
