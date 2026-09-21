# Audio Drama Pipeline 🎧📖

Modularny pipeline AI przekształcający książki w formacie **EPUB** w immersyjne, wielogłosowe **słuchowiska radiowe (Audio Drama)** ze zintegrowaną reżyserią tekstu, efektami foley (SFX), tłem atmosferycznym (BGM) i automatycznym miksem wielościeżkowym (DSP).

Projekt został specjalnie zaprojektowany i zoptymalizowany pod sprzęt:
* **Karta graficzna:** AMD Radeon RX 9060 XT (RDNA 4, Navi 44, `gfx1200`), **16 GB VRAM pod ROCm 7.2**.
* **Procesor:** Intel Core i5-14600KF (14 rdzeni, 20 wątków logicznych do 5.3 GHz).
* **Pamięć RAM:** 32 GB DDR5.
* **System operacyjny:** Ubuntu 24.04.4 LTS (Noble Numbat).

---

## 🏗️ Architektura Przepływu Danych (Sekwencyjne Fazowanie VRAM)

Uruchamianie wszystkich modeli naraz w 16 GB VRAM prowadzi do fragmentacji pamięci i błędów Out-Of-Memory. Pipeline działa w architekturze etapowej z izolacją procesów:

```
[EPUB] ➔ (Krok 0: Ekstrakcja CPU) ➔ [project.db SQLite]
       ➔ (Faza 1: Reżyseria LLM GPU - Gemma 4 12B) ➔ [Scenariusz Audio Cues] ➔ zwolnienie VRAM
       ➔ (Faza 2: Synteza Mowy TTS - F5-TTS / Kokoro) ➔ [Stemy Głosowe WAV] ➔ zwolnienie VRAM
       ➔ (Faza 3: Dźwięki & Muzyka - Bank SFX + Stable Audio) ➔ [Stemy Dźwiękowe] ➔ zwolnienie VRAM
       ➔ (Faza 4 & 5: Silnik Miksu DSP & Mastering CPU) ➔ [Gotowy Audiobook .m4b / .flac]
```

---

## ⚡ Modele SOTA w Pipeline

| Rola w Pipeline | Model | Architektura / Backend | Zużycie VRAM / RAM |
| :--- | :--- | :--- | :--- |
| **Reżyseria Makro & Mikro** | **Gemma 4 12B-it (`UD-Q4_K_XL`)** | `llama-server` (`gfx1200` + GBNF) lub `ollama` | ~9.4 GB VRAM (z KV 32k `q8_0`) |
| **Głosy Postaci (Dialogi)** | **F5-TTS Polish (`Sticzu/marek-f5tts-polish`)** | Non-autoregressive Flow Matching (ROCm 7.2) | ~4.2 GB VRAM |
| **Lektor / Narrator** | **Kokoro-82M Polish** | ONNX Runtime na CPU (20 wątków i5-14600KF) | **0 GB VRAM** (~600 MB RAM) |
| **Efekty Foley punktowe** | **Lokalny Bank Sampli 48kHz WAV** | Indeks SQLite + Wyszukiwanie wektorowe CLAP | **0 GB VRAM** (<5 ms czas dostępu) |
| **Atmosfery & Drony** | **Stable Audio Open 1.0** | Latent Diffusion (ROCm 7.2) | ~4.0 GB VRAM |
| **Miks DSP & Mastering** | **Pedalboard + FFmpeg** | Sidechain Ducking, Limiter, EBU R128 (-16 LUFS) | **0 GB VRAM** (przetwarzanie per scena) |

---

## 🛠️ Kompilacja `llama.cpp` pod architekturę RDNA 4 (`gfx1200`)

Dla osiągnięcia maksymalnej wydajności i obsługi gramatyki GBNF:
```bash
git clone https://github.com/ggerganov/llama.cpp.git ~/llama.cpp
cd ~/llama.cpp
mkdir build && cd build
cmake .. \
  -DGGML_HIPBLAS=ON \
  -DAMDGPU_TARGETS=gfx1200 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA_FA_ALL_QUANTS=ON
cmake --build . --config Release -j$(nproc)
```

---

## 🚀 Szybki Start (Instrukcja CLI)

### 1. Instalacja środowiska (uv)
```bash
cd "audio-drama-pipeline"
uv sync
```

### 2. Wyeksportowanie gramatyki GBNF
```bash
uv run audio-drama export-grammar --out grammar/audio_screenplay.gbnf
```

### 3. Inicjalizacja bazy i import książki EPUB
```bash
# Inicjalizacja bazy project.db i podział książki na sceny
uv run audio-drama import-epub sciezka/do/ksiazki.epub --db project.db --chunk-size 1500 --overlap 200

# Sprawdzenie stanu zaimportowanych scen
uv run audio-drama status --db project.db
```

---

## 🧪 Uruchomienie Testów Jednostkowych

Wszystkie moduły objęte są testami jednostkowymi TDD:
```bash
uv run pytest
```
Wynik: **12/12 passed (100% green)**.
