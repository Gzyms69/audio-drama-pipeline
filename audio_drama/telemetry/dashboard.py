from typing import Dict, Any, List
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn

class PipelineDashboard:
    def __init__(self, book_title: str = "Audio Drama"):
        self.book_title = book_title

    def generate_view(
        self,
        telemetry: Dict[str, Any],
        stage_progress: Dict[str, Any],
        elapsed_str: str,
        eta_str: str,
        recent_logs: List[str]
    ) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", size=10),
            Layout(name="storage", size=3),
            Layout(name="logs")
        )

        # 1. Header
        header_text = Text()
        header_text.append(f"🎧 AUDIO DRAMA PIPELINE | {self.book_title}\n", style="bold cyan")
        header_text.append(f"⏱️ Czas trwania: {elapsed_str}  │  ⏳ ETA: {eta_str}", style="bold yellow")
        layout["header"].update(Panel(header_text, style="cyan"))

        # 2. Main split: Hardware & Stage Progress
        layout["main"].split_row(
            Layout(name="hardware", ratio=1),
            Layout(name="stages", ratio=1)
        )

        # Hardware Table
        hw_table = Table(box=None, expand=True)
        hw_table.add_column("Komponent", style="bold white")
        hw_table.add_column("Wartość", style="green")

        hw_table.add_row("Procesor (i5-14600KF)", f"{telemetry.get('cpu_percent', 0)}%")
        hw_table.add_row("Pamięć RAM", f"{telemetry.get('ram_used_gb', 0)} / {telemetry.get('ram_total_gb', 0)} GB")
        hw_table.add_row("Karta GPU (RX 9060 XT)", f"{telemetry.get('gpu_busy_percent', 0)}%")
        hw_table.add_row("Pamięć VRAM (ROCm 7.2)", f"{telemetry.get('vram_used_gb', 0)} / {telemetry.get('vram_total_gb', 0)} GB")

        layout["hardware"].update(Panel(hw_table, title="[bold green]Telemetria Sprzętowa[/bold green]"))

        # Stages Table
        st_table = Table(box=None, expand=True)
        st_table.add_column("Faza", style="bold white")
        st_table.add_column("Postęp", style="magenta")

        st_table.add_row("Krok 0: Ekstrakcja EPUB", stage_progress.get("step0", "Zakończono"))
        st_table.add_row("Faza 1: Reżyseria (LLM)", stage_progress.get("stage1", "Oczekuje"))
        st_table.add_row("Faza 2: Synteza (TTS)", stage_progress.get("stage2", "Oczekuje"))
        st_table.add_row("Faza 3: Dźwięki (SFX/BGM)", stage_progress.get("stage3", "Oczekuje"))
        st_table.add_row("Faza 4: Miks DSP", stage_progress.get("stage4", "Oczekuje"))

        layout["stages"].update(Panel(st_table, title="[bold magenta]Stan Pipeline'u[/bold magenta]"))

        # 3. Storage Info
        storage_text = (
            f"Baza SQLite: {telemetry.get('db_size_mb', 0)} MB  │  "
            f"Stemy Audio: {telemetry.get('stems_size_mb', 0)} MB ({telemetry.get('stems_count', 0)} plików)"
        )
        layout["storage"].update(Panel(Text(storage_text, style="bright_blue"), title="[bold blue]Dane i Pliki[/bold blue]"))

        # 4. Logs Window
        logs_text = "\n".join(recent_logs[-8:]) if recent_logs else "Oczekiwanie na pierwsze zdarzenia..."
        layout["logs"].update(Panel(Text(logs_text, style="white"), title="[bold yellow]Dziennik Zdarzeń (Live Log)[/bold yellow]"))

        return layout
