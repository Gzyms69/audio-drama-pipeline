import time
import pytest
from audio_drama.telemetry.hardware import get_hardware_telemetry
from audio_drama.telemetry.eta import EtaCalculator
from audio_drama.telemetry.logger import EventLogger

def test_hardware_telemetry_fields():
    metrics = get_hardware_telemetry(db_path="project.db", stems_dir="stems")
    assert "cpu_percent" in metrics
    assert "ram_used_gb" in metrics
    assert "ram_total_gb" in metrics
    assert "gpu_busy_percent" in metrics
    assert "vram_used_gb" in metrics
    assert "vram_total_gb" in metrics
    assert "db_size_mb" in metrics

def test_eta_calculator_progression():
    eta_calc = EtaCalculator(total_items=10)
    assert eta_calc.get_eta_seconds() == 0.0
    assert eta_calc.format_time(0.0) == "00:00:00"

    # Symulacja wykonania 2 elementów (po 2.0s każdy)
    eta_calc.item_completed(duration_seconds=2.0)
    eta_calc.item_completed(duration_seconds=2.0)

    # 8 elementów pozostało * 2.0s = ~16s
    rem = eta_calc.get_eta_seconds()
    assert 15.0 <= rem <= 17.0
    formatted = eta_calc.get_formatted_eta()
    assert "00:00:16" in formatted or "00:00:15" in formatted

def test_event_logger_buffer():
    logger = EventLogger(max_buffer_size=5)
    for i in range(10):
        logger.log("SYSTEM", f"Message {i}")

    buf = logger.get_recent_logs()
    assert len(buf) == 5
    assert "Message 9" in buf[-1]
