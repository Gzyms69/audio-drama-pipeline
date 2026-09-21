import os
from pathlib import Path
from typing import Dict, Any, Optional
import psutil

def _find_amd_sysfs_path() -> Optional[Path]:
    for card_dir in Path("/sys/class/drm").glob("card*"):
        dev_dir = card_dir / "device"
        if (dev_dir / "gpu_busy_percent").exists():
            return dev_dir
    return None

def get_hardware_telemetry(db_path: Optional[str | Path] = None, stems_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    # 1. CPU & RAM
    cpu_pct = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    ram_used_gb = vm.used / (1024 ** 3)
    ram_total_gb = vm.total / (1024 ** 3)

    # 2. AMD GPU & VRAM (sysfs)
    gpu_busy = 0
    vram_used_gb = 0.0
    vram_total_gb = 16.0

    sysfs = _find_amd_sysfs_path()
    if sysfs:
        try:
            busy_file = sysfs / "gpu_busy_percent"
            if busy_file.exists():
                gpu_busy = int(busy_file.read_text().strip())
            
            used_file = sysfs / "mem_info_vram_used"
            if used_file.exists():
                vram_used_gb = int(used_file.read_text().strip()) / (1024 ** 3)

            total_file = sysfs / "mem_info_vram_total"
            if total_file.exists():
                vram_total_gb = int(total_file.read_text().strip()) / (1024 ** 3)
        except Exception:
            pass

    # 3. Rozmiar bazy danych i stemów
    db_size_mb = 0.0
    if db_path:
        p = Path(db_path)
        if p.exists():
            db_size_mb = p.stat().st_size / (1024 ** 2)

    stems_size_mb = 0.0
    stems_count = 0
    if stems_dir:
        s_dir = Path(stems_dir)
        if s_dir.exists():
            for f in s_dir.glob("**/*"):
                if f.is_file():
                    stems_count += 1
                    stems_size_mb += f.stat().st_size / (1024 ** 2)

    return {
        "cpu_percent": round(cpu_pct, 1),
        "ram_used_gb": round(ram_used_gb, 2),
        "ram_total_gb": round(ram_total_gb, 1),
        "gpu_busy_percent": gpu_busy,
        "vram_used_gb": round(vram_used_gb, 2),
        "vram_total_gb": round(vram_total_gb, 1),
        "db_size_mb": round(db_size_mb, 2),
        "stems_size_mb": round(stems_size_mb, 2),
        "stems_count": stems_count
    }
