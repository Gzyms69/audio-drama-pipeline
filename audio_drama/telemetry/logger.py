import datetime
from pathlib import Path
from typing import List
from collections import deque
import threading

class EventLogger:
    def __init__(self, log_file: str | Path = "pipeline.log", max_buffer_size: int = 15):
        self.log_file = Path(log_file)
        self.max_buffer_size = max_buffer_size
        self._buffer: deque = deque(maxlen=max_buffer_size)
        self._lock = threading.Lock()

    def log(self, category: str, message: str) -> None:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] [{category.upper()}] {message}"
        
        with self._lock:
            self._buffer.append(formatted)
            try:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(formatted + "\n")
            except Exception:
                pass

    def get_recent_logs(self) -> List[str]:
        with self._lock:
            return list(self._buffer)
