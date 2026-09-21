import time
from typing import Optional

class EtaCalculator:
    def __init__(self, total_items: int, alpha: float = 0.2):
        self.total_items = max(total_items, 1)
        self.completed_items = 0
        self.alpha = alpha  # Współczynnik wygładzania EMA
        self.ema_duration: Optional[float] = None
        self.start_time = time.time()

    def item_completed(self, duration_seconds: float) -> None:
        self.completed_items += 1
        if self.ema_duration is None:
            self.ema_duration = duration_seconds
        else:
            self.ema_duration = self.alpha * duration_seconds + (1.0 - self.alpha) * self.ema_duration

    def get_eta_seconds(self) -> float:
        remaining = max(self.total_items - self.completed_items, 0)
        if remaining == 0 or self.ema_duration is None:
            return 0.0
        return remaining * self.ema_duration

    def get_elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @staticmethod
    def format_time(seconds: float) -> str:
        s = int(round(seconds))
        hours = s // 3600
        minutes = (s % 3600) // 60
        secs = s % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def get_formatted_eta(self) -> str:
        return self.format_time(self.get_eta_seconds())

    def get_formatted_elapsed(self) -> str:
        return self.format_time(self.get_elapsed_seconds())
