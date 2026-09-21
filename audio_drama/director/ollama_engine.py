import json
import subprocess
import requests
from typing import Dict, Any, Optional
from audio_drama.core.models import ScreenplayResponse

class OllamaEngine:
    def __init__(
        self,
        model_name: str = "hf.co/unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL",
        base_url: str = "http://127.0.0.1:11434"
    ):
        self.model_name = model_name
        self.base_url = base_url

    def generate(self, prompt: str, temperature: float = 0.3) -> ScreenplayResponse:
        url = f"{self.base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": ScreenplayResponse.model_json_schema(),
            "options": {
                "temperature": temperature,
                "num_ctx": 16384
            }
        }
        
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        raw_content = data.get("message", {}).get("content", "").strip()
        parsed = json.loads(raw_content)
        return ScreenplayResponse.model_validate(parsed)

    def unload(self) -> None:
        """Zwalnia model z pamięci VRAM."""
        try:
            subprocess.run(["ollama", "stop", self.model_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
