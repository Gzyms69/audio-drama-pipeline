import time
import json
import subprocess
import requests
from pathlib import Path
from typing import Optional, Dict, Any
from audio_drama.core.models import ScreenplayResponse
from audio_drama.director.gbnf_grammar import AUDIO_SCREENPLAY_GBNF

class LlamaServerEngine:
    def __init__(
        self,
        server_bin: str = "llama-server",
        host: str = "127.0.0.1",
        port: int = 8080
    ):
        self.server_bin = server_bin
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.proc: Optional[subprocess.Popen] = None

    def start_server(
        self,
        model_path: str | Path,
        context_size: int = 16384,
        gpu_layers: int = 99,
        kv_cache_quant: str = "q8_0"
    ) -> None:
        cmd = [
            self.server_bin,
            "-m", str(model_path),
            "-c", str(context_size),
            "-ngl", str(gpu_layers),
            "--cache-type-k", kv_cache_quant,
            "--cache-type-v", kv_cache_quant,
            "--chat-template-kwargs", '{"enable_thinking":false}',
            "--port", str(self.port),
            "--host", self.host
        ]
        
        # Uruchomienie serwera w tle
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._wait_until_ready(timeout=60)

    def _wait_until_ready(self, timeout: int = 60) -> None:
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                resp = requests.get(f"{self.base_url}/health", timeout=2)
                if resp.status_code in (200, 503): # 503 gdy ładuje model, 200 gdy gotowy
                    if resp.status_code == 200:
                        return
            except requests.RequestException:
                pass
            time.sleep(1)
        raise TimeoutError("llama-server nie uruchomił się w zadanym czasie.")

    def generate(self, prompt: str, grammar: Optional[str] = AUDIO_SCREENPLAY_GBNF, temperature: float = 0.3) -> ScreenplayResponse:
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "temperature": temperature,
            "n_predict": 4096,
            "stream": False
        }
        if grammar:
            payload["grammar"] = grammar

        resp = requests.post(f"{self.base_url}/completion", json=payload, timeout=300)
        resp.raise_for_status()
        result_json = resp.json()
        content = result_json.get("content", "").strip()
        
        # Parsowanie do modelu Pydantic
        parsed = json.loads(content)
        return ScreenplayResponse.model_validate(parsed)

    def stop(self) -> None:
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
