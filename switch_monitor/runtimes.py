from __future__ import annotations

import json
import atexit
import os
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests

from .competition_assets import download_competition_files


DEFAULT_TIMEOUT = 600
OFFICIAL_GEMMA_MODEL_ID = "google/gemma-4-E2B-it"
OFFICIAL_GEMMA_DISPLAY_NAME = "Google Gemma 4 E2B IT (official)"
LOCAL_GEMMA_MAX_TOKENS = int(os.environ.get("SWITCH_GEMMA_MAX_TOKENS", "1024"))
LOCAL_GEMMA_CONTEXT_SIZE = int(os.environ.get("SWITCH_GEMMA_CTX_SIZE", "4096"))
LOCAL_GEMMA_SAFE_GPU_LAYERS = int(os.environ.get("SWITCH_GEMMA_GPU_LAYERS", "15"))
LOCAL_GEMMA_LOW_VRAM_GPU_LAYERS = int(os.environ.get("SWITCH_GEMMA_LOW_VRAM_GPU_LAYERS", "8"))
LMSTUDIO_TARGET_CONTEXT = 8192
DEFAULT_GPU_MODE = "max"
LMSTUDIO_IDENTIFIER = "switch-vision"
OLLAMA_MAX_GPU_LAYERS = 999
LMSTUDIO_GPU_RUNTIME_MARKERS = ("nvidia-cuda", "cuda", "vulkan", "rocm", "metal")


def subprocess_creationflags() -> int:
    return subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0


def format_seconds(seconds: float | int) -> str:
    value = float(seconds)
    if abs(value - round(value)) < 0.005:
        return f"{int(round(value))} сек"
    return f"{value:.2f} сек"


def format_file_size(path: Path) -> str:
    size = path.stat().st_size
    if size >= 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024 * 1024):.2f} GB"
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"


def get_nvidia_gpu_status() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=subprocess_creationflags(),
        )
    except Exception:
        return "GPU NVIDIA: не обнаружена или nvidia-smi недоступен"

    line = next((item.strip() for item in completed.stdout.splitlines() if item.strip()), "")
    parts = [item.strip() for item in line.split(",")]
    if len(parts) >= 3:
        return f"GPU NVIDIA: {parts[0]}, VRAM {parts[2]}/{parts[1]} MB"
    if line:
        return f"GPU NVIDIA: {line}"
    return "GPU NVIDIA: не обнаружена или nvidia-smi недоступен"


def get_nvidia_total_memory_mib() -> int | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=subprocess_creationflags(),
        )
    except Exception:
        return None
    line = next((item.strip() for item in completed.stdout.splitlines() if item.strip()), "")
    try:
        return int(float(line))
    except Exception:
        return None


def find_nvidia_vulkan_icd() -> str | None:
    explicit = os.environ.get("SWITCH_VK_ICD_FILENAMES")
    if explicit and Path(explicit).exists():
        return explicit

    driver_store = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "DriverStore" / "FileRepository"
    try:
        matches = sorted(driver_store.glob("nv*/nv-vk64.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    except Exception:
        return None
    return str(matches[0]) if matches else None


def normalize_gpu_mode(gpu_mode: str | None) -> str:
    mode = (gpu_mode or DEFAULT_GPU_MODE).strip().lower()
    aliases = {
        "": DEFAULT_GPU_MODE,
        "gpu": "max",
        "cuda": "max",
        "vulkan": "max",
        "cpu": "off",
        "none": "off",
    }
    mode = aliases.get(mode, mode)
    if mode in {"max", "auto", "off"}:
        return mode
    try:
        ratio = float(mode)
    except ValueError:
        return DEFAULT_GPU_MODE
    if ratio <= 0:
        return "off"
    if ratio >= 1:
        return "max"
    return str(ratio)


@dataclass
class RuntimeModel:
    runtime: str
    model_id: str
    display_name: str
    details: str = ""


class RuntimeErrorBase(RuntimeError):
    pass


class BaseRuntime:
    name = "base"

    def detect_models(self) -> list[RuntimeModel]:
        raise NotImplementedError

    def prepare(self, model_id: str, gpu_mode: str = DEFAULT_GPU_MODE) -> str:
        return model_id

    def analyze(
        self,
        prepared_model_id: str,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        timeout=None,
    ) -> str:
        raise NotImplementedError


class OfficialGemmaRuntime(BaseRuntime):
    name = "official-gemma"

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or Path.cwd()
        self.model_id = os.environ.get("SWITCH_GEMMA_MODEL_ID", OFFICIAL_GEMMA_MODEL_ID).strip()
        self.pipeline = None
        self.competition_files_path: Path | None = None
        self.progress_callback: Callable[[str], None] | None = None

    def detect_models(self) -> list[RuntimeModel]:
        return [
            RuntimeModel(
                self.name,
                self.model_id,
                OFFICIAL_GEMMA_DISPLAY_NAME,
                "Official Google Hugging Face model loaded with Transformers",
            )
        ]

    def prepare(self, model_id: str, gpu_mode: str = DEFAULT_GPU_MODE) -> str:
        if model_id != self.model_id:
            raise RuntimeErrorBase(f"Unknown official Gemma model: {model_id}")
        if not model_id.startswith("google/"):
            raise RuntimeErrorBase("Only official Google model repositories are allowed. Use google/gemma-4-E2B-it.")
        if self.pipeline is not None:
            return model_id
        self._emit_progress("Downloading Kaggle competition files: gemma-4-good-hackathon")
        try:
            self.competition_files_path = download_competition_files()
        except Exception as error:
            raise RuntimeErrorBase(f"Kaggle competition files could not be downloaded: {error}") from error
        self._emit_progress(f"Kaggle competition files: {self.competition_files_path}")
        self._emit_progress(f"Loading official Google model: {model_id}")
        try:
            from transformers import pipeline
        except ImportError as error:
            raise RuntimeErrorBase(
                "Transformers is not installed. Run: python -m pip install -r requirements.txt"
            ) from error
        try:
            self.pipeline = pipeline(
                "image-text-to-text",
                model=model_id,
                device_map="auto",
                torch_dtype="auto",
            )
        except TypeError:
            self.pipeline = pipeline(
                "image-text-to-text",
                model=model_id,
                device_map="auto",
            )
        except Exception as error:
            raise RuntimeErrorBase(f"Official Gemma setup failed: {error}") from error
        self._emit_progress("Official Google Gemma is ready")
        return model_id

    def analyze(
        self,
        prepared_model_id: str,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        timeout=None,
    ) -> str:
        if self.pipeline is None:
            self.prepare(prepared_model_id)
        assert self.pipeline is not None
        from io import BytesIO

        from PIL import Image

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        prompt = (
            f"{system_prompt}\n\n{user_prompt}\n\n"
            "Return only valid JSON. Do not add Markdown fences or commentary."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            result = self.pipeline(text=messages, max_new_tokens=LOCAL_GEMMA_MAX_TOKENS, return_full_text=False)
        except TypeError:
            result = self.pipeline(messages, max_new_tokens=LOCAL_GEMMA_MAX_TOKENS)
        return self._extract_generated_text(result)

    def set_progress_callback(self, callback: Callable[[str], None] | None) -> None:
        self.progress_callback = callback

    def _emit_progress(self, message: str) -> None:
        if not self.progress_callback:
            return
        try:
            self.progress_callback(message)
        except Exception:
            pass

    def _extract_generated_text(self, result) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, list) and result:
            return self._extract_generated_text(result[0])
        if isinstance(result, dict):
            value = result.get("generated_text") or result.get("text") or result.get("content")
            if isinstance(value, str):
                return value
            if isinstance(value, list) and value:
                last = value[-1]
                if isinstance(last, dict):
                    content = last.get("content")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        return "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)
                return str(last)
        return str(result)


class OllamaRuntime(BaseRuntime):
    name = "ollama"
    base_url = "http://127.0.0.1:11434"

    def __init__(self) -> None:
        self.gpu_mode = DEFAULT_GPU_MODE
        self.num_gpu = self._num_gpu_for_mode(self.gpu_mode)

    def detect_models(self) -> list[RuntimeModel]:
        models: list[RuntimeModel] = []
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            payload = response.json()
            for item in payload.get("models", []):
                name = item.get("name", "")
                if "gemma" in name.lower() and self._supports_vision(name):
                    models.append(RuntimeModel(self.name, name, name, "Detected via Ollama API"))
            if models:
                return models
        except Exception:
            pass

        try:
            completed = subprocess.run(
                ["ollama", "list"],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
                creationflags=subprocess_creationflags(),
            )
            for line in completed.stdout.splitlines()[1:]:
                line = line.strip()
                if not line:
                    continue
                name = line.split()[0]
                if "gemma" in name.lower() and self._supports_vision(name):
                    models.append(RuntimeModel(self.name, name, name, "Detected via ollama list"))
        except Exception:
            return []
        return models

    def _supports_vision(self, model_id: str) -> bool:
        try:
            response = requests.post(f"{self.base_url}/api/show", json={"model": model_id}, timeout=15)
            response.raise_for_status()
            capabilities = response.json().get("capabilities", [])
            return "vision" in capabilities
        except Exception:
            return False

    def prepare(self, model_id: str, gpu_mode: str = DEFAULT_GPU_MODE) -> str:
        self.gpu_mode = normalize_gpu_mode(gpu_mode)
        self.num_gpu = self._num_gpu_for_mode(self.gpu_mode)
        return model_id

    def analyze(
        self,
        prepared_model_id: str,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        timeout=None,
    ) -> str:
        import base64

        options = {"temperature": 0.1, "num_predict": 768}
        if self.num_gpu is not None:
            options["num_gpu"] = self.num_gpu
        payload = {
            "model": prepared_model_id,
            "stream": False,
            "format": "json",
            "keep_alive": "10m",
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [base64.b64encode(image_bytes).decode("ascii")],
                },
            ],
            "options": options,
        }
        response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=timeout or DEFAULT_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")

    def _num_gpu_for_mode(self, gpu_mode: str) -> int | None:
        mode = normalize_gpu_mode(gpu_mode)
        if mode == "auto":
            return None
        if mode == "off":
            return 0
        return OLLAMA_MAX_GPU_LAYERS


class LMStudioRuntime(BaseRuntime):
    name = "lmstudio"
    base_url = "http://127.0.0.1:1234/v1"
    rest_url = "http://127.0.0.1:1234/api/v1"

    def __init__(self) -> None:
        self.install_dir = Path.home() / "AppData" / "Local" / "Programs" / "LM Studio"
        self.models_root = Path.home() / ".lmstudio" / "models"

    def detect_models(self) -> list[RuntimeModel]:
        models: list[RuntimeModel] = []
        try:
            completed = subprocess.run(
                ["lms", "ls", "--json"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
                creationflags=subprocess_creationflags(),
            )
            payload = json.loads(completed.stdout)
            for item in payload:
                if item.get("type") != "llm":
                    continue
                if "gemma" not in str(item.get("modelKey", "")).lower():
                    continue
                if not item.get("vision", False):
                    continue
                models.append(
                    RuntimeModel(
                        self.name,
                        item["modelKey"],
                        item.get("displayName", item["modelKey"]),
                        f"Detected via lms ls: {item.get('path', '')}",
                    )
                )
        except Exception:
            pass
        return models

    def prepare(self, model_id: str, gpu_mode: str = DEFAULT_GPU_MODE) -> str:
        gpu_mode = normalize_gpu_mode(gpu_mode)
        self._ensure_api()
        if gpu_mode != "auto":
            if gpu_mode != "off":
                self._select_gpu_runtime()
            self._unload_identifier(LMSTUDIO_IDENTIFIER)
            self._load_model(model_id, gpu_mode)
            return LMSTUDIO_IDENTIFIER

        loaded = self._get_loaded_instances()
        matching = [item for item in loaded if item.get("modelKey") == model_id]
        suitable = [item for item in matching if int(item.get("contextLength", 0) or 0) >= LMSTUDIO_TARGET_CONTEXT]
        if suitable:
            suitable.sort(key=lambda item: int(item.get("contextLength", 0) or 0), reverse=True)
            return suitable[0]["identifier"]

        try:
            subprocess.run(
                [
                    "lms",
                    "load",
                    model_id,
                    "--identifier",
                    LMSTUDIO_IDENTIFIER,
                    "--context-length",
                    str(LMSTUDIO_TARGET_CONTEXT),
                    "--ttl",
                    "900",
                    "-y",
                ],
                capture_output=True,
                text=True,
                timeout=180,
                check=True,
                creationflags=subprocess_creationflags(),
            )
            return LMSTUDIO_IDENTIFIER
        except Exception:
            loaded = self._get_loaded_instances()
            suitable = [
                item
                for item in loaded
                if item.get("modelKey") == model_id and int(item.get("contextLength", 0) or 0) >= LMSTUDIO_TARGET_CONTEXT
            ]
            if suitable:
                suitable.sort(key=lambda item: int(item.get("contextLength", 0) or 0), reverse=True)
                return suitable[0]["identifier"]
            raise RuntimeErrorBase(
                "LM Studio сервер доступен, но не удалось подготовить Gemma с достаточным контекстом. Откройте LM Studio и загрузите модель заново."
            )

    def analyze(
        self,
        prepared_model_id: str,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes,
        timeout=None,
    ) -> str:
        import base64

        data_url = f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('ascii')}"
        payload = {
            "model": prepared_model_id,
            "system_prompt": system_prompt,
            "temperature": 0.1,
            "max_output_tokens": 768,
            "stream": False,
            "store": False,
            "input": [
                {"type": "text", "content": user_prompt},
                {"type": "image", "data_url": data_url},
            ],
        }
        response = requests.post(f"{self.rest_url}/chat", json=payload, timeout=timeout or (10, 360))
        response.raise_for_status()
        data = response.json()
        output = data.get("output", [])
        if not output:
            raise RuntimeErrorBase("LM Studio вернул пустой ответ.")
        return output[0].get("content", "")

    def _load_model(self, model_id: str, gpu_mode: str) -> None:
        command = [
            "lms",
            "load",
            model_id,
            "--identifier",
            LMSTUDIO_IDENTIFIER,
            "--context-length",
            str(LMSTUDIO_TARGET_CONTEXT),
            "--ttl",
            "900",
            "-y",
        ]
        if gpu_mode != "auto":
            command[3:3] = ["--gpu", gpu_mode]
        try:
            subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=180,
                check=True,
                creationflags=subprocess_creationflags(),
            )
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or error.stdout or "").strip()
            if detail:
                detail = f" Detail: {detail}"
            raise RuntimeErrorBase(
                f"LM Studio could not load the model with GPU mode '{gpu_mode}'.{detail}"
            ) from error

    def _unload_identifier(self, identifier: str) -> None:
        loaded = self._get_loaded_instances()
        if not any(item.get("identifier") == identifier for item in loaded):
            return
        try:
            subprocess.run(
                ["lms", "unload", identifier],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
                creationflags=subprocess_creationflags(),
            )
        except Exception:
            pass

    def _select_gpu_runtime(self) -> None:
        try:
            completed = subprocess.run(
                ["lms", "runtime", "ls"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
                creationflags=subprocess_creationflags(),
            )
        except Exception as error:
            raise RuntimeErrorBase("LM Studio GPU runtime check failed.") from error

        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        runtime_lines = [line for line in lines if line.startswith("llama.cpp")]
        selected = [line for line in runtime_lines if self._is_selected_runtime_line(line)]
        if selected and self._is_gpu_runtime_line(selected[0]):
            return

        gpu_alias = self._find_gpu_runtime_alias(runtime_lines)
        if not gpu_alias:
            raise RuntimeErrorBase(
                "LM Studio has no GPU runtime selected or installed. Install/select a CUDA or Vulkan runtime in LM Studio."
            )

        try:
            subprocess.run(
                ["lms", "runtime", "select", gpu_alias],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
                creationflags=subprocess_creationflags(),
            )
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or error.stdout or "").strip()
            if detail:
                detail = f" Detail: {detail}"
            raise RuntimeErrorBase(f"LM Studio could not select GPU runtime '{gpu_alias}'.{detail}") from error

    def _find_gpu_runtime_alias(self, runtime_lines: list[str]) -> str | None:
        for marker in LMSTUDIO_GPU_RUNTIME_MARKERS:
            for line in runtime_lines:
                if marker in line.lower():
                    return line.split()[0]
        return None

    def _is_gpu_runtime_line(self, line: str) -> bool:
        lowered = line.lower()
        return any(marker in lowered for marker in LMSTUDIO_GPU_RUNTIME_MARKERS)

    def _is_selected_runtime_line(self, line: str) -> bool:
        return len(line.split()) > 2

    def _ensure_api(self) -> None:
        if self._is_api_ready():
            return
        self._launch_gui()
        for _ in range(2):
            try:
                subprocess.run(
                    ["lms", "server", "start", "--port", "1234"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                    creationflags=subprocess_creationflags(),
                )
            except Exception:
                pass
            for _ in range(10):
                if self._is_api_ready():
                    return
                time.sleep(1)
        if not self._is_api_ready():
            raise RuntimeErrorBase(
                "Не удалось поднять API LM Studio автоматически. Откройте LM Studio и включите Local Server."
            )

    def _launch_gui(self) -> None:
        executable = self.install_dir / "LM Studio.exe"
        if executable.exists():
            try:
                subprocess.Popen([str(executable)], creationflags=subprocess_creationflags())
            except Exception:
                pass

    def _is_api_ready(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            return response.ok
        except Exception:
            return False

    def _get_loaded_models(self) -> list[str]:
        response = requests.get(f"{self.base_url}/models", timeout=10)
        response.raise_for_status()
        payload = response.json()
        return [item.get("id", "") for item in payload.get("data", []) if item.get("id")]

    def _get_loaded_instances(self) -> list[dict]:
        try:
            completed = subprocess.run(
                ["lms", "ps", "--json"],
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
                creationflags=subprocess_creationflags(),
            )
            payload = json.loads(completed.stdout)
            return [item for item in payload if item.get("type") == "llm" and item.get("identifier")]
        except Exception:
            return []


class RuntimeRegistry:
    def __init__(self, project_root: Path | None = None) -> None:
        project_root = project_root or Path.cwd()
        self._runtimes: dict[str, BaseRuntime] = {
            OfficialGemmaRuntime.name: OfficialGemmaRuntime(project_root),
        }

    def detect_all_models(self) -> list[RuntimeModel]:
        models: list[RuntimeModel] = []
        for runtime in self._runtimes.values():
            try:
                models.extend(runtime.detect_models())
            except Exception:
                continue
        return models

    def get(self, name: str) -> BaseRuntime:
        runtime = self._runtimes.get(name)
        if not runtime:
            raise RuntimeErrorBase(f"Unknown runtime: {name}")
        return runtime
