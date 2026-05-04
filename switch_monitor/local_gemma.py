from __future__ import annotations

import os
import shutil
from urllib.parse import urlparse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


DEFAULT_LOCAL_GEMMA_REPO = os.environ.get(
    "SWITCH_GEMMA_REPO",
    "bartowski/google_gemma-4-E2B-it-GGUF",
)
DEFAULT_LOCAL_GEMMA_MODEL_FILE = os.environ.get(
    "SWITCH_GEMMA_MODEL_FILE",
    "google_gemma-4-E2B-it-Q4_K_M.gguf",
)
DEFAULT_LOCAL_GEMMA_MMPROJ_FILE = os.environ.get(
    "SWITCH_GEMMA_MMPROJ_FILE",
    "mmproj-google_gemma-4-E2B-it-f16.gguf",
)
DEFAULT_LOCAL_GEMMA_DIR = os.environ.get(
    "SWITCH_GEMMA_DIR",
    "gemma-4-e2b-it-gguf",
)

LOCAL_GEMMA_MODEL_ID = "gemma-4-e2b-it-q4_k_m"
LOCAL_GEMMA_DISPLAY_NAME = "Gemma 4 E2B IT Q4_K_M (local GGUF)"

KNOWN_FILE_SIZES = {
    "google_gemma-4-E2B-it-Q4_K_M.gguf": 3_462_677_760,
    "mmproj-google_gemma-4-E2B-it-f16.gguf": 985_653_760,
}
DOWNLOAD_BUFFER_BYTES = 1_000_000_000

LLAMA_CPP_RELEASE_TAG = os.environ.get("SWITCH_LLAMA_CPP_RELEASE_TAG", "b8902")
LLAMA_CPP_BACKEND = os.environ.get("SWITCH_LLAMA_CPP_BACKEND", "vulkan").strip().lower()
LLAMA_CPP_ZIP_NAMES = {
    "cpu": f"llama-{LLAMA_CPP_RELEASE_TAG}-bin-win-cpu-x64.zip",
    "vulkan": f"llama-{LLAMA_CPP_RELEASE_TAG}-bin-win-vulkan-x64.zip",
    "cuda": f"llama-{LLAMA_CPP_RELEASE_TAG}-bin-win-cuda-12.4-x64.zip",
}
LLAMA_CPP_DIR_NAMES = {
    "cpu": f"llama.cpp-{LLAMA_CPP_RELEASE_TAG}-cpu-x64",
    "vulkan": f"llama.cpp-{LLAMA_CPP_RELEASE_TAG}-vulkan-x64",
    "cuda": f"llama.cpp-{LLAMA_CPP_RELEASE_TAG}-cuda-12.4-x64",
}
LLAMA_CPP_ZIP_NAME = LLAMA_CPP_ZIP_NAMES.get(LLAMA_CPP_BACKEND, LLAMA_CPP_ZIP_NAMES["vulkan"])
LLAMA_CPP_ZIP_URL = os.environ.get(
    "SWITCH_LLAMA_CPP_ZIP_URL",
    f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_RELEASE_TAG}/{LLAMA_CPP_ZIP_NAME}",
)
LLAMA_CPP_DIR = os.environ.get(
    "SWITCH_LLAMA_CPP_DIR",
    LLAMA_CPP_DIR_NAMES.get(LLAMA_CPP_BACKEND, LLAMA_CPP_DIR_NAMES["vulkan"]),
)
LLAMA_CPP_SERVER_EXE = "llama-server.exe"


@dataclass(frozen=True)
class LocalGemmaAssets:
    repo_id: str
    model_file: str
    mmproj_file: str
    model_dir: Path
    model_path: Path
    mmproj_path: Path

    @property
    def installed(self) -> bool:
        return self.model_path.exists() and self.mmproj_path.exists()


def get_local_gemma_assets(project_root: Path) -> LocalGemmaAssets:
    model_dir = project_root / "models" / DEFAULT_LOCAL_GEMMA_DIR
    return LocalGemmaAssets(
        repo_id=DEFAULT_LOCAL_GEMMA_REPO,
        model_file=DEFAULT_LOCAL_GEMMA_MODEL_FILE,
        mmproj_file=DEFAULT_LOCAL_GEMMA_MMPROJ_FILE,
        model_dir=model_dir,
        model_path=model_dir / DEFAULT_LOCAL_GEMMA_MODEL_FILE,
        mmproj_path=model_dir / DEFAULT_LOCAL_GEMMA_MMPROJ_FILE,
    )


def ensure_llama_cpp_server(
    project_root: Path,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    local_exe = project_root / "bin" / LLAMA_CPP_DIR / LLAMA_CPP_SERVER_EXE
    if local_exe.exists():
        return local_exe

    path_exe = shutil.which(LLAMA_CPP_SERVER_EXE)
    if path_exe:
        return Path(path_exe)

    archive_dir = project_root / "bin" / "_downloads"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_name = Path(urlparse(LLAMA_CPP_ZIP_URL).path).name or "llama-cpp.zip"
    archive_path = archive_dir / archive_name

    if not archive_path.exists():
        if progress_callback:
            progress_callback(f"Downloading llama.cpp runtime from {LLAMA_CPP_ZIP_URL}")
        urllib.request.urlretrieve(LLAMA_CPP_ZIP_URL, archive_path)

    target_dir = project_root / "bin" / LLAMA_CPP_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(target_dir)

    if not local_exe.exists():
        raise RuntimeError(f"llama.cpp server was not found after extraction: {local_exe}")
    return local_exe


def ensure_local_gemma_assets(
    project_root: Path,
    progress_callback: Callable[[str], None] | None = None,
) -> LocalGemmaAssets:
    assets = get_local_gemma_assets(project_root)
    if assets.installed:
        return assets

    _ensure_download_space(assets)
    assets.model_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is not installed. Run: python -m pip install -r requirements.txt"
        ) from error

    for filename in (assets.model_file, assets.mmproj_file):
        target = assets.model_dir / filename
        if target.exists():
            continue
        if progress_callback:
            progress_callback(f"Downloading {filename} from {assets.repo_id}")
        _download_hf_file(hf_hub_download, assets.repo_id, filename, assets.model_dir)

    assets = get_local_gemma_assets(project_root)
    if not assets.installed:
        raise RuntimeError(f"Gemma files were not downloaded into {assets.model_dir}")
    return assets


def _download_hf_file(hf_hub_download, repo_id: str, filename: str, local_dir: Path) -> None:
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_dir),
    )


def _ensure_download_space(assets: LocalGemmaAssets) -> None:
    missing_bytes = 0
    for filename, path in ((assets.model_file, assets.model_path), (assets.mmproj_file, assets.mmproj_path)):
        if not path.exists():
            missing_bytes += KNOWN_FILE_SIZES.get(filename, 0)

    if missing_bytes <= 0:
        return

    free_bytes = _free_bytes_for(assets.model_dir)
    required = missing_bytes + DOWNLOAD_BUFFER_BYTES
    if free_bytes is not None and free_bytes < required:
        free_gb = free_bytes / (1024**3)
        required_gb = required / (1024**3)
        raise RuntimeError(
            f"Not enough free disk space for local Gemma. Free: {free_gb:.1f} GiB, "
            f"required: {required_gb:.1f} GiB."
        )


def _free_bytes_for(path: Path) -> int | None:
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return None
