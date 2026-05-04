from __future__ import annotations

from pathlib import Path


GEMMA_COMPETITION_SLUG = "gemma-4-good-hackathon"


def download_competition_files() -> Path:
    import kagglehub

    path = kagglehub.competition_download(GEMMA_COMPETITION_SLUG)
    return Path(path)
