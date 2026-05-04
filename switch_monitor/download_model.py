from __future__ import annotations

import kagglehub


def main() -> int:
    # Download latest version
    path = kagglehub.competition_download("gemma-4-good-hackathon")

    print("Path to competition files:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
