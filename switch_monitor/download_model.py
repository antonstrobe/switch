from __future__ import annotations

import argparse
from pathlib import Path

from .local_gemma import ensure_llama_cpp_server, ensure_local_gemma_assets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the bundled local Gemma model.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root. Defaults to the current directory.",
    )
    args = parser.parse_args(argv)

    server_path = ensure_llama_cpp_server(args.root, progress_callback=print)
    assets = ensure_local_gemma_assets(args.root, progress_callback=print)
    print(f"Runtime: {server_path}")
    print(f"Model:  {assets.model_path}")
    print(f"MMProj: {assets.mmproj_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
