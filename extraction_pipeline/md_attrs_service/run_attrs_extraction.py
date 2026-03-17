#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPTS = {
    "doc": "parse_doc.py",
    "page": "parse_page.py",
    "row": "parse_row.py",
    "table": "parse_table.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run attrs extraction from prepared OCR markdown directory.")
    parser.add_argument(
        "--strategy",
        choices=["doc", "page", "row", "table", "all"],
        default="all",
        help="Extraction strategy to run.",
    )
    parser.add_argument(
        "--md_dir",
        required=True,
        help="Path to OCR markdown root (contains doc folders with page_*.md and <doc>.md).",
    )
    parser.add_argument(
        "--openrouter_model",
        default="deepseek/deepseek-v3.2",
        help="OpenRouter model name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    md_dir = Path(args.md_dir)
    if not md_dir.exists():
        raise FileNotFoundError(f"md dir does not exist: {md_dir}")

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required")

    env = os.environ.copy()
    env["OCR_MD_DIR"] = md_dir.as_posix()
    env["OPENROUTER_MODEL"] = args.openrouter_model

    strategies = list(SCRIPTS.keys()) if args.strategy == "all" else [args.strategy]
    for strategy in strategies:
        script_name = SCRIPTS[strategy]
        script_path = Path("/app") / script_name
        print(f"[attrs] run strategy={strategy} script={script_name}")
        subprocess.run([sys.executable, script_path.as_posix()], check=True, env=env, cwd="/app")

    print("[attrs] completed")


if __name__ == "__main__":
    main()
