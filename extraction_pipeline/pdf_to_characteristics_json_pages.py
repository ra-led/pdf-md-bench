#!/usr/bin/env python3
"""End-to-end pipeline (pages window): PDF -> DeepSeek OCR det.mmd -> page-window blocks -> extracted JSON."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Callable

from ocr_runner import run_ocr_with_deepseek_runner
from openrouter_stream import call_openrouter_json_streaming

try:
    from dedupe_repetitions import fairy_tame as _fairy_tame  # type: ignore
except Exception:
    def _fairy_tame(text: str) -> str:
        return text


IE_SYSTEM_PROMPT = "You are a precise information extraction system."

PAGE_IE_PROMPT_TEMPLATE = """You extract structured characteristics from the provided page window fragment.

Input: text merged from several neighboring pages of a technical questionnaire document.
Output: a single VALID JSON array only. No markdown, no comments.

JSON format:
[
  {{
    "name": "<parameter name or full requirement wording>",
    "class": "<top-level section name if known; otherwise infer from page context>",
    "request_value": "<value requested by customer>",
    "islink": <true/false>
  }},
  ...
]

STRICT RULES:
1) Output ONLY valid JSON (array at top-level). No extra text.
2) Use ONLY information present in this page window fragment. Do not use outside context.
3) Preserve units, references, signs, numeric formatting.
4) For requirements without explicit value, use "Да" unless text says otherwise.
5) islink=true if name/request_value references external docs/tables/standards (ГОСТ, СП, ПУЭ, Приложение, "по таблице", "согласно", etc.).
6) Do not invent attributes that are not in this page window.

WINDOW:
<<<WINDOW
{window_text}
WINDOW>>>
"""

SPLIT_CHECK_PROMPT_TEMPLATE = """Decide whether two characteristics are actually one requirement split across neighboring windows.

Return STRICT JSON object only.

Previous window last characteristic:
{prev_item}

Next window first characteristic:
{next_item}

Rules:
1) Return result=\"one splitted\" if second item clearly continues/completes the first.
2) Return result=\"two different\" if they are independent rows/requirements.
"""

JOIN_SPLIT_PROMPT_TEMPLATE = """Merge two characteristics that are confirmed to be one split requirement.

Return STRICT JSON object only with fields:
- name
- class
- request_value
- islink

Item A:
{prev_item}

Item B:
{next_item}

Rules:
1) Build one coherent merged requirement while preserving meaning and key numbers/units/references.
2) class should be the most appropriate of the two contexts.
3) islink=true if either input has link/reference.
"""

IE_RESPONSE_SCHEMA = {
    "name": "ie_extraction_list",
    "strict": True,
    "schema": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "class": {"type": "string"},
                "request_value": {"type": "string"},
                "islink": {"type": "boolean"},
            },
            "required": ["name", "class", "request_value", "islink"],
            "additionalProperties": False,
        },
    },
}

SPLIT_DECISION_SCHEMA = {
    "name": "split_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "result": {"type": "string", "enum": ["one splitted", "two different"]},
        },
        "required": ["result"],
        "additionalProperties": False,
    },
}

MERGED_ITEM_SCHEMA = {
    "name": "merged_item",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "class": {"type": "string"},
            "request_value": {"type": "string"},
            "islink": {"type": "boolean"},
        },
        "required": ["name", "class", "request_value", "islink"],
        "additionalProperties": False,
    },
}

FENCE_LINE_RE = re.compile(r'^[\'\"]?```.*$', re.MULTILINE)
PAGE_SPLIT_PATTERN = re.compile(r"(?:\r?\n)*<--- Page Split --->(?:\r?\n)*")
TAG_PATTERN = re.compile(
    r"<\|ref\|>(?P<ref>.+?)<\|/ref\|><\|det\|>\s*\[\[(?P<bbox>[^\]]+)\]\]\s*<\|/det\|>",
    flags=re.DOTALL,
)

def clean_input_text(text: str) -> str:
    text = FENCE_LINE_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_from_page_window(
    *,
    window_text: str,
    api_key: str,
    model: str,
    timeout_s: int,
    max_retries: int,
    max_tokens: int,
    require_parameters: bool,
) -> list[dict]:
    prompt = PAGE_IE_PROMPT_TEMPLATE.format(window_text=window_text)
    parsed = call_openrouter_json_streaming(
        api_key=api_key,
        model=model,
        system_prompt=IE_SYSTEM_PROMPT,
        user_prompt=prompt,
        schema=IE_RESPONSE_SCHEMA,
        temperature=0.0,
        timeout_s=timeout_s,
        max_retries=max_retries,
        max_tokens=max_tokens,
        require_parameters=require_parameters,
        log_label="[LLM]",
    )
    if not isinstance(parsed, list):
        raise RuntimeError(f"Expected list from window extraction, got {type(parsed).__name__}")
    return parsed


def check_split_between_windows(
    *,
    prev_item: dict,
    next_item: dict,
    api_key: str,
    model: str,
    timeout_s: int,
    max_retries: int,
    max_tokens: int,
    require_parameters: bool,
) -> str:
    prompt = SPLIT_CHECK_PROMPT_TEMPLATE.format(
        prev_item=json.dumps(prev_item, ensure_ascii=False, indent=2),
        next_item=json.dumps(next_item, ensure_ascii=False, indent=2),
    )
    parsed = call_openrouter_json_streaming(
        api_key=api_key,
        model=model,
        system_prompt=IE_SYSTEM_PROMPT,
        user_prompt=prompt,
        schema=SPLIT_DECISION_SCHEMA,
        temperature=0.0,
        timeout_s=timeout_s,
        max_retries=max_retries,
        max_tokens=max_tokens,
        require_parameters=require_parameters,
        log_label="[LLM]",
    )
    if not isinstance(parsed, dict) or "result" not in parsed:
        raise RuntimeError("Invalid split check response")
    result = str(parsed["result"]).strip().lower()
    if result not in {"one splitted", "two different"}:
        raise RuntimeError(f"Unexpected split decision: {result}")
    return result


def join_split_characteristics(
    *,
    prev_item: dict,
    next_item: dict,
    api_key: str,
    model: str,
    timeout_s: int,
    max_retries: int,
    max_tokens: int,
    require_parameters: bool,
) -> dict:
    prompt = JOIN_SPLIT_PROMPT_TEMPLATE.format(
        prev_item=json.dumps(prev_item, ensure_ascii=False, indent=2),
        next_item=json.dumps(next_item, ensure_ascii=False, indent=2),
    )
    parsed = call_openrouter_json_streaming(
        api_key=api_key,
        model=model,
        system_prompt=IE_SYSTEM_PROMPT,
        user_prompt=prompt,
        schema=MERGED_ITEM_SCHEMA,
        temperature=0.0,
        timeout_s=timeout_s,
        max_retries=max_retries,
        max_tokens=max_tokens,
        require_parameters=require_parameters,
        log_label="[LLM]",
    )
    if not isinstance(parsed, dict):
        raise RuntimeError("Invalid merge response")
    return parsed


def _normalize_bbox(raw_bbox: str) -> str:
    parts = [value.strip() for value in raw_bbox.replace("[", "").replace("]", "").split(",")]
    return " ".join(filter(None, parts))


def _strip_tags(content: str) -> str:
    def _replacer(match: re.Match[str]) -> str:
        ref_type = match.group("ref").strip().lower()
        bbox = _normalize_bbox(match.group("bbox"))
        if ref_type == "image" and bbox:
            return f'<img data-bbox="{bbox}"></img>\n'
        return ""

    cleaned = TAG_PATTERN.sub(_replacer, content)
    cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines())
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _split_pages(raw_text: str) -> list[str]:
    if not raw_text.strip():
        return []
    segments = PAGE_SPLIT_PATTERN.split(raw_text.strip())
    return segments if segments else []


def extract_pages_from_det(det_file: Path) -> list[str]:
    raw_text = det_file.read_text(encoding="utf-8").replace("\r\n", "\n")
    pages_raw = _split_pages(raw_text) or [raw_text]

    pages: list[str] = []
    for page in pages_raw:
        if not page.strip():
            continue
        cleaned = clean_input_text(_fairy_tame(_strip_tags(page)))
        if cleaned:
            pages.append(cleaned)

    return pages


def build_page_windows(pages: list[str], window_size: int) -> list[tuple[int, int, str]]:
    windows: list[tuple[int, int, str]] = []
    for start in range(0, len(pages), window_size):
        end = min(start + window_size, len(pages))
        merged = "\n\n".join(pages[start:end]).strip()
        windows.append((start, end - 1, merged))
    return windows


def merge_cross_window_splits(
    window_results: list[list[dict]],
    *,
    api_key: str,
    model: str,
    timeout_s: int,
    max_retries: int,
    max_tokens: int,
    require_parameters: bool,
) -> list[list[dict]]:
    for i in range(len(window_results) - 1):
        print(f"[STEP] Boundary check between window {i + 1} and window {i + 2}")
        current = window_results[i]
        nxt = window_results[i + 1]
        if not current or not nxt:
            continue

        decision = check_split_between_windows(
            prev_item=current[-1],
            next_item=nxt[0],
            api_key=api_key,
            model=model,
            timeout_s=timeout_s,
            max_retries=max_retries,
            max_tokens=max_tokens,
            require_parameters=require_parameters,
        )

        if decision == "one splitted":
            print(f"[INFO] Window boundary {i + 1}/{i + 2}: detected split, joining")
            merged_item = join_split_characteristics(
                prev_item=current[-1],
                next_item=nxt[0],
                api_key=api_key,
                model=model,
                timeout_s=timeout_s,
                max_retries=max_retries,
                max_tokens=max_tokens,
                require_parameters=require_parameters,
            )
            current[-1] = merged_item
            del nxt[0]
        else:
            print(f"[INFO] Window boundary {i + 1}/{i + 2}: two different characteristics")

    return window_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, help="Input PDF file path")
    parser.add_argument("--out_json", required=True, help="Output JSON path")

    parser.add_argument("--openrouter_api_key", default=None, help="OpenRouter API key")
    parser.add_argument("--openrouter_model", default="deepseek/deepseek-v3.2")
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--timeout_s", type=int, default=300)
    parser.add_argument("--max_tokens", type=int, default=30000)
    parser.add_argument("--no_require_parameters", action="store_true")

    parser.add_argument(
        "--dpsk_workdir",
        default=os.environ.get("DPSK_WORKDIR"),
        help="Path to DeepSeek-OCR-vllm folder containing run_dpsk_ocr_pdf.py",
    )
    parser.add_argument("--dpsk_run_script", default="run_dpsk_ocr_pdf.py")
    parser.add_argument("--dpsk_model_path", default="deepseek-ai/DeepSeek-OCR")
    parser.add_argument("--python_bin", default=sys.executable)

    parser.add_argument("--det_file", default=None, help="Skip OCR and use existing *_det.mmd file")
    parser.add_argument("--workdir", default=None, help="Intermediate workdir; default is a temp directory")
    parser.add_argument("--keep_intermediate", action="store_true")
    parser.add_argument("--page_window", type=int, default=3, help="Number of pages per extraction window")
    parser.add_argument("--save_pages_dir", default=None, help="Optional directory to save extracted page blocks")

    args = parser.parse_args()
    print("[STEP] Starting pages-window pipeline")

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    api_key = args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Missing OpenRouter API key. Pass --openrouter_api_key or set OPENROUTER_API_KEY")
    if args.page_window < 1:
        raise SystemExit("--page_window must be >= 1")

    if args.workdir:
        base_workdir = Path(args.workdir)
        base_workdir.mkdir(parents=True, exist_ok=True)
        cleanup: Callable[[], None] = lambda: None
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="pdf2md_pages_pipeline_")
        base_workdir = Path(temp_dir.name)
        cleanup = temp_dir.cleanup

    try:
        raw_ocr_root = base_workdir / "raw_ocr"
        print(f"[INFO] Intermediate workdir: {base_workdir}")

        if args.det_file:
            det_file = Path(args.det_file)
            if not det_file.exists():
                raise SystemExit(f"det file not found: {det_file}")
            print(f"[STEP] Using existing det file: {det_file}")
        else:
            if not args.dpsk_workdir:
                raise SystemExit("Pass --dpsk_workdir for OCR run or provide --det_file to skip OCR")
            det_file = run_ocr_with_deepseek_runner(
                pdf_path=pdf_path,
                dpsk_workdir=Path(args.dpsk_workdir),
                raw_ocr_root=raw_ocr_root,
                dpsk_model_path=args.dpsk_model_path,
                python_bin=args.python_bin,
                run_script=args.dpsk_run_script,
            )
            print(f"[INFO] OCR det file: {det_file}")

        print("[STEP] Splitting OCR output by pages")
        pages = extract_pages_from_det(det_file)
        if not pages:
            raise SystemExit("No non-empty pages found in OCR output")
        print(f"[INFO] Non-empty pages found: {len(pages)}")

        windows = build_page_windows(pages, args.page_window)
        print(f"[INFO] Windows to process: {len(windows)} (window size={args.page_window})")
        save_pages_dir = Path(args.save_pages_dir) if args.save_pages_dir else None
        if save_pages_dir is not None:
            save_pages_dir.mkdir(parents=True, exist_ok=True)

        window_results: list[list[dict]] = []
        for w_idx, (start_idx, end_idx, window_text) in enumerate(windows, start=1):
            print(f"[STEP] Extracting window {w_idx}/{len(windows)} (pages {start_idx + 1}-{end_idx + 1})")
            if save_pages_dir is not None:
                window_name = "_".join(str(i + 1) for i in range(start_idx, end_idx + 1))
                (save_pages_dir / f"{window_name}.md").write_text(window_text + "\n", encoding="utf-8")

            items = extract_from_page_window(
                window_text=window_text,
                api_key=api_key,
                model=args.openrouter_model,
                timeout_s=args.timeout_s,
                max_retries=args.max_retries,
                max_tokens=args.max_tokens,
                require_parameters=not args.no_require_parameters,
            )
            print(f"[INFO] Extracted characteristics from window {w_idx}: {len(items)}")
            window_results.append(items)

        print("[STEP] Checking neighboring windows for split characteristics")
        merged_window_results = merge_cross_window_splits(
            window_results,
            api_key=api_key,
            model=args.openrouter_model,
            timeout_s=args.timeout_s,
            max_retries=args.max_retries,
            max_tokens=args.max_tokens,
            require_parameters=not args.no_require_parameters,
        )

        final_items: list[dict] = []
        for items in merged_window_results:
            final_items.extend(items)
        print(f"[INFO] Final characteristics count: {len(final_items)}")

        out_json_path = Path(args.out_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[STEP] Writing JSON output: {out_json_path}")
        out_json_path.write_text(json.dumps(final_items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        print(f"Done. JSON saved to: {out_json_path.resolve()}")
        print(f"Pages processed: {len(pages)}")
        print(f"Windows processed: {len(windows)} (window size={args.page_window})")

    finally:
        if not args.keep_intermediate:
            cleanup()


if __name__ == "__main__":
    main()
