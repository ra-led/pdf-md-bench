#!/usr/bin/env python3
"""
pdf_to_md_openrouter.py

Pipeline:
1) PDF -> images (one PNG per page)
2) OpenRouter VLM -> Markdown (per page)
3) Postprocess <img data-bbox="x1 y1 x2 y2">desc</img>:
   - crop from page image
   - replace with ![desc](./images/...)
4) Concat pages -> one .md per PDF
5) Zip outputs per model

Usage:
  export OPENROUTER_API_KEY="..."
  python pdf_to_md_openrouter.py --pdf_dir ./pdfs --out_dir ./outputs

Optional:
  python pdf_to_md_openrouter.py --pdf_dir ./pdfs --models google/gemma-3-27b-it:free qwen/qwen3-vl-30b-a3b-instruct
"""

import argparse
import base64
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import requests
from PIL import Image
from pypdfium2 import PdfDocument
from dedupe_repetitions import fairy_tame


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = "You are helpful assistant."

INSTRUCTIONS = """Covetr this page image to specific markdown

---

1) Output format

- Output must be Markdown, ordered in natural human reading order.
- Use '#' to represent heading levels ('#', '##', '###', etc). Keep titles and section headers with correct hierarchy.
- Preserve paragraphs as continuous text blocks.
- Use '-' for bulleted lists.
- Use '1.', '2.', '3.' for numbered lists.
- Tables MUST be represented strictly in HTML.
- Preserve original text and number formatting exactly.
- Ignore page headers and footers unless semantically meaningful.

---

2) Universal image region tagging (MANDATORY)

- For EVERY image-like region (drawings, photos, charts, diagrams, stamps, logos):
  - Output exactly:
    <img data-bbox="x1 y1 x2 y2">DESCRIPTION</img>
- "x1 y1 x2 y2" is xmin, ymin, xmax, ymax normalized in range from 0 to 1000
- Use a nearby caption if present; otherwise a short factual description.
- Place <img> tags in reading order.

---

Answer only with extracted content in described format without any comments and additions
"""


DEFAULT_MODELS = [
    # "google/gemma-3-27b-it",
    # "qwen/qwen3-vl-30b-a3b-instruct",
    "z-ai/glm-4.6v",
    "baidu/ernie-4.5-vl-28b-a3b",
    "mistralai/ministral-14b-2512",
    # "google/gemini-3-flash-preview",
]


@dataclass
class PageImage:
    pdf_stem: str
    page_index: int
    path: Path
    width: int
    height: int


def slugify_model(model_id: str) -> str:
    # safe folder name
    s = model_id.strip().lower()
    s = s.replace("/", "__").replace(":", "__")
    s = re.sub(r"[^a-z0-9_\-]+", "_", s)
    return s


def convert_pdf_to_images(pdf_path: Path, images_root: Path, scale: float = 1.0) -> list[PageImage]:
    """Render each PDF page to a PNG using pypdfium2."""
    doc = PdfDocument(str(pdf_path))
    out: list[PageImage] = []

    pdf_stem = pdf_path.stem
    pdf_img_dir = images_root / pdf_stem
    pdf_img_dir.mkdir(parents=True, exist_ok=True)

    for i in range(len(doc)):
        page = doc[i]
        pil_image = page.render(scale=scale).to_pil()
        w, h = pil_image.size
        img_path = pdf_img_dir / f"{i}.png"
        pil_image.save(img_path)
        out.append(PageImage(pdf_stem=pdf_stem, page_index=i, path=img_path, width=w, height=h))

    return out


def image_to_data_url_png(image_path: Path) -> str:
    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def build_openrouter_messages(page_image_data_url: str) -> list[dict]:
    # OpenRouter vision format: content as array of typed parts
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": INSTRUCTIONS},
                {"type": "image_url", "image_url": {"url": page_image_data_url}},
            ],
        },
    ]


def call_openrouter_markdown(
    api_key: str,
    model: str,
    page_image_path: Path,
    timeout_s: int = 120,
    max_retries: int = 5,
    temperature: float = 0.3,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data_url = image_to_data_url_png(page_image_path)
    payload = {
        "model": model,
        "messages": build_openrouter_messages(data_url),
        "temperature": 0.1,
        "frequency_penalty": 0.3,
        "top_p": 0.9,
        "presence_penalty": 0.3,
        "max_completion_tokens": 5000
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout_s)
            if resp.status_code >= 400:
                # try to surface useful OpenRouter error payload
                try:
                    j = resp.json()
                except Exception:
                    j = {"error": resp.text}
                raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {json.dumps(j, ensure_ascii=False)[:2000]}")

            j = resp.json()
            # Standard chat completions shape
            content = j["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                # Some providers might return array parts; join any text
                if isinstance(content, list):
                    content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
                else:
                    content = str(content)
            return content

        except Exception as e:
            last_err = e
            # simple exponential backoff
            sleep_s = min(2 ** (attempt - 1), 20)
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed after {max_retries} retries: {last_err}") from last_err


def clean_markdown_response(response: str) -> str:
    """Strip common wrappers (code fences, etc)."""
    if not response:
        return ""

    s = response.strip()

    # Remove fenced blocks if the whole thing is fenced
    if s.startswith("```markdown"):
        s = s[len("```markdown") :].strip()
    if s.startswith("```"):
        s = s[len("```") :].strip()
    if s.endswith("```"):
        s = s[: -len("```")].strip()

    # Some models may embed their own tokens; keep this conservative.
    return s.strip()


def replace_img_tags_with_cropped_images(
    markdown_content: str,
    page_image_path: Path,
    page_num: int,
    output_path: Path,
    normalize_factor: Optional[float] = 1000.0,
) -> str:
    """
    Replace:
      <img data-bbox="x1 y1 x2 y2">desc</img>
    with:
      ![desc](./images/page_XXX_img_YY.png)
    and save cropped PNGs into output_path/images/.
    """
    output_path = Path(output_path)
    cropped_dir = output_path / "images"
    cropped_dir.mkdir(parents=True, exist_ok=True)

    # matches <img data-bbox="...">...</img> or with single quotes
    img_pattern = re.compile(
        r'<img\s+[^>]*data-bbox\s*=\s*["\']([^"\']+)["\'][^>]*>(.*?)</img>',
        re.IGNORECASE | re.DOTALL,
    )

    img_counter = 0

    def replace_img_tag(match: re.Match) -> str:
        nonlocal img_counter
        bbox_str = match.group(1).strip()
        description = re.sub(r"\s+", " ", match.group(2).strip())

        try:
            parts = re.split(r"[,\s]+", bbox_str)
            vals = [float(p) for p in parts if p]
            if len(vals) != 4:
                return match.group(0)

            x1, y1, x2, y2 = vals

            if not page_image_path.exists():
                return match.group(0)

            with Image.open(page_image_path) as img:
                width, height = img.size

                # If model returned 0..1000, convert to 0..1
                if normalize_factor is not None and max(x1, y1, x2, y2) > 1.0:
                    x1, y1, x2, y2 = [c / normalize_factor for c in (x1, y1, x2, y2)]

                # If coords are 0..1, scale to pixels
                if max(x1, y1, x2, y2) <= 1.0:
                    x1, x2 = x1 * width, x2 * width
                    y1, y2 = y1 * height, y2 * height

                # Clamp and convert
                x1i, y1i = max(0, int(round(x1))), max(0, int(round(y1)))
                x2i, y2i = min(width, int(round(x2))), min(height, int(round(y2)))
                if x2i <= x1i or y2i <= y1i:
                    return match.group(0)

                cropped = img.crop((x1i, y1i, x2i, y2i))

                img_counter += 1
                crop_filename = f"page_{page_num:03d}_img_{img_counter:02d}.png"
                crop_path = cropped_dir / crop_filename
                cropped.save(crop_path)

                rel_path = f"./images/{crop_filename}"
                return f"![{description}]({rel_path})"

        except Exception:
            return match.group(0)

    return img_pattern.sub(replace_img_tag, markdown_content)


def concat_pages_to_doc(pages_md: list[str]) -> str:
    # Keep pages separated (optional)
    return "\n\n".join(pages_md).strip() + "\n"


def zip_dir(src_dir: Path, zip_path_no_ext: Path) -> Path:
    # shutil.make_archive wants base_name without extension
    archive = shutil.make_archive(str(zip_path_no_ext), "zip", root_dir=str(src_dir))
    return Path(archive)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", type=str, required=True, help="Folder containing PDFs")
    ap.add_argument("--out_dir", type=str, default="./outputs", help="Output folder")
    ap.add_argument("--tmp_dir", type=str, default="./_page_images", help="Temp folder for rendered page images")
    ap.add_argument("--scale", type=float, default=1.0, help="PDF render scale (try 1.5 or 2.0 for better OCR)")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help="OpenRouter model IDs")
    ap.add_argument("--sleep_s", type=float, default=0.0, help="Sleep between requests (rate limiting)")
    ap.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    ap.add_argument("--max_retries", type=int, default=5)
    args = ap.parse_args()
    print(args)

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY")
    if not api_key:
        raise SystemExit("Missing OPENROUTER_API_KEY env var.")

    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir)
    tmp_dir = Path(args.tmp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in: {pdf_dir}")

    # 1) Render all PDFs once (shared across models)
    print(f"[1/5] Rendering PDFs -> page images into {tmp_dir} ...")
    all_pages: list[PageImage] = []
    for pdf_path in pdf_paths:
        pages = convert_pdf_to_images(pdf_path, tmp_dir, scale=args.scale)
        all_pages.extend(pages)

    # Index by doc
    pages_by_doc: dict[str, list[PageImage]] = {}
    for p in all_pages:
        pages_by_doc.setdefault(p.pdf_stem, []).append(p)
    for k in pages_by_doc:
        pages_by_doc[k].sort(key=lambda x: x.page_index)

    # 2-5) For each model: infer, postprocess, concat, zip
    for model_id in args.models:
        model_slug = slugify_model(model_id)
        model_out = out_dir / model_slug
        if model_out.exists():
            shutil.rmtree(model_out)
        model_out.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Model: {model_id} -> {model_out} ===")

        for pdf_stem, pages in pages_by_doc.items():
            doc_out = model_out / pdf_stem
            doc_out.mkdir(parents=True, exist_ok=True)

            page_mds: list[str] = []

            for p in pages:
                print(f"  - {pdf_stem} page {p.page_index} ...")
                raw = call_openrouter_markdown(
                    api_key=api_key,
                    model=model_id,
                    page_image_path=p.path,
                    max_retries=args.max_retries,
                    temperature=args.temperature,
                )
                cleaned = clean_markdown_response(raw)

                # 3) postprocess <img> tags -> crops
                processed = replace_img_tags_with_cropped_images(
                    cleaned,
                    page_image_path=p.path,
                    page_num=p.page_index,
                    output_path=doc_out,
                    normalize_factor=1000.0,
                )
                
                # remove repetitions
                # processed = fairy_tame(processed)

                # Optional: save per-page md too
                (doc_out / f"page_{p.page_index:03d}.md").write_text(processed + "\n", encoding="utf-8")
                page_mds.append(processed)

                if args.sleep_s > 0:
                    time.sleep(args.sleep_s)

            # 4) concat pages -> doc
            full_md = concat_pages_to_doc(page_mds)
            (doc_out / f"{pdf_stem}.md").write_text(full_md, encoding="utf-8")

        # 5) zip this model
        zip_path = zip_dir(model_out, out_dir / model_slug)
        print(f"[OK] Zipped: {zip_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
