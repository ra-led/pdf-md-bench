#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from pypdfium2 import PdfDocument


IMAGE_IE_PROMPT_TEMPLATE = """Тебе будет дано изображение страницы  PDF из опросного листа (технического документа на оборудование).
Извлеки спсиок характеристик и требований в JSON формате. Без комментариев, без пояснений, без Markdown.

JSON format:
[
  {
    "name": "<название параметра, которое запрашивает Заказчик>",
    "request_value": "<значение парметра, которое запрашивает Заказчик>"
  },
  ...
]

СТРОГИЕ ПРАВИЛА:

1) Верни ТОЛЬКО JSON. Любой текст вне JSON запрещён.

2) name:
   - Используй формулировку характеристики или требования из документа.
   - Нормализуй переносы строк и пробелы, но НЕ меняй смысл.
   - Если требование оформлено предложением (например: "… должен …"), используй ВЕСЬ текст требования как name, в одну строку.

3) request_value:
   - Для параметров, заданных в таблицах или структурированных блоках, указывай значение, сопоставленное этому параметру в документе.
   - Для декларативных требований, не имеющих явного значения, используй подтверждающее значение "Да".
   - Значения вида:
       "Определяет завод-изготовитель",
       "Уточняется согласно рабочей документации",
       "По таблице …",
       "___", "____"
     считаются валидными значениями и должны сохраняться без изменений.
   - Если параметр содержит несколько подзначений, объединяй их в ОДНУ строку через "; ", сохраняя все числа, единицы измерения и ссылки.
   - Для формы с чек-боксами укажи только выбранные варианты

4) НЕ добавляй атрибуты, которых нет в документе.
5) НЕ изменяй единицы измерения, знаки «−», «+», десятичные разделители.
6) Игнорируй титульный лист, подписи, реквизиты, контактную информацию — извлекай только технические характеристики и требования.
7) Не оставляй несколько требований в одном. Если они оформлены в одной ячейке таблицы или одним абзацем, разбей их в отдельные.
8) Верни все характеристики из текста страниц, даже исли они представленны частично.
9) Игнорируй справочные таблицы в приложениях, со сводными значениями/прогнозами без требований или характеристик относящихся к какой-либо продукции.
"""

IMAGE_FACTOR = 32
MIN_TOKENS = 4
MAX_TOKENS = 2150
SYSTEM_PROMPT_RU = (
    "Ты точно извлекаешь пары ключ-значение из технических документов. "
    "Возвращай только валидный JSON без пояснений."
)
USER_PROMPT = IMAGE_IE_PROMPT_TEMPLATE.strip() + "\n\n<image>"
IMAGE_NAME_RE = re.compile(r"^image_(?P<doc>.+)_p(?P<page>\d+)\.png$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract attrs directly from PDF using Qwen3-VL adapter (Visual_IE_Inference flow)."
    )
    parser.add_argument("--input_pdf_dir", required=True, help="Directory with input PDF files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save *_qwen-8b-ft.json.")
    parser.add_argument("--adapter_checkpoint", required=True, help="Path to adapter checkpoint.")
    parser.add_argument("--work_dir", default="/workspace/work", help="Working dir for dataset/images/raw outputs.")
    parser.add_argument("--base_model", default="Qwen/Qwen3-VL-8B-Instruct", help="Base VL model.")
    parser.add_argument("--scale", type=float, default=3.0, help="PDF render scale.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Inference temperature.")
    parser.add_argument("--num_beams", type=int, default=2, help="Beam count.")
    parser.add_argument("--max_new_tokens", type=int, default=5000, help="Max new tokens for inference.")
    parser.add_argument("--vllm_max_model_len", type=int, default=16000, help="vLLM max model len.")
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization.",
    )
    return parser.parse_args()


def convert_pdf_to_images(pdf_path: Path, scale: float = 3.0):
    pdf_document = PdfDocument(pdf_path.as_posix())
    pages = []
    for page_index in range(len(pdf_document)):
        page = pdf_document[page_index]
        pil_image = page.render(scale=scale).to_pil()
        pages.append(pil_image)
        page.close()
    pdf_document.close()
    return pages


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    import math

    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    import math

    return math.floor(number / factor) * factor


def smart_resize(height: int, width: int, min_pixels=MIN_TOKENS, max_pixels=MAX_TOKENS, factor=IMAGE_FACTOR):
    import math

    max_ratio = 200
    if max(height, width) / min(height, width) > max_ratio:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    min_area = min_pixels * factor * factor
    max_area = max_pixels * factor * factor
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_area:
        beta = math.sqrt((height * width) / max_area)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_area:
        beta = math.sqrt(min_area / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def prepare_dataset(input_pdf_dir: Path, out_dir: Path, scale: float):
    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "dataset.jsonl"

    rows = 0
    with jsonl_path.open("w", encoding="utf-8") as f:
        for pdf_path in sorted(input_pdf_dir.glob("*.pdf")):
            pdf_tag = pdf_path.stem
            pages = convert_pdf_to_images(pdf_path, scale=scale)
            for page_idx, img in enumerate(pages):
                w, h = img.size
                new_h, new_w = smart_resize(height=h, width=w)
                img = img.convert("RGB").resize((new_w, new_h))

                img_name = f"image_{pdf_tag}_p{page_idx}.png"
                img_path = images_out / img_name
                img.save(img_path, format="PNG")

                item = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT_RU},
                        {"role": "user", "content": USER_PROMPT},
                    ],
                    "images": [img_path.as_posix()],
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                rows += 1

    return jsonl_path, rows


def run_swift_infer(args: argparse.Namespace, dataset_path: Path, result_path: Path):
    env = os.environ.copy()
    env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    cmd = [
        "swift",
        "infer",
        "--adapters",
        Path(args.adapter_checkpoint).as_posix(),
        "--model",
        args.base_model,
        "--merge_lora",
        "True",
        "--infer_backend",
        "vllm",
        "--temperature",
        str(args.temperature),
        "--num_beams",
        str(args.num_beams),
        "--torch_dtype",
        "bfloat16",
        "--vllm_gpu_memory_utilization",
        str(args.vllm_gpu_memory_utilization),
        "--vllm_max_model_len",
        str(args.vllm_max_model_len),
        "--val_dataset",
        dataset_path.as_posix(),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--result_path",
        result_path.as_posix(),
        "--use_hf=1",
    ]
    print("[qwen-ie] running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def parse_extracted_payload(response_text: str):
    try:
        data = json.loads(response_text)
        return data if isinstance(data, list) else []
    except Exception:
        pass

    match = re.search(r"\[[\s\S]*\]", response_text)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def read_image_path(images_field):
    if not images_field:
        return None
    first = images_field[0]
    if isinstance(first, str):
        return first
    if isinstance(first, dict):
        return first.get("path")
    return None


def aggregate_results(input_pdf_dir: Path, result_jsonl: Path, output_dir: Path):
    docs = {}
    with result_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            image_path = read_image_path(row.get("images"))
            if not image_path:
                continue

            image_name = Path(image_path).name
            m = IMAGE_NAME_RE.match(image_name)
            if not m:
                continue

            doc_name = m.group("doc")
            page = int(m.group("page"))
            extracted = parse_extracted_payload(str(row.get("response", "")))
            docs.setdefault(doc_name, {})[page] = extracted

    output_dir.mkdir(parents=True, exist_ok=True)
    for pdf_path in sorted(input_pdf_dir.glob("*.pdf")):
        doc_name = pdf_path.stem
        by_page = docs.get(doc_name, {})
        merged = []
        for page in sorted(by_page.keys()):
            merged.extend(by_page[page])
        out_path = output_dir / f"{doc_name}_qwen-8b-ft.json"
        out_path.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")


def main():
    args = parse_args()

    input_pdf_dir = Path(args.input_pdf_dir)
    output_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir)
    adapter_checkpoint = Path(args.adapter_checkpoint)

    if not input_pdf_dir.exists():
        raise FileNotFoundError(f"input_pdf_dir not found: {input_pdf_dir}")
    if not adapter_checkpoint.exists():
        raise FileNotFoundError(f"adapter_checkpoint not found: {adapter_checkpoint}")

    infer_dataset_dir = work_dir / "qwen_infer_dataset"
    infer_dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path, rows = prepare_dataset(input_pdf_dir, infer_dataset_dir, args.scale)
    print(f"[qwen-ie] dataset rows: {rows}")
    print(f"[qwen-ie] dataset path: {dataset_path}")

    raw_result_path = work_dir / "infer_results.jsonl"
    run_swift_infer(args, dataset_path, raw_result_path)

    aggregate_results(input_pdf_dir, raw_result_path, output_dir)
    print(f"[qwen-ie] done, output dir: {output_dir}")


if __name__ == "__main__":
    main()
