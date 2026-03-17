# Быстрый запуск пайплайнов

## 1) Извлечение OCR + LLM (doc, page, table, row)
1. Запустить ноутбук `DeepSeek_OCR_Patch_PDF2MD.ipynb`.
   - Что делает: запускает DeepSeek OCR с патчем.
   - Вход: папка с PDF.
   - Выход: папка `ocr_md` с `.md` файлами.
2. Запустить скрипты:
   - `md_attrs_service/parse_doc.py` -> из `ocr_md/*.md` в `*_doc.json`
   - `md_attrs_service/parse_page.py` -> из `ocr_md/*.md` в `*_page.json`
   - `md_attrs_service/parse_table.py` -> из `ocr_md/*.md` в `*_table.json`
   - `md_attrs_service/parse_row.py` -> из `ocr_md/*.md` в `*_row.json`
   - Результаты сохраняются рядом с исходными `.md` в `ocr_md`.
3. Для `md_attrs_service/parse_*.py` нужен ключ:
   - `OPENROUTER_API_KEY` (берётся из переменной окружения).

## 2) Извлечение с fine-tuned Qwen
1. (Опционально) Трейн: запустить `Visual_IE_Train.ipynb`.
   - Вход: `qwen_dataset.zip` (распаковывается в наборы `qwen_dataset_*`).
   - Выход: чекпоинт адаптера в `output_sft/.../checkpoint-*` (пример в папке: `checkpoint-50`).
2. Инференс: запустить `Visual_IE_Inference.ipynb`.
   - Вход: папка с PDF (в проекте это `OL_dataset`) + чекпоинт адаптера.
   - Выход: для каждого PDF в `OL_dataset` создаётся `*_qwen-8b-ft.json`.

## 3) Оценка результатов извлечения
1. Запустить `Attrs_Match_Pipeline.ipynb`.
   - Вход: папка `gt_old` с файлами стратегий (`*_doc.json`, `*_page.json`, `*_table.json`, `*_row.json`, `*_qwen-8b-ft.json`) и разметкой `*_gt.json`.
   - Выход:
     - детальные результаты по каждой паре в `gt_old_debug/*.json`,
     - сводная таблица в `gt_old_debug/agg_results.xlsx`,
     - сводная таблица в `gt_old_debug/agg_results.csv`.

## Docker

### 1) PDF -> Markdown (DeepSeek OCR + patch)
- Папка сервиса: `./pdf2md_service`
- Что делает: конвертирует папку с `*.pdf` в структуру `ocr_md` (`page_XXX.md` + общий `<doc>.md`).

```bash
cd ./pdf2md_service
docker build -t pdf2md-deepseek:latest .
docker run --rm --gpus all \
  -v /abs/path/to/pdfs:/workspace/input \
  -v /abs/path/to/md:/workspace/output \
  -v /abs/path/to/raw_ocr:/workspace/raw_ocr \
  pdf2md-deepseek:latest
```

Volumes:
- `/workspace/input` — входные `*.pdf` (исходные документы).
- `/workspace/output` — итоговые `*.md` (папка `ocr_md`-структуры).
- `/workspace/raw_ocr` — промежуточные `.mmd/.pdf` артефакты OCR (генерируются сервисом, опционально для отладки).

Например:
```bash
docker run --rm --gpus all \
  -v /home/ubuntu/pdf2md_service/pdfs:/workspace/input \
  -v /home/ubuntu/pdf2md_service/md:/workspace/output \
  pdf2md-deepseek:latest
```

### 2) Attrs extraction из готового `ocr_md` (doc/page/row/table)
- Папка сервиса: `./md_attrs_service`
- Что делает: запускает `md_attrs_service/parse_doc.py`, `md_attrs_service/parse_page.py`, `md_attrs_service/parse_row.py`, `md_attrs_service/parse_table.py` (по одной стратегии или все сразу).

```bash
cd ./md_attrs_service
docker build -t md-attrs:latest .
docker run --rm \
  -e OPENROUTER_API_KEY=sk-or-v1-... \
  -v /abs/path/to/ocr_md:/workspace/ocr_md \
  md-attrs:latest \
  --strategy all \
  --openrouter_model deepseek/deepseek-v3.2
```

Volumes:
- `/workspace/ocr_md` — готовые markdown-файлы после OCR (`<doc>/<doc>.md`, `page_XXX.md`), обычно это выход сервиса `pdf2md_service`.
- JSON-результаты (`*_doc.json`, `*_page.json`, `*_row.json`, `*_table.json`) пишутся в ту же папку рядом с `.md`.

Например:
```bash
docker run --rm \
  -e OPENROUTER_API_KEY=sk-or-v1-... \
  -v /home/ubuntu/pdf2md_service/ocr_md:/workspace/ocr_md \
  md-attrs:latest \
  --strategy doc \
  --openrouter_model deepseek/deepseek-v3.2
```

### 3) Attrs extraction напрямую из PDF (Qwen FT, Visual IE)
- Папка сервиса: `./pdf_visual_ie_service`
- Что делает: рендерит PDF-страницы, запускает `swift infer` с LoRA-адаптером, пишет `<doc>_qwen-8b-ft.json`.

```bash
cd ./pdf_visual_ie_service
docker build -t pdf-visual-ie:gpu .
docker run --rm --gpus all \
  -v /abs/path/to/pdfs:/workspace/input \
  -v /abs/path/to/output:/workspace/output \
  -v /abs/path/to/adapter:/workspace/adapter \
  -v /abs/path/to/work:/workspace/work \
  pdf-visual-ie:gpu
```

Volumes:
- `/workspace/input` — входные `*.pdf` (исходные документы).
- `/workspace/output` — итоговые `*_qwen-8b-ft.json` по каждому PDF.
- `/workspace/adapter` — LoRA checkpoint (выход `Visual_IE_Train.ipynb`, например `checkpoint-50`).
- `/workspace/work` — промежуточные изображения страниц, `dataset.jsonl`, raw `infer` результаты.

Например:
```bash
docker run --rm --gpus all \
  -v /home/ubuntu/pdf2md_service/pdfs:/workspace/input \
  -v ./output:/workspace/output \
  -v ./checkpoint-50:/workspace/adapter \
  -v ./work:/workspace/work \
  pdf-visual-ie:gpu \
  --num_beams 1 \
  --vllm_max_model_len 6000
```

### 4) Оценка качества извлечения (embedding + reranker)
- Папка сервиса: `./attrs_eval_service`
- Что делает: сравнивает все стратегии с `*_gt.json`, сохраняет детальные `json` + `agg_results.xlsx/csv`.

```bash
cd ./attrs_eval_service
docker build -t attrs-eval:gpu .
docker run --rm --gpus all \
  -v /abs/path/to/gt_old:/workspace/input \
  -v /abs/path/to/eval_out:/workspace/output \
  attrs-eval:gpu \
  --out_xlsx agg_results.xlsx \
  --match_threshold 0.9
```

Volumes:
- `/workspace/input` — папка со всеми стратегиями и эталонами `*_gt.json` (обычно `gt_old`).
- `/workspace/output` — детальные сравнения по парам (`*.json`) и сводные `agg_results.xlsx/csv`.

Например:
```bash
docker run --rm --gpus all \
  -v /home/ubuntu/pdf2md_service/gt_old:/workspace/input \
  -v /home/ubuntu/pdf2md_service/eval_out:/workspace/output \
  attrs-eval:gpu \
  --debug_subdir gt_old_debug \
  --out_xlsx agg_results.xlsx \
  --match_threshold 0.9
```
