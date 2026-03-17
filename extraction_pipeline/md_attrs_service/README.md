# MD -> Attrs JSON (Docker)

Только извлечение атрибутов из готовой папки `ocr_md` (без OCR, без PDF).

Поддерживаемые стратегии:
- `doc` -> `*_doc.json`
- `page` -> `*_page.json`
- `row` -> `*_row.json`
- `table` -> `*_table.json`
- `all` -> запускает все четыре стратегии подряд

## Требования
- Docker Engine
- `OPENROUTER_API_KEY`
- Подготовленная папка `ocr_md` со структурой:
  - `ocr_md/<doc_id>/<doc_id>.md`
  - `ocr_md/<doc_id>/page_000.md`, `page_001.md`, ...

## Сборка образа
```bash
cd /Users/yuratomakov/innopolis/pdf2md/results/md_attrs_service
docker build -t md-attrs:latest .
```

## Запуск (все стратегии)
```bash
docker run --rm \
  -e OPENROUTER_API_KEY=sk-or-v1-... \
  -v /abs/path/to/ocr_md:/workspace/ocr_md \
  md-attrs:latest \
  --strategy all \
  --openrouter_model deepseek/deepseek-v3.2
```

## Запуск по отдельной стратегии

### `parse_doc.py`
```bash
docker run --rm \
  -e OPENROUTER_API_KEY=sk-or-v1-... \
  -v /abs/path/to/ocr_md:/workspace/ocr_md \
  md-attrs:latest \
  --strategy doc \
  --openrouter_model deepseek/deepseek-v3.2
```

### `parse_page.py`
```bash
docker run --rm \
  -e OPENROUTER_API_KEY=sk-or-v1-... \
  -v /abs/path/to/ocr_md:/workspace/ocr_md \
  md-attrs:latest \
  --strategy page \
  --openrouter_model deepseek/deepseek-v3.2
```

### `parse_row.py`
```bash
docker run --rm \
  -e OPENROUTER_API_KEY=sk-or-v1-... \
  -v /abs/path/to/ocr_md:/workspace/ocr_md \
  md-attrs:latest \
  --strategy row \
  --openrouter_model deepseek/deepseek-v3.2
```

### `parse_table.py`
```bash
docker run --rm \
  -e OPENROUTER_API_KEY=sk-or-v1-... \
  -v /abs/path/to/ocr_md:/workspace/ocr_md \
  md-attrs:latest \
  --strategy table \
  --openrouter_model deepseek/deepseek-v3.2
```

## Где результаты
JSON создаются рядом с исходными `.md` внутри смонтированной `ocr_md` папки:
- `ocr_md/<doc_id>/<doc_id>_doc.json`
- `ocr_md/<doc_id>/<doc_id>_page.json`
- `ocr_md/<doc_id>/<doc_id>_row.json`
- `ocr_md/<doc_id>/<doc_id>_table.json`
