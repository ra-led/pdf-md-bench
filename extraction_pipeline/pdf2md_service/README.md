# PDF -> MD service (DeepSeek OCR + patch)

Сервис контейнеризован и делает только одно: конвертирует все `*.pdf` из входной папки в Markdown.

## Структура результата
Для каждого `file.pdf` создаётся папка:
- `<output_dir>/file/page_000.md`, `page_001.md`, ...
- `<output_dir>/file/file.md` (общий markdown по документу)

## Сборка
```bash
cd pdf2md_service
docker build -t pdf2md-deepseek:latest .
```

## Запуск
```bash
docker run --rm --gpus all \
  -v /abs/path/to/pdfs:/workspace/input \
  -v /abs/path/to/md:/workspace/output \
  -v /abs/path/to/raw_ocr:/workspace/raw_ocr \
  pdf2md-deepseek:latest
```

Например:
```bash
docker run --rm --gpus all \
  -v /home/ubuntu/pdf2md_service/pdfs:/workspace/input \
  -v /home/ubuntu/pdf2md_service/md:/workspace/output \
  pdf2md-deepseek:latest

```

`/workspace/raw_ocr` можно не монтировать, но полезно для отладки промежуточных `.mmd`.

## Параметры
- `--input_dir` (опционально): по умолчанию `/workspace/input`
- `--output_dir` (опционально): по умолчанию `/workspace/output`
- `--raw_ocr_dir` (опционально): по умолчанию `/workspace/raw_ocr`
- `--model_path` (опционально): модель, по умолчанию `deepseek-ai/DeepSeek-OCR`
