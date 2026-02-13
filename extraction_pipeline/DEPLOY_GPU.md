# Деплой в Docker с NVIDIA GPU

## Требования
- Установлен Docker Engine.
- Настроен доступ к GPU через `nvidia-container-toolkit`.
- Драйвер NVIDIA установлен на хосте (проверка: `nvidia-smi`).
- Задан ключ OpenRouter:

```bash
export OPENROUTER_API_KEY='your_key_here'
```

## Сборка образа

```bash
cd /Users/yuratomakov/innopolis/pdf2md/extraction_pipeline
docker build -t pdf2char:gpu -f ./Dockerfile .
```

## Запуск: обработка документа целиком

```bash
docker run --rm --gpus all \
  -e OPENROUTER_API_KEY \
  -v ./input:/workspace/input \
  -v ./output:/workspace/output \
  -v ./work:/workspace/work \
  pdf2char:gpu \
  --pdf /workspace/input/your.pdf \
  --out_json /workspace/output/your.json \
  --openrouter_model deepseek/deepseek-v3.2 \
  --max_tokens 30000 \
  --dpsk_workdir /opt/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm \
  --workdir /workspace/work \
  --keep_intermediate
```

## Запуск: только таблицы

```bash
docker run --rm --gpus all \
  -e OPENROUTER_API_KEY \
  -e PIPELINE_SCRIPT=pdf_to_characteristics_json_tables.py \
  -v ./input:/workspace/input \
  -v ./output:/workspace/output \
  -v ./work:/workspace/work \
  pdf2char:gpu \
  --pdf /workspace/input/your.pdf \
  --out_json /workspace/output/your.tables.json \
  --openrouter_model deepseek/deepseek-v3.2 \
  --max_tokens 30000 \
  --dpsk_workdir /opt/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm \
  --workdir /workspace/work \
  --keep_intermediate \
  --save_tables_dir /workspace/work/tables
```

## Запуск: постраничная/оконная обработка

```bash
docker run --rm --gpus all \
  -e OPENROUTER_API_KEY \
  -e PIPELINE_SCRIPT=pdf_to_characteristics_json_pages.py \
  -v ./input:/workspace/input \
  -v ./output:/workspace/output \
  -v ./work:/workspace/work \
  pdf2char:gpu \
  --pdf /workspace/input/your.pdf \
  --out_json /workspace/output/your.pages.json \
  --openrouter_model deepseek/deepseek-v3.2 \
  --page_window 3 \
  --max_tokens 30000 \
  --dpsk_workdir /opt/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm \
  --workdir /workspace/work \
  --keep_intermediate \
  --save_pages_dir /workspace/work/pages
```

## Конкретные примеры для файла `3.pdf`

### Обработка документа целиком (DeepSeek v3.2)

```bash
docker run --rm --gpus all \
  -e OPENROUTER_API_KEY \
  -v ./input:/workspace/input \
  -v ./output:/workspace/output \
  -v ./work:/workspace/work \
  pdf2char:gpu \
  --pdf /workspace/input/3.pdf \
  --out_json /workspace/output/3.json \
  --openrouter_model deepseek/deepseek-v3.2 \
  --max_tokens 60000 \
  --dpsk_workdir /opt/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm \
  --workdir /workspace/work \
  --keep_intermediate
```

### Обработка документа постранично (qwen/qwen3-30b-a3b-instruct-2507) окном в 3 страницы

```bash
docker run --rm --gpus all \
  -e OPENROUTER_API_KEY \
  -e PIPELINE_SCRIPT=pdf_to_characteristics_json_pages.py \
  -v ./input:/workspace/input \
  -v ./output:/workspace/output \
  -v ./work:/workspace/work \
  pdf2char:gpu \
  --pdf /workspace/input/3.pdf \
  --out_json /workspace/output/3_pages.json \
  --openrouter_model qwen/qwen3-30b-a3b-instruct-2507 \
  --page_window 3 \
  --max_tokens 30000 \
  --dpsk_workdir /opt/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm \
  --workdir /workspace/work \
  --keep_intermediate \
  --save_pages_dir /workspace/work/pages
```

## Примечания
- Первая сборка образа может идти долго (CUDA + torch + vLLM + зависимости DeepSeek OCR).
- Веса моделей DeepSeek OCR загружаются во время выполнения.
- Для режима `tables` извлечение идёт по таблицам с дополнительной проверкой границ между соседними таблицами.
- Для режима `pages` извлечение идёт по страницам/окнам с дополнительной проверкой границ между соседними окнами.
