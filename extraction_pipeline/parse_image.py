import os
import base64
import requests
import json
from pathlib import Path
from time import time
from pypdfium2 import PdfDocument
from tqdm import tqdm


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def openrouter_call(messages, schema, api_key, openrouter_model, temperature, max_tokens):
    s_time = time()
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": openrouter_model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": schema,
            },
            "plugins": [{"id": "response-healing"}],
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
            "provider": {"require_parameters": True},
        },
    )
    response_wait = time() - s_time
    print("Spent", response_wait, "s")
    return response


def convert_pdf_to_images(pdf_path):
    # Load the PDF document
    pdf_document = PdfDocument(pdf_path)
    # print('PdfDocument length', len(pdf_document))
    # Iterate over each page in the PDF
    pages = []
    for i in range(len(pdf_document)):
        # Get the page
        page = pdf_document[i]
        # Render the page to a PIL image
        pil_image = page.render(scale=3).to_pil()
        pages.append(pil_image)

    return pages


IMAGE_IE_PROMPT_TEMPLATE = """Тебе будет дано изображение страницы  PDF из опросного листа (технического документа на оборудование).
Извлеки спсиок характеристик и требований в JSON формате. Без комментариев, без пояснений, без Markdown.

JSON format:
{{
  "requirements": [
    {{
      "name": "<название параметра, которое запрашивает Заказчик>",
      "request_value": "<значение парметра, которое запрашивает Заказчик>",
    }},
    ...
  ]
}}

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

SPLIT_CHECK_PROMPT_TEMPLATE = """Decide whether two characteristics are actually one requirement split across neighboring pages.

Return STRICT JSON object only.

Previous page last characteristic:
{prev_item}

Next page first characteristic:
{next_item}

Rules:
1) Return {{"result": "one splitted"}} if second item clearly continues/completes the first.
2) Return {{"result": "two different"}} if they are independent rows/requirements.
"""

JOIN_SPLIT_PROMPT_TEMPLATE = """Merge two characteristics that are confirmed to be one split requirement.

Return STRICT JSON object only with fields:
- name
- request_value

Item A:
{prev_item}

Item B:
{next_item}

Rules:
1) Build one coherent merged requirement while preserving meaning and key numbers/units/references.
"""

APPENDIX_START_PROMPT_TEMPLATE = """Ты классификатор страниц технического документа.

Задача: определить, начинается ли на этой странице любой раздел приложений.

Считай, что раздел приложений начался, если на странице есть заголовок/маркер вида:
- "Приложение"
- "Приложение 1" (или другой номер)
- "Приложение А" (или другая буква)
- "Appendix"

Верни СТРОГО JSON без текста вне JSON:
{{
  "result": "appendix_start" | "not_appendix_start",
  "reason": "<короткая причина в 3-12 слов>"
}}

Правила:
- Если есть явный заголовок приложения, верни "appendix_start".
- Если это просто упоминание слова "приложение" внутри текста без заголовка начала раздела, верни "not_appendix_start".
"""


IE_RESPONSE_SCHEMA = {
    "name": "requirements_list",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "requirements": {
                "type": "array",
                "description": "List of extracted requirements from the document.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "request_value": {"type": "string"},
                    },
                    "required": ["name", "request_value"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["requirements"],
        "additionalProperties": False,
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
            "request_value": {"type": "string"},
        },
        "required": ["name", "request_value"],
        "additionalProperties": False,
    },
}

APPENDIX_START_SCHEMA = {
    "name": "appendix_start",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "result": {
                "type": "string",
                "enum": ["appendix_start", "not_appendix_start"],
                "description": "Is this page the beginning of appendices section"
            },
            "reason": {"type": "string"},
        },
        "required": ["result", "reason"],
        "additionalProperties": False,
    },
}

api_key = os.environ["OPENROUTER_API_KEY"]
openrouter_model = "qwen/qwen3-vl-235b-a22b-instruct"
max_reqs_per_page = 70

input_root = Path('./OL_dataset/')
image_path = 'tmp.jpg'
for doc_md_path in input_root.glob("*.pdf"):

    print("Processing for", doc_md_path.name)

    pages = convert_pdf_to_images(doc_md_path)

    # 1) Extract requirements page-by-page
    pages_items = {}
    usage = 0
    for page_idx, page_image in tqdm(list(enumerate(pages))):
        page_image.save(image_path)
        base64_image = encode_image_to_base64(image_path)
        data_url = f"data:image/jpeg;base64,{base64_image}"

        appendix_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": APPENDIX_START_PROMPT_TEMPLATE
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                ]
            }
        ]
        appendix_response = openrouter_call(
            messages=appendix_messages,
            schema=APPENDIX_START_SCHEMA,
            api_key=api_key,
            openrouter_model=openrouter_model,
            temperature=0.0,
            max_tokens=120,
        )
        try:
            appendix_verdict = json.loads(
                appendix_response.json()["choices"][0]["message"]["content"]
            )
            if appendix_verdict["result"] == "appendix_start":
                print("APPENDIX START DETECTED:", appendix_verdict["reason"])
                print("Stop extraction for this document at page index", page_idx)
                break
        except Exception as e:
            print("APPENDIX CHECK ERROR:", e)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": IMAGE_IE_PROMPT_TEMPLATE
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                ]
            }
        ]
        response = openrouter_call(
            messages=messages,
            schema=IE_RESPONSE_SCHEMA,
            api_key=api_key,
            openrouter_model=openrouter_model,
            temperature=0.7,
            max_tokens=7000,
        )
        data = response.json()
        pages_items[page_idx] = json.loads(
            data["choices"][0]["message"]["content"]
        )["requirements"][:max_reqs_per_page]

    # 2) Decide/merge split requirements between neighboring pages
    for page_idx in range(len(pages) - 1):
        if not pages_items.get(page_idx):
            continue
        if not pages_items.get(page_idx + 1):
            continue

        prev_item = pages_items[page_idx][-1]
        next_item = pages_items[page_idx + 1][0]

        print("CHECK FOR SPLIT (page boundary)")
        print(prev_item)
        print("---")
        print(next_item)

        messages = [
            {
                "role": "user",
                "content": SPLIT_CHECK_PROMPT_TEMPLATE.format(
                    prev_item=json.dumps(prev_item, ensure_ascii=False),
                    next_item=json.dumps(next_item, ensure_ascii=False),
                ),
            }
        ]
        response = openrouter_call(
            messages=messages,
            schema=SPLIT_DECISION_SCHEMA,
            api_key=api_key,
            openrouter_model=openrouter_model,
            temperature=0.0,
            max_tokens=100,
        )
        try:
            decision = json.loads(
                response.json()["choices"][0]["message"]["content"]
            )["result"]
        except:
            decision = "two different"

        if decision == "one splitted":
            messages = [
                {
                    "role": "user",
                    "content": JOIN_SPLIT_PROMPT_TEMPLATE.format(
                        prev_item=json.dumps(prev_item, ensure_ascii=False),
                        next_item=json.dumps(next_item, ensure_ascii=False),
                    ),
                }
            ]
            response = openrouter_call(
                messages=messages,
                schema=MERGED_ITEM_SCHEMA,
                api_key=api_key,
                openrouter_model=openrouter_model,
                temperature=0.0,
                max_tokens=1500,
            )
            merged_item = json.loads(response.json()["choices"][0]["message"]["content"])

            print("MERGED")
            print(merged_item)

            # Replace last item of previous page and drop first item of next page
            pages_items[page_idx][-1] = merged_item
            pages_items[page_idx + 1] = pages_items[page_idx + 1][1:]

    # 3) Flatten all pages into one list
    items = []
    for page_idx in range(len(pages)):
        for it in pages_items.get(page_idx, []):
            items.append(it)

    print("Tokens generated", usage)
    print(len(items), "were extracted")

    out_path = doc_md_path.with_suffix("").as_posix() + "_image.json"
    with open(out_path, "w") as f:
        f.write(json.dumps(items, ensure_ascii=False))
