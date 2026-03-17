import os
import requests
import json
from pathlib import Path
from time import time

from tqdm import tqdm


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
            "plugins": [
                {"id": "response-healing"}
            ],
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "provider": {"require_parameters": True}
        },
    )
    response_wait = time() - s_time
    print('Spent', response_wait, 's')
    return response


PAGE_IE_PROMPT_TEMPLATE = """Извлеки характеристики и требования из опросного листа (технического документа на оборудование).

На входе: текст нескольких страниц документа, полученный из PDF (может содержать заголовки, списки, таблицы в HTML, переносы строк).
На выходе: спсиок требований в JSON формате. Без комментариев, без пояснений, без Markdown.

JSON format:
[
  {{
    "name": "<название параметра, которое запрашивает Заказчик>",
    "request_value": "<значение парметра, которое запрашивает Заказчик>",
  }},
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

ТЕКСТ ДОКУМЕНТА:
<<<DOCUMENT

{window_text}

DOCUMENT>>>
"""

# 4) class:
#    - Используй название раздела верхнего уровня, к которому логически относится параметр (например: «КЛИМАТИЧЕСКИЕ УСЛОВИЯ», «ТРЕБОВАНИЯ К ЭЛЕКТРОПРИВОДУ», «СЕРТИФИКАЦИЯ», «ОСНОВНЫЕ ТРЕБОВАНИЯ»).
#    - Если точный раздел неочевиден, используй ближайший предыдущий заголовок раздела.

# 5) islink:
#    - true, если name или request_value содержит ссылку или отсылку к внешнему документу, стандарту, приложению или таблице (ГОСТ, СП, ПУЭ, Приложение, «по таблице …», «согласно …» и т.п.).
#    - false во всех остальных случаях.


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
                        "name": {
                            "type": "string",
                            "description": "Required parameter name or full requirement wording"
                        },
                        "request_value": {
                            "type": "string",
                            "description": (
                                "Required parameter value or \"Да\" "
                                "if the value is not specified "
                                "and the wording of the name simply requires conformity"
                            )
                        },
                        # "class": {
                        #     "type": "string",
                        #     "description": (
                        #         "The name of the section and subsection "
                        #         "in which the requirement is specified"
                        #     )
                        # },
                        # "islink": {
                        #     "type": "boolean",
                        #     "description": (
                        #         "Does requirement contains a reference to external "
                        #         "documents/standards/appendices/tables"
                        #     )
                        #   },
                    },
                    "required": ["name", "request_value"], # "class", "islink"],
                    "additionalProperties": False
                },
                
            },
        },
        "required": ["requirements"],
        "additionalProperties": False
    },
}

api_key = os.environ["OPENROUTER_API_KEY"]
openrouter_model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v3.2")


input_root = Path(os.getenv("OCR_MD_DIR", "./ocr_md/"))
for doc in tqdm(input_root.glob('*')):
    doc_md_path = doc / (doc.stem + '.md')
    if not doc_md_path.exists():
        continue
    
    # if doc_md_path.name not in ['3.md', '5.md']:
    #     continue

    print('Processing for', doc_md_path.name)
    with open(doc_md_path) as f:
        window_text = f.read()

    messages = [
        {
            "role": "user",
            "content": PAGE_IE_PROMPT_TEMPLATE.format(window_text=window_text)
        }
    ]
    response = openrouter_call(
        messages=messages,
        schema=IE_RESPONSE_SCHEMA,
        api_key=api_key,
        openrouter_model=openrouter_model,
        temperature=0.0,
        max_tokens=30000
    )

    data = response.json()
    usage = data["usage"]
    items = json.loads(data["choices"][0]["message"]["content"])["requirements"]
    
    print(len(items), 'were extracted')
    print('Tokens generated', usage['completion_tokens'])
    
    with open(doc_md_path.with_suffix('').as_posix() + '_doc.json', 'w') as f:
        f.write(json.dumps(items, ensure_ascii=False))
