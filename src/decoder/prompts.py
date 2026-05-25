import json
import re
SYSTEM_PROMPT = 'Classify survey open-ended answers into all applicable codebook labels.\nUse only listed numeric codes.\nReturn only JSON. Do not add explanations.'

def format_questions(questions):
    return '\n'.join((f'- {question}' for question in questions))

def format_labeled_examples(examples):
    blocks = []
    for idx, example in enumerate(examples, start=1):
        labels = ', '.join(example['labels']) if example['labels'] else 'none'
        blocks.append(f'Example {idx}\nText: {example['text']}\nCodes: {labels}')
    return '\n\n'.join(blocks)

def prompt_item_id(case_id):
    return f'ID_{case_id}'

def normalize_item_id(item_id):
    text = str(item_id).strip()
    return text[3:] if text.startswith('ID_') else text

def format_batch_texts(items):
    return '\n'.join((f'item_id={prompt_item_id(str(item['case_id']))} | text={item['text']}' for item in items))

def response_contract(label_cols):
    return f'Output JSON object with one key, predictions. predictions must be a list with one object per input text: {{"item_id": "ID_<caseID>", "codes": ["<code>", "..."]}}. Every input item_id must appear exactly once. Treat item_id as an identifier, not as a code. Allowed codes: {', '.join(label_cols)}. Use an empty list when no code applies.'

def build_messages(*, mode, questions, codebook_text, label_cols, batch_items, examples=None):
    parts = ['Survey question context:', format_questions(questions), '', 'Codebook:', codebook_text, '']
    if mode == 'zero_shot_codebook':
        parts.append('Classify the following answers using the codebook.')
    elif mode == 'retrieval_few_shot_codebook':
        parts.extend(['Use these labeled examples as guidance. They are semantically close to the current batch.', format_labeled_examples(examples or []), '', 'Now classify the following answers using the codebook.'])
    elif mode == 'in_context_100_codebook':
        parts.extend(['Use these labeled training examples as guidance.', format_labeled_examples(examples or []), '', 'Now classify the following answers using the codebook.'])
    parts.extend(['', 'Answers:', format_batch_texts(batch_items), '', response_contract(label_cols)])
    return [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': '\n'.join(parts)}]

def parse_prediction_json(content, expected_ids, allowed_codes, require_all=True):
    text = _remove_control_chars(_extract_json_object(_strip_code_fences(content)))
    try:
        payload = json.loads(text)
        rows = payload['predictions']
        result = {}
        for row in rows:
            row_id = row.get('item_id', row.get('caseID'))
            case_id = normalize_item_id(row_id)
            codes = [str(code) for code in row.get('codes', []) if str(code) in allowed_codes]
            result[case_id] = sorted(set(codes), key=lambda code: int(code))
    except json.JSONDecodeError:
        result = _regex_parse_predictions(text, allowed_codes)
    return result

def _strip_code_fences(content):
    text = content.strip()
    if text.startswith('```'):
        text = text.strip('`').strip()
        if text.lower().startswith('json'):
            text = text[4:].strip()
    return text

def _extract_json_object(text):
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start:end + 1]

def _remove_control_chars(text):
    return ''.join((ch for ch in text if ch in '\n\r\t' or ord(ch) >= 32))

def _regex_parse_predictions(text, allowed_codes):
    result = {}
    object_blocks = re.findall('\\{[^{}]*?(?:caseID|item_id)[^{}]*?\\}', text, flags=re.IGNORECASE | re.DOTALL)
    for block in object_blocks:
        case_match = re.search('"?(?:caseID|item_id)"?\\s*:\\s*"?([^",\\n\\r}]+)"?', block, flags=re.IGNORECASE)
        if not case_match:
            continue
        case_id = normalize_item_id(case_match.group(1))
        codes_match = re.search('"?codes"?\\s*:\\s*\\[(.*?)\\]', block, flags=re.IGNORECASE | re.DOTALL)
        if not codes_match:
            result[case_id] = []
            continue
        raw_codes = re.findall('\\d+', codes_match.group(1))
        result[case_id] = sorted({code for code in raw_codes if code in allowed_codes}, key=lambda code: int(code))
    return result
