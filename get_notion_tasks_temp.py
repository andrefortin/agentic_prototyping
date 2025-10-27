import os
import sys
import json
import requests
import re
from datetime import datetime, timedelta

# Hardcoded arguments
database_id = "29524ba1e67b80e3bf15000b74cc2405"
statuses = ["Not started", "HIL Review"]
limit = 3

try:
    token = os.environ['NOTION_API_KEY']
except KeyError:
    print(json.dumps([{"error": "NOTION_API_KEY not found"}]))
    sys.exit(1)

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
    "Notion-Version": "2025-09-03"
}

print(f"Debug: Database ID: {database_id}", file=sys.stderr)
query_url = f"https://api.notion.com/v1/databases/{database_id}/query"
print(f"Debug: Query URL: {query_url}", file=sys.stderr)
# Query database
filter_properties = {
    "or": [
        {
            "property": "Status",
            "select": {
                "equals": status
            }
        } for status in statuses
    ]
}
sorts = [
    {
        "property": "Created time",
        "direction": "ascending"
    }
]
body = {
    "filter": filter_properties,
    "sorts": sorts,
    "page_size": limit
}

response = requests.post(
    f"https://api.notion.com/v1/databases/{database_id}/query",
    headers=headers,
    json=body
)

if response.status_code != 200:
    print(json.dumps([{"error": f"Database query failed: {response.status_code} {response.text}"}]))
    sys.exit(1)

data = response.json()
results = data.get('results', [])

eligible_tasks = []
now = datetime.utcnow()

def extract_all_text(blocks):
    texts = []
    block_types_with_rich_text = [
        'paragraph', 'heading_1', 'heading_2', 'heading_3',
        'bulleted_list_item', 'numbered_list_item', 'code', 'quote',
        'callout', 'toggle', 'synced_block'
    ]
    for block in blocks:
        block_type = block.get('type', '')
        if block_type in block_types_with_rich_text:
            rich_text = block.get(block_type, {}).get('rich_text', [])
            for rt in rich_text:
                if 'text' in rt and 'content' in rt['text']:
                    texts.append(rt['text']['content'])
        # Recurse on children if present
        if 'children' in block:
            texts.extend(extract_all_text(block['children']))
    return ' '.join(texts).strip()

for page in results:
    page_id = page['id']
    # Check last_edited_time
    last_edited_time_str = page.get('last_edited_time', '')
    if last_edited_time_str:
        last_edited_time = datetime.fromisoformat(last_edited_time_str.replace('Z', '+00:00'))
        if (now - last_edited_time).total_seconds() < 30:
            continue

    properties = page.get('properties', {})
    status_obj = properties.get('Status', {}).get('select', {})
    status = status_obj.get('name', '')
    if status == "In progress":
        continue

    # Title from Name property
    title_obj = properties.get('Name', {}).get('title', [{}])[0]
    title = title_obj.get('text', {}).get('content', '') if title_obj.get('text') else ''

    # Retrieve blocks
    blocks_resp = requests.post(
        f"https://api.notion.com/v1/blocks/{page_id}/children",
        headers=headers,
        json={"page_size": 100}
    )
    if blocks_resp.status_code != 200:
        print(f"Debug: Failed to retrieve blocks for {page_id}", file=sys.stderr)
        continue

    blocks_data = blocks_resp.json()
    blocks = blocks_data.get('results', [])
    if not blocks:
        continue

    # Last block text
    last_block = blocks[-1]
    last_text = ''
    block_type = last_block.get('type', '')
    block_types_with_rich_text = [
        'paragraph', 'heading_1', 'heading_2', 'heading_3',
        'bulleted_list_item', 'numbered_list_item', 'code', 'quote'
    ]
    if block_type in block_types_with_rich_text:
        rich_text = last_block.get(block_type, {}).get('rich_text', [])
        for rt in rich_text:
            if 'text' in rt and 'content' in rt['text']:
                last_text += rt['text']['content'] + ' '
    last_text = last_text.strip()

    if not (last_text.startswith('execute') or last_text.startswith('continue -')):
        continue

    # Determine trigger
    execution_trigger = 'execute' if last_text.startswith('execute') else 'continue'

    # Full content for tags
    full_prompt = extract_all_text(blocks)

    # Task prompt
    if execution_trigger == 'execute':
        task_prompt = full_prompt
    else:
        # For continue, last block text after 'continue - '
        if last_text.startswith('continue -'):
            task_prompt = last_text[10:].strip()  # 10 for 'continue -'
        else:
            task_prompt = last_text

    # Tags
    tags = {}
    tag_pattern = r'\{\{([^:]+):\s*([^\}]+)\}\}'
    matches = re.findall(tag_pattern, full_prompt)
    for key, val in matches:
        tags[key.strip().lower()] = val.strip()

    # Content blocks
    content_blocks = blocks  # Full structure

    task = {
        "page_id": page_id,
        "title": title,
        "status": status,
        "content_blocks": content_blocks,
        "tags": tags,
        "execution_trigger": execution_trigger,
        "task_prompt": task_prompt
    }
    eligible_tasks.append(task)
    print(f"Debug: Added task {title} with trigger {execution_trigger}", file=sys.stderr)

print(json.dumps(eligible_tasks, default=str))  # default=str for datetime if any
