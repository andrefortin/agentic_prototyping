#!/usr/bin/env python3
import os
import sys
import json
import requests

# Manual .env parsing for NOTION_INTERNAL_INTEGRATION_SECRET
def load_env_manual(filename='.env'):
    if not os.path.exists(filename):
        return
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith('NOTION_INTERNAL_INTEGRATION_SECRET='):
            key_value = line.split('=', 1)[1]
            if key_value.startswith('"') and key_value.endswith('"'):
                key_value = key_value[1:-1]
            os.environ['MANUAL_NOTION_KEY'] = key_value
            return

load_env_manual()

# Set database ID
primary_db_id = '29524ba1-e67b-80cb9eb9e56f6af7e9c0'
fallback_db_id = '29524ba1-e67b-80e3-bf15-000b74cc2405'

API_KEY = os.getenv('NOTION_INTERNAL_INTEGRATION_SECRET') or os.getenv('MANUAL_NOTION_KEY')

if not API_KEY:
    print(json.dumps({'error': 'NOTION_INTERNAL_INTEGRATION_SECRET not set'}), file=sys.stderr)
    sys.exit(1)

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Notion-Version": "2025-09-03",
    "Content-Type": "application/json"
}

BASE_URL = "https://api.notion.com/v1"

def query_database(db_id):
    query_url = f"{BASE_URL}/data_sources/{db_id}/query"
    payload = {
        "page_size": 100
    }
    response = requests.post(query_url, headers=HEADERS, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data.get('results', [])
    else:
        raise requests.HTTPError(f"Query failed: {response.status_code} - {response.text}")

# Try primary DB, fallback if fails
try:
    all_pages = query_database(primary_db_id)
    database_used = primary_db_id
    print(f"Successfully queried database {primary_db_id}")
except Exception as e:
    print(f"Primary database query failed: {str(e)}")
    print(f"Falling back to {fallback_db_id}")
    all_pages = query_database(fallback_db_id)
    database_used = fallback_db_id

all_ids = [page['id'] for page in all_pages]
print(f"Total pages found: {len(all_ids)} (expected 12, using DB {database_used})")

accessible_ids = []
inaccessible = {}  # id: status

for page_id in all_ids:
    retrieve_url = f"{BASE_URL}/pages/{page_id}"
    resp = requests.get(retrieve_url, headers=HEADERS)
    if resp.status_code == 200:
        accessible_ids.append(page_id)
    else:
        inaccessible[page_id] = f"initial retrieve failed with {resp.status_code}"
        # Attempt to archive
        update_payload = {"archived": True}
        patch_resp = requests.patch(retrieve_url, headers=HEADERS, json=update_payload)
        if patch_resp.status_code == 200:
            inaccessible[page_id] = "successfully archived"
        else:
            inaccessible[page_id] += f" (patch {patch_resp.status_code}: {patch_resp.text[:100]} - likely already removed)"

print("\n1. List of all IDs:")
print(json.dumps(all_ids, indent=2))

print("\n2. Accessible IDs (keep):")
print(json.dumps(accessible_ids, indent=2))

print("\n3. Inaccessible IDs and removal status:")
for page_id, status in inaccessible.items():
    print(f"  - {page_id}: {status}")

print(f"\n4. Final clean count (accessible pages): {len(accessible_ids)}")
