#!/usr/bin/env python3
import os
import sys
import json
import re
from typing import Dict, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests
import click
import traceback as tb

load_dotenv()

# Define eligibility tokens
ELIGIBILITY_TOKENS = {
    "execute": ["execute", "{{execute}}"],
    "continue": ["continue -", "{{continue -"]
}

def parse_rich_text(rich_text):
    return "".join([t.get("plain_text", "") for t in rich_text])

def extract_tags(content):
    tags = {}
    for match in re.finditer(r"{{([^:]+):\s*([^\}]+)}}", content):
        key, value = match.groups()
        tags[key.strip()] = value.strip()
    return tags

def is_eligible(rich_text, verbose):
    if not rich_text:
        return False, None

    last_item = rich_text[-1]
    last_text = last_item["plain_text"].strip()

    if verbose:
        print(f"VERBOSE: Last rich_text item plain_text: {repr(last_text)}")

    # Check for execute trigger
    for token in ELIGIBILITY_TOKENS["execute"]:
        if last_text == token:
            return True, "execute"

    # Check for continue trigger
    for token in ELIGIBILITY_TOKENS["continue"]:
        if last_text.startswith(token):
            return True, "continue"

    return False, None

def get_task_prompt(rich_text, trigger):
    if not rich_text:
        return ""

    if trigger == "execute":
        # All content except the last trigger item
        return parse_rich_text(rich_text[:-1]).strip()
    elif trigger == "continue":
        last_text = rich_text[-1]["plain_text"].strip()
        # Strip braces if present
        if last_text.startswith("{{") and last_text.endswith("}}"):
            last_text = last_text[2:-2].strip()
        # Extract after "continue -"
        if "continue -" in last_text:
            return last_text.split("continue -", 1)[1].strip()
        return last_text
    return ""

def get_database_schema(datasource_id: str, headers: Dict[str, str], verbose: bool, milestone: bool) -> Optional[str]:
    """Retrieve database and extract data_source_id if available. Calls https://api.notion.com/v1/databases/{database_id}."""
    BASE_URL = "https://api.notion.com/v1"
    schema_url = f"{BASE_URL}/databases/{datasource_id}"
    if verbose or milestone:
        print(f"LOG: Milestone 5 - Calling DB API: {schema_url}")
    if verbose:
        print(f"VERBOSE: Full DB API URL: {schema_url}")
    schema_response = requests.get(schema_url, headers=headers)
    if verbose or milestone:
        print(f"LOG: Milestone 5.1 - DB API response status: {schema_response.status_code}")
    if schema_response.status_code != 200:
        if verbose:
            print(f"VERBOSE: Failed to retrieve database {datasource_id}: {schema_response.status_code} - {schema_response.text}")
        return None
    try:
        schema_data = schema_response.json()
        if verbose:
            print(f"VERBOSE: Database API response keys: {list(schema_data.keys())}")
        # In Notion API, check for data_sources or similar fields
        data_sources = schema_data.get("data_sources", [])
        if verbose:
            print(f"VERBOSE: Data sources for database {datasource_id}: {data_sources}")
        if data_sources:
            ds_id = data_sources[0].get("id")
            if verbose:
                print(f"VERBOSE: Extracted data_source_id: {ds_id}")
            return ds_id
        # Fallback: if no data_sources, assume database_id itself is usable
        if verbose:
            print(f"VERBOSE: No data_sources found in database {datasource_id}; using database_id directly.")
        return datasource_id
    except json.JSONDecodeError:
        if verbose:
            print(f"VERBOSE: Invalid JSON response from database retrieval: {schema_response.text}")
        return None

@click.command()
@click.argument(
    "database_input",
    required=False,
    default=os.getenv("NOTION_AGENTIC_TASK_TABLE_ID", "29524ba1e67b80e3bf15000b74cc2405"),
)
@click.argument("status_filter", required=False, default='["Not started", "HIL Review"]')
@click.argument("limit", required=False, type=int, default=10)
@click.option("--debug", is_flag=True, help="Debug mode: Print fetch details")
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
@click.option("--milestone", is_flag=True, help="Display milestone logs")
def main(database_input: str, status_filter: str, limit: int, debug: bool, verbose: bool, milestone: bool):
    if verbose or milestone:
        print(f"LOG: Milestone 0 - main() started")

    # Extract database_id from URL if provided, otherwise use as-is
    def extract_db_id_from_url(url: str) -> Optional[str]:
        if "notion.so" not in url.lower():
            return None
        # Pattern to match Notion database ID: 32-char alphanumeric after / and before ? or end
        match = re.search(r'/([a-z0-9]{32})(?:\?|$)', url)
        if match:
            return match.group(1)
        return None

    extracted_id = extract_db_id_from_url(database_input)
    datasource_id = extracted_id if extracted_id else database_input
    datasource_id = datasource_id.replace("-", "")
    if len(datasource_id) != 32:
        error_msg = f"Invalid database_id: {datasource_id} (expected exactly 32 alphanumeric chars)"
        if extracted_id:
            error_msg += f" from URL: {database_input}"
        print(json.dumps({"error": error_msg}), file=sys.stderr)
        sys.exit(1)

    API_KEY = os.getenv("NOTION_API_KEY")
    if not API_KEY:
        print("ERROR: NOTION_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    if verbose or milestone:
        print(f"LOG: Milestone 1 - API key loaded")
    if verbose:
        api_key_masked = API_KEY[:4] + "****" + API_KEY[-4:] if API_KEY else "NOT SET"
        print(f"VERBOSE: Original input: '{database_input}'")
        print(f"VERBOSE: Extracted/used database_id: '{datasource_id}' (len={len(datasource_id)})")
        print(f"VERBOSE: API_KEY (masked): {api_key_masked}")
        print(f"VERBOSE: Status filter arg: {status_filter} (raw repr: {repr(status_filter)})")
        print(f"VERBOSE: Limit: {limit}")

    if verbose or milestone:
        print(f"LOG: Milestone 1.5 - Setting up headers")
    HEADERS = {
        "Authorization": f"Bearer {API_KEY}",
        "Notion-Version": "2025-09-03",
        "Content-Type": "application/json",
    }
    if verbose:
        print(f"VERBOSE: Headers keys: {list(HEADERS.keys())}")

    BASE_URL = "https://api.notion.com/v1"
    if verbose or milestone:
        print(f"LOG: Milestone 2 - Headers and BASE_URL ready")

    if verbose or milestone:
        print(f"LOG: Milestone 3 - Parsing status_filter")
    try:
        STATUS_FILTER = json.loads(status_filter)
        if verbose:
            print(f"VERBOSE: Status filter parsed: {STATUS_FILTER}")
    except json.JSONDecodeError as parse_err:
        if verbose:
            print(f"VERBOSE: JSON parse failed ({parse_err}), using default")
        STATUS_FILTER = ["Not started", "HIL Review"]

    if verbose:
        for s in STATUS_FILTER:
            print(f"  - [{s}]")
    if verbose or milestone:
        print(f"LOG: Milestone 4 - Filter ready ({len(STATUS_FILTER)} items)")

    # Retrieve database info and extract data_source_id if available
    ds_id = get_database_schema(datasource_id, HEADERS, verbose, milestone)
    if verbose or milestone:
        print(f"LOG: Milestone 6 - DB schema returned (ds_id = {ds_id or 'None'})")
    if verbose:
        print(f"VERBOSE: Resolved data_source_id: {ds_id}")

    if ds_id and ds_id != datasource_id:
        query_url = f"{BASE_URL}/data_sources/{ds_id}/query"
        if verbose:
            print(f"VERBOSE: Using /data_sources/ query for {ds_id}")
    else:
        query_url = f"{BASE_URL}/databases/{datasource_id}/query"
        if verbose:
            print(f"VERBOSE: Using /databases/ query for {datasource_id}")
    if verbose or milestone:
        print(f"LOG: Milestone 7 - Query URL determined")

    if verbose or milestone:
        print(f"LOG: Milestone 7.5 - Preparing payload")
    filter_obj = {"or": [{"property": "Status", "select": {"equals": s}} for s in STATUS_FILTER]}
    payload = {
        "filter": filter_obj,
        "page_size": 100,  # Max page size for efficiency
        "sorts": [{"timestamp": "created_time", "direction": "ascending"}],
    }
    if verbose or milestone:
        print(f"LOG: Milestone 8 - Payload ready, entering query loop")

    try:
        if verbose or milestone:
            print(f"LOG: Milestone 9 - Starting try block for query")
        eligible_tasks = []
        has_more = True
        start_cursor = None

        while has_more and len(eligible_tasks) < limit:
            if start_cursor:
                payload["start_cursor"] = start_cursor

            if verbose:
                print(f"VERBOSE: Making POST request to {query_url}")
            response = requests.post(query_url, headers=HEADERS, json=payload)
            if verbose:
                print(f"VERBOSE: Response status: {response.status_code}")
            if response.status_code != 200:
                raise requests.HTTPError(
                    f"Query failed: {response.status_code} - {response.text}"
                )

            data = response.json()
            results = data.get("results", [])
            has_more = data.get("has_more", False)
            start_cursor = data.get("next_cursor", None)

            if debug:
                print(f"DEBUG: Fetched {len(results)} pages, has_more: {has_more}")

            for page in results:
                if len(eligible_tasks) >= limit:
                    break

                status_prop = page["properties"].get("Status", {}).get("select", {}).get("name", "")
                if status_prop not in STATUS_FILTER:
                    if verbose:
                        print(f"VERBOSE: Skipping page {page['id']} - status '{status_prop}' not in filter")
                    continue

                # Skip recently edited
                last_edited_str = page.get("last_edited_time", "")
                if last_edited_str:
                    try:
                        last_edited = datetime.fromisoformat(last_edited_str.replace("Z", "+00:00"))
                        if datetime.now(last_edited.tzinfo) - last_edited < timedelta(seconds=30):
                            if verbose:
                                print(f"VERBOSE: Skipping recently edited page {page['id']}")
                            continue
                    except Exception:
                        pass

                # Get title
                title_prop = page["properties"].get("Title", {}).get("title", [])
                title = parse_rich_text(title_prop) or page["id"]

                # Get content rich_text
                content_rich_text = page["properties"].get("Content", {}).get("rich_text", [])

                eligible, trigger = is_eligible(content_rich_text, verbose)
                if not eligible:
                    if verbose:
                        print(f"VERBOSE: Page {page['id']} ineligible (no trigger in content)")
                    continue

                full_content = parse_rich_text(content_rich_text)
                tags = extract_tags(full_content)
                task_prompt = get_task_prompt(content_rich_text, trigger)

                eligible_tasks.append(
                    {
                        "page_id": page["id"],
                        "title": title,
                        "status": status_prop,
                        "execution_trigger": trigger,
                        "task_prompt": task_prompt,
                        "tags": tags,
                        "content": full_content
                    }
                )
                if verbose:
                    print(f"VERBOSE: Added eligible task: {title} (trigger: {trigger})")

        output = {
            "database_id": datasource_id,
            "data_source_id": ds_id,
            "tasks": eligible_tasks
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
        if verbose or milestone:
            print(f"LOG: Milestone 10 - Output printed ({len(eligible_tasks)} tasks)")
        if verbose:
            print(f"VERBOSE: Query complete ({len(eligible_tasks)} eligible out of {len(results)} total)")
    except requests.HTTPError as e:
        print(f"ERROR: HTTP request failed: {str(e)}", file=sys.stderr)
        if verbose:
            tb.print_exc(file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected failure: {str(e)}", file=sys.stderr)
        if verbose:
            tb.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: Unhandled error: {str(e)}", file=sys.stderr)
        tb.print_exc(file=sys.stderr)
        sys.exit(1)