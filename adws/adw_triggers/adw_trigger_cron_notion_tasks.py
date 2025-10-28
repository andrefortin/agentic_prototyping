#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pydantic",
#   "python-dotenv",
#   "click",
#   "rich",
#   "schedule",
#   "psutil",
#   "requests",
# ]
# ///
"""
Notion-based cron trigger for the multi-agent rapid prototyping system.

This script monitors Notion tasks and automatically distributes them to agents.
It runs continuously, checking for eligible tasks at a configurable interval.

Usage:
    # Method 1: Direct execution (requires uv)
    ./adws/adw_triggers/adw_trigger_cron_notion_tasks.py

    # Method 2: Using uv run
    uv run adws/adw_triggers/adw_trigger_cron_notion_tasks.py

    # With custom polling interval (seconds)
    ./adws/adw_triggers/adw_trigger_cron_notion_tasks.py --interval 15

    # Dry run mode (no changes made)
    ./adws/adw_triggers/adw_trigger_cron_notion_tasks.py --dry-run

Examples:
    # Run with custom database ID
    ./adws/adw_triggers/adw_trigger_cron_notion_tasks.py --database-id <notion-db-id>

    # Run once and exit
    ./adws/adw_triggers/adw_trigger_cron_notion_tasks.py --once

    # Run with verbose logging
    ./adws/adw_triggers/adw_trigger_cron_notion_tasks.py --verbose --milestone
"""

import os
import sys
import json
import time
import requests
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
import click  # type: ignore
import schedule  # type: ignore
import logging  # Standard logging
from rich.console import Console  # type: ignore
from rich.table import Table  # type: ignore
from rich.panel import Panel  # type: ignore
from rich.live import Live  # type: ignore
from rich.layout import Layout  # type: ignore
from rich.align import Align  # type: ignore

from contextlib import suppress

with suppress(ImportError):
    from dotenv import load_dotenv  # type: ignore
    # Load environment variables
    load_dotenv()


# Add the parent directory to the path so we can import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, "adw_modules"))

from agent import (
    AgentTemplateRequest,
    execute_template,
    generate_short_id,
)

# Import our data models
from data_models import (
    NotionTask,
    NotionCronConfig,
    NotionTaskUpdate,
    WorktreeCreationRequest,
    SystemTag,
)

# Import utility functions
from utils import parse_json  # type: ignore


def extract_tags(content):
    tags = {}
    import re
    for match in re.finditer(r"{{([^:]+):\s*([^\}]+)}}", content):
        key, value = match.groups()
        tags[key.strip()] = value.strip()
    return tags

def parse_rich_text(rich_text):
    return "".join([t.get("plain_text", "") for t in rich_text])


# Configuration constants - get from environment or use fallback
# Force load from env; raise error if missing
DEFAULT_DATABASE_ID = os.getenv("NOTION_AGENTIC_TASK_TABLE_ID", "")
DEFAULT_DATABASE_ID = DEFAULT_DATABASE_ID.replace("-", "") if DEFAULT_DATABASE_ID else ""
if not DEFAULT_DATABASE_ID:
    raise ValueError("NOTION_AGENTIC_TASK_TABLE_ID not found in .env file. Please set it to your clean Notion DB ID (32 chars, no hyphens: e.g., 29524ba1e67b80e3bf15000b74cc2405).")
DEFAULT_APPS_DIRECTORY = "apps"


class NotionTaskManager:
    """Manages Notion task operations using natural language commands."""

    def __init__(self, database_id: str, verbose: bool = False, milestone: bool = False):
        self.database_id = database_id
        self.console = Console()
        self.verbose = verbose
        self.milestone = milestone

    def _parse_raw_notion_results(self, pages, status_filter=None):
        """Parse raw Notion API results into eligible task dicts."""
        if status_filter is None:
            status_filter = ["Not started", "HIL Review"]
        tasks = []
        for page in pages:
            properties = page.get('properties', {})
            status_prop = properties.get('Status', {})
            status = status_prop.get('select', {}).get('name', 'Not started') if 'select' in status_prop else \
                     status_prop.get('rich_text', [{}])[0].get('plain_text', '') if 'rich_text' in status_prop else 'Not started'
            if status not in status_filter:
                if self.verbose:
                    print(f"VERBOSE: Skipping page {page.get('id', 'unknown')} - status '{status}' not in filter")
                continue

            # Skip recent edits (<30s)
            last_edited_str = page.get('last_edited_time', '')
            if last_edited_str:
                try:
                    last_edited = datetime.fromisoformat(last_edited_str.replace('Z', '+00:00'))
                    if datetime.now(timezone.utc) - last_edited < timedelta(seconds=30):
                        if self.verbose:
                            print(f"VERBOSE: Skipping recent edit for page {page.get('id', 'unknown')}")
                        continue
                except:
                    pass

            page_id = str(page.get('id', ''))
            title_prop = properties.get('Title', {}) or properties.get('Name', {})
            title = ''.join([t.get('plain_text', '') for t in title_prop.get('title', []) or []])

            content_rich = properties.get('Content', {}).get('rich_text', [])
            full_content = parse_rich_text(content_rich)
            tags = extract_tags(full_content)

            eligible, trigger = self._is_eligible_refined(content_rich, tags)
            if not eligible:
                if self.verbose:
                    print(f"VERBOSE: Page {page_id} ineligible (no --execute trigger)")
                continue

            task_prompt = self._get_task_prompt_refined(content_rich, trigger)

            tasks.append({
                "page_id": page_id,
                "title": title or page_id,
                "status": status,
                "content": full_content,
                "content_blocks": content_rich,
                "tags": tags,
                "execution_trigger": trigger,
                "task_prompt": task_prompt
            })
            if self.verbose:
                print(f"VERBOSE: Added eligible task '{title}' (trigger: {trigger})")

        if self.verbose:
            print(f"VERBOSE: Parsed {len(tasks)} eligible from {len(pages)} raw pages")
        return tasks

    def get_eligible_tasks(
        self, status_filter: Optional[List[str]] = None, limit: int = 10, debug: bool = False
    ) -> List[NotionTask]:
        """Get eligible tasks from Notion database using /get_notion_tasks command."""
        try:
            response = self._fetch_tasks_from_notion(
                status_filter or ["Not started", "HIL Review"], limit, debug
            )
            if not response.success:
                self._show_error(
                    "❌ Notion Query Failed",
                    f"Failed to get Notion tasks: {response.output}",
                )
                return []

            if self.verbose:
                print(f"VERBOSE: Raw response.output type = {type(response.output)}, length = {len(response.output) if isinstance(response.output, str) else 'N/A'}")
                print(f"VERBOSE: Raw response.output preview = {repr(response.output[:400]) + '...' if isinstance(response.output, str) and len(response.output) > 400 else repr(response.output)}")
            task_data = parse_json(response.output, list)
            if self.verbose:
                print(f"VERBOSE: After parse task_data len = {len(task_data) if task_data else 0}, type = {type(task_data)}")
                if task_data and isinstance(task_data, list) and task_data:
                    first_item = task_data[0]
                    print(f"VERBOSE: First item type = {type(first_item)}")
                    if isinstance(first_item, dict):
                        print(f"VERBOSE: First item keys = {list(first_item.keys())}")
                    else:
                        print(f"VERBOSE: First item value = {repr(str(first_item)[:100])}")
            return self._parse_task_response(response.output)
        except Exception as e:
            self._show_error(
                "❌ Task Retrieval Error",
                f"Error getting eligible Notion tasks: {str(e)}",
            )
            return []

    def _fetch_tasks_from_notion(self, status_filter: List[str], limit: int, debug: bool = False):
        # Fetch tasks directly from Notion API, modeled after adws/adw_get_notion_tasks.py.
        # Normalize database_id
        datasource_id = self.database_id
        if len(datasource_id.replace("-", "")) != 32:
            if self.verbose:
                print(f"VERBOSE: Warning - database_id length (without hyphens) {len(datasource_id.replace('-', ''))} != 32, using as-is: {datasource_id}")

        API_KEY = os.getenv("NOTION_API_KEY")
        if not API_KEY:
            raise ValueError("NOTION_API_KEY not set in .env")

        if self.verbose:
            api_key_masked = API_KEY[:4] + "****" + API_KEY[-4:]
            print(f"VERBOSE: API_KEY (masked): {api_key_masked}")
            print(f"VERBOSE: Using database_id: '{datasource_id}'")

        HEADERS = {
            "Authorization": f"Bearer {API_KEY}",
            "Notion-Version": os.getenv("NOTION_VERSION_API", "2025-09-03"),
            "Content-Type": "application/json",
        }

        # Get data_source_id
        ds_id = self._get_database_schema(datasource_id, HEADERS)
        if self.verbose:
            print(f"VERBOSE: Resolved data_source_id: {ds_id or 'None (using database_id)'}")

        # Determine query URL
        if ds_id and ds_id != datasource_id.replace("-", ""):
            query_url = f"https://api.notion.com/v1/data_sources/{ds_id}/query"
            if self.verbose:
                print(f"VERBOSE: Using data_sources query for {ds_id}")
        else:
            query_url = f"https://api.notion.com/v1/databases/{datasource_id}/query"
            if self.verbose:
                print(f"VERBOSE: Using databases query for {datasource_id}")

        # Prepare payload
        filter_obj = {"or": [{"property": "Status", "select": {"equals": s}} for s in status_filter]}
        payload = {
            "filter": filter_obj,
            "page_size": 100,
            "sorts": [{"timestamp": "created_time", "direction": "ascending"}],
        }

        if self.verbose:
            print(f"VERBOSE: Query URL: {query_url}")
            print(f"VERBOSE: Filter: {filter_obj}")
            print(f"VERBOSE: Limit: {limit}")

        # Query loop for pagination
        all_results = []
        has_more = True
        start_cursor = None
        retries = 3

        while has_more and len(all_results) < limit:
            if start_cursor:
                payload["start_cursor"] = start_cursor

            current_payload = payload.copy()  # Avoid mutating original

            for attempt in range(retries):
                if self.verbose:
                    print(f"VERBOSE: Query attempt {attempt + 1}/{retries}")
                try:
                    response = requests.post(query_url, headers=HEADERS, json=current_payload)
                    if response.status_code == 200:
                        break
                    if "rate" in response.text.lower() or response.status_code == 429:
                        wait_time = min(2 ** attempt, 60)
                        if self.verbose:
                            print(f"VERBOSE: Rate limit detected, waiting {wait_time}s (attempt {attempt + 1}/{retries})")
                        time.sleep(wait_time)
                        continue
                    raise requests.HTTPError(f"{response.status_code}: {response.text}")
                except requests.RequestException as e:
                    if attempt == retries - 1:
                        raise Exception(f"Query failed after {retries} attempts: {str(e)}")
                    time.sleep(2 ** attempt)

            data = response.json()
            results = data.get("results", [])
            all_results.extend(results)
            has_more = data.get("has_more", False)
            start_cursor = data.get("next_cursor", None)

            if self.verbose:
                print(f"VERBOSE: Fetched {len(results)} pages (total: {len(all_results)}), has_more: {has_more}")

        if self.verbose:
            print(f"VERBOSE: Total pages fetched: {len(all_results)}")

        # Parse raw results
        tasks = self._parse_raw_notion_results(all_results, status_filter)

        # Mock response object for consistency
        class MockResponse:
            def __init__(self, success, output):
                self.success = success
                self.output = json.dumps(tasks) if tasks else "[]"

        response = MockResponse(success=True, output=None)
        if self.verbose:
            print(f"LOG: Directly fetched and parsed {len(tasks)} eligible tasks")
        return response

    def _get_database_schema(self, datasource_id: str, headers: Dict[str, str]) -> Optional[str]:
        # Retrieve database and extract data_source_id if available. Calls https://api.notion.com/v1/databases/{database_id}.
        BASE_URL = "https://api.notion.com/v1"
        schema_url = f"{BASE_URL}/databases/{datasource_id}"
        if self.verbose:
            print(f"VERBOSE: Full DB API URL: {schema_url}")
        schema_response = requests.get(schema_url, headers=headers)
        if self.verbose:
            print(f"VERBOSE: DB API response status: {schema_response.status_code}")
        if schema_response.status_code != 200:
            if self.verbose:
                print(f"VERBOSE: Failed to retrieve database {datasource_id}: {schema_response.status_code} - {schema_response.text}")
            return None
        try:
            schema_data = schema_response.json()
            if self.verbose:
                print(f"VERBOSE: Database API response keys: {list(schema_data.keys())}")
            # In Notion API, check for data_sources or similar fields
            data_sources = schema_data.get("data_sources", [])
            if self.verbose:
                print(f"VERBOSE: Data sources for database {datasource_id}: {data_sources}")
            if data_sources:
                ds_id = data_sources[0].get("id")
                if self.verbose:
                    print(f"VERBOSE: Extracted data_source_id: {ds_id}")
                return ds_id
            # Fallback: if no data_sources, assume database_id itself is usable
            if self.verbose:
                print(f"VERBOSE: No data_sources found in database {datasource_id}; using database_id directly.")
            return datasource_id
        except json.JSONDecodeError:
            if self.verbose:
                print(f"VERBOSE: Invalid JSON response from database retrieval: {schema_response.text}")
            return None

    def _parse_task_response(self, response_output: str) -> List[NotionTask]:
        """Parse JSON response and convert to NotionTask objects."""
        try:
            task_data = parse_json(response_output, list)
            if not task_data:
                return []

            # Fallback for raw Notion API response
            if task_data and isinstance(task_data, list) and len(task_data) > 0 and isinstance(task_data[0], dict) and task_data[0].get('object') == 'list':
                if self.verbose:
                    print("VERBOSE: Detected raw Notion API response, processing raw results")
                raw_results = task_data[0].get('results', [])
                task_data = []  # Reset to build processed items
                for page in raw_results:
                    page_id = str(page.get('id', ''))
                    if not page_id:
                        continue
                    title_prop = page['properties'].get('Title', {}) or page['properties'].get('Name', {})
                    title = ''.join([t.get('plain_text', '') for t in title_prop.get('title', [])])
                    status_prop = page['properties'].get('Status', {})
                    status = status_prop.get('select', {}).get('name', 'Not started') if 'select' in status_prop else \
                             (status_prop.get('rich_text', [{}])[0].get('plain_text', '') if 'rich_text' in status_prop else 'Not started')
                    content_rich = page['properties'].get('Content', {}).get('rich_text', [])
                    full_content = ''.join([item.get('plain_text', '') for item in content_rich])
                    # Parse eligibility using refined logic
                    eligible, trigger = self._is_eligible_refined(content_rich)
                    if not eligible:
                        continue
                    tags = extract_tags(full_content)
                    task_prompt = self._get_task_prompt_refined(content_rich, trigger)
                    task_data.append({
                        "page_id": page_id,
                        "title": title,
                        "status": status,
                        "content": full_content,
                        "content_blocks": content_rich,
                        "tags": tags,
                        "execution_trigger": trigger,
                        "task_prompt": task_prompt
                    })
                if self.verbose:
                    print(f"VERBOSE: Processed {len(task_data)} eligible tasks from raw response")

            tasks = []
            for i, task_item in enumerate(task_data):
                if self.verbose:
                    print(f"VERBOSE: Processing task_item {i} type = {type(task_item)}")
                if task := self._create_notion_task(task_item, i):
                    tasks.append(task)
            if self.verbose or self.milestone:
                print(f"LOG: Parsed {len(tasks)} eligible tasks")
            return tasks
        except (ValueError, json.JSONDecodeError) as e:
            self._show_error(
                "❌ Parse Error",
                f"Failed to parse Notion tasks response: {e}. Text was: {response_output[:500]}",
            )
            return []

    def _is_eligible_refined(self, rich_text, tags=None):
        """Check eligibility: prototype tags or explicit triggers make it eligible."""
        if self.verbose:
            print(f"VERBOSE: Checking eligibility for rich_text (len={len(rich_text)})")
        if not rich_text:
            return False, None

        full_content = ''.join([item.get('plain_text', '') for item in rich_text]).strip()

        # Check for explicit triggers (anywhere)
        if 'execute' in full_content or '{{execute}}' in full_content:
            if self.verbose:
                print(f"VERBOSE: Found explicit 'execute' trigger")
            return True, "execute"
        elif 'continue -' in full_content:
            if self.verbose:
                print(f"VERBOSE: Found 'continue' trigger")
            return True, "continue"

        # If no explicit trigger, check for prototype tags (implied execute for prototypes)
        if tags and tags.get('prototype'):
            if self.verbose:
                print(f"VERBOSE: Found prototype tag '{tags['prototype']}', treating as eligible 'execute'")
            return True, "execute"

        return False, None

    def _get_task_prompt_refined(self, rich_text, trigger):
        """Extract task_prompt based on trigger, using full content."""
        if not rich_text or not trigger:
            return ''.join([item.get('plain_text', '') for item in rich_text]).strip()

        full_content = ''.join([item.get('plain_text', '') for item in rich_text]).strip()

        if trigger == "execute":
            # Remove the '--execute' part for prompt
            prompt = full_content.replace('--execute', '').replace('{{--execute}}', '').strip()
            return prompt
        elif trigger == "continue":
            # Extract after the first "--continue -"
            if '--continue -' in full_content:
                parts = full_content.split('--continue -', 1)
                return parts[1].strip() if len(parts) > 1 else full_content
            return full_content  # Fallback
        return full_content

    def _create_notion_task(self, task_item: dict, index: int) -> Optional[NotionTask]:
        """Create NotionTask from task item data."""
        try:
            if not isinstance(task_item, dict):
                if self.verbose:
                    print(
                        f"VERBOSE: Skipping invalid task_item {index} (type: {type(task_item)}) - expected dict"
                    )
                return None

            # Use content from response if present, fallback to parsing
            content = task_item.get("content", "")
            if not content:
                content_rich = task_item.get("content_blocks", [])
                content = ''.join([item.get('plain_text', '') for item in content_rich])

            eligible, trigger = self._is_eligible_refined(task_item.get("content_blocks", []), task_item.get("tags"))
            if not eligible:
                if self.verbose:
                    print(f"VERBOSE: Task {index} not eligible after refined check")
                return None

            notion_task = NotionTask(
                page_id=str(task_item.get("page_id", "")),
                title=str(task_item.get("title", "")),
                status=str(task_item.get("status", "Not started")),
                content_blocks=list(task_item.get("content_blocks", [])),
                tags=dict(task_item.get("tags", {})),
                execution_trigger=task_item.get("execution_trigger") or trigger or None,
                task_prompt=str(task_item.get("task_prompt", content)),
                worktree=task_item.get("tags", {}).get("worktree"),
                model=task_item.get("tags", {}).get("model"),
                workflow_type=task_item.get("tags", {}).get("workflow"),
                prototype=task_item.get("tags", {}).get("prototype"),
            )
            return notion_task if notion_task.is_eligible_for_processing() else None
        except Exception as e:
            self.console.print(
                f"[yellow]Warning: Failed to parse task {task_item.get('page_id', 'unknown')}: {e}[/yellow]"
            )
            return None

    def _show_error(self, title: str, message: str):
        """Display error panel."""
        error_panel = Panel(
            message, title=f"[bold red]{title}[/bold red]", border_style="red"
        )
        self.console.print(error_panel)

    def update_task_status(
        self, page_id: str, status: str, update_content: str = ""
    ) -> bool:
        """Update a Notion task status using /update_notion_task command."""
        try:
            request = AgentTemplateRequest(
                agent_name="notion-task-updater",
                slash_command="/update_notion_task",
                args=[page_id, status, update_content],
                adw_id=generate_short_id(),
                model="x-ai/grok-4-fast",
                working_dir=os.getcwd(),
            )

            # Retry with backoff on rate limit errors
            retries = 3
            for attempt in range(retries):
                if self.verbose:
                    print(f"VERBOSE: Update status attempt {attempt + 1}/{retries}")
                response = execute_template(request)
                if response.success:
                    if self.verbose or self.milestone:
                        print(f"LOG: Status update successful for page {page_id[:12]}...")
                    return True
                if "rate" in response.output.lower() or "429" in str(response):  # Detect rate limit
                    wait_time = min(2 ** attempt, 60)  # Exponential backoff, max 60s
                    self.logger.warning(f"Rate limit detected, waiting {wait_time}s (attempt {attempt + 1}/{retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    error_panel = Panel(
                        f"Failed to update Notion task status: {response.output}",
                        title="[bold red]❌ Status Update Failed[/bold red]",
                        border_style="red",
                    )
                    self.console.print(error_panel)
                    return False
            self.logger.error("All retries failed for status update")
            return False
        except Exception as e:
            error_panel = Panel(
                f"Error updating Notion task status: {str(e)}",
                title="[bold red]❌ Status Update Error[/bold red]",
                border_style="red",
            )
            self.console.print(error_panel)
            return False

    def fetch_single_task(self, page_id: str) -> Optional[NotionTask]:
        """Fetch a single task by page_id to verify eligibility using direct API."""
        try:
            API_KEY = os.getenv("NOTION_API_KEY")
            if not API_KEY:
                if self.verbose:
                    print(f"VERBOSE: NOTION_API_KEY not set")
                return None

            HEADERS = {
                "Authorization": f"Bearer {API_KEY}",
                "Notion-Version": os.getenv("NOTION_VERSION_API", "2025-09-03"),
            }

            page_url = f"https://api.notion.com/v1/pages/{page_id}"
            if self.verbose:
                print(f"VERBOSE: Fetching single page: {page_url}")

            response = requests.get(page_url, headers=HEADERS)
            if response.status_code != 200:
                if self.verbose:
                    print(f"VERBOSE: Failed to fetch page {page_id}: {response.status_code} - {response.text}")
                return None

            page_data = response.json()
            properties = page_data.get('properties', {})

            status_prop = properties.get('Status', {})
            status = status_prop.get('select', {}).get('name', '') if 'select' in status_prop else \
                     status_prop.get('rich_text', [{}])[0].get('plain_text', '') if 'rich_text' in status_prop else 'Unknown'

            if status not in ['Not started', 'HIL Review']:
                if self.verbose:
                    print(f"VERBOSE: Page {page_id} status '{status}' not eligible")
                return None

            # Minimal task for re-verify (status check only, no content needed for eligibility)
            title_prop = properties.get('Title', {}) or properties.get('Name', {})
            title = ''.join([t.get('plain_text', '') for t in title_prop.get('title', []) or []])

            content_rich = properties.get('Content', {}).get('rich_text', [])
            full_content = parse_rich_text(content_rich)
            tags = extract_tags(full_content)

            # For re-verify, we don't need full eligibility check, just status
            # But to create NotionTask, run the check
            eligible, trigger = self._is_eligible_refined(content_rich, tags)
            if not eligible:
                if self.verbose:
                    print(f"VERBOSE: Page {page_id} not eligible after content check")
                return None

            task_prompt = self._get_task_prompt_refined(content_rich, trigger)

            task_item = {
                "page_id": page_id,
                "title": title or page_id,
                "status": status,
                "content": full_content,
                "content_blocks": content_rich,
                "tags": tags,
                "execution_trigger": trigger,
                "task_prompt": task_prompt
            }

            return self._create_notion_task(task_item, 0)

        except Exception as e:
            if self.verbose:
                print(f"VERBOSE: Exception in fetch_single_task: {str(e)}")
            self.console.print(f"[yellow]Warning: Failed to fetch single task {page_id}: {e}[/yellow]")
            return None

    def generate_worktree_name(
        self, task_description: str, prefix: str = ""
    ) -> Optional[str]:
        """Generate a worktree name using /make_worktree_name command."""
        try:
            request = AgentTemplateRequest(
                agent_name="worktree-namer",
                slash_command="/make_worktree_name",
                args=[task_description, prefix],
                adw_id=generate_short_id(),
                model="x-ai/grok-4-fast",
                working_dir=os.getcwd(),
            )

            # Retry with backoff on rate limit errors
            retries = 3
            for attempt in range(retries):
                if self.verbose:
                    print(f"VERBOSE: Generate name attempt {attempt + 1}/{retries}")
                response = execute_template(request)
                if response.success:
                    # Trust the prompt to return just the worktree name
                    worktree_name = response.output.strip()

                    # Display the generated worktree name in a panel
                    name_panel = Panel(
                        f"[bold cyan]{worktree_name}[/bold cyan]",
                        title="[bold blue]📁 Generated Worktree Name[/bold blue]",
                        border_style="blue",
                        padding=(0, 2),
                    )
                    self.console.print(name_panel)

                    if self.verbose or self.milestone:
                        print(f"LOG: Worktree name generated: {worktree_name}")
                    return worktree_name
                if "rate" in response.output.lower() or "429" in str(response):  # Detect rate limit
                    wait_time = min(2 ** attempt, 60)  # Exponential backoff, max 60s
                    self.logger.warning(f"Rate limit detected, waiting {wait_time}s (attempt {attempt + 1}/{retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    self.logger.warning(f"Failed to generate worktree name (attempt {attempt + 1}/{retries}): {response.output}")
            self.logger.error("All retries failed for generating worktree name")
            return None
        except Exception as e:
            self.console.print(f"[red]Error generating worktree name: {e}[/red]")
            return None


class NotionCronTrigger:
    """Main Notion-based cron trigger implementation."""

    def __init__(self, config: NotionCronConfig, verbose: bool = False, milestone: bool = False):
        self.config = config
        self.console = Console()
        self.task_manager = NotionTaskManager(config.database_id, verbose=verbose, milestone=milestone)
        self.running = True
        self.verbose = verbose
        self.milestone = milestone
        self.stats = {
            "checks": 0,
            "tasks_started": 0,
            "worktrees_created": 0,
            "notion_updates": 0,
            "errors": 0,
            "last_check": None,
            "rate_limit_backoff": 0,  # Track backoff time
        }
        # Track subprocess PIDs for monitoring
        self.active_pids = set()

        # Utility function to ensure directory exists
        def ensure_dir(path, label):
            if not os.path.exists(path):
                os.makedirs(path)
                if self.verbose:
                    print(f"VERBOSE: Created directory: {path} ({label})")
            else:
                if self.verbose:
                    print(f"VERBOSE: Directory exists: {path} ({label})")

        # Ensure all required directories at project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        required_dirs = ["agents", "apps", "logs", "specs", "trees"]
        for dir_name in required_dirs:
            dir_path = os.path.join(project_root, dir_name)
            ensure_dir(dir_path, f"[App Start] {dir_name}")

        # Setup logging with checks
        log_file = os.path.join(project_root, "logs", "notion_cron.log")
        logs_dir = os.path.dirname(log_file)

        # Function to ensure logs dir before write
        def ensure_logs_dir():
            ensure_dir(logs_dir, "[Before Write] logs")
            return log_file

        log_level = logging.DEBUG if config.debug else logging.INFO
        if self.verbose:
            log_level = logging.DEBUG

        class CheckedFileHandler(logging.FileHandler):
            def __init__(self, filename, mode='a', encoding=None, delay=False):
                ensure_logs_dir()
                super().__init__(filename, mode, encoding, delay)

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                CheckedFileHandler(ensure_logs_dir()),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Logging initialized with directory checks")

        # Update nested keys in .mcp.notion.json with NOTION_VERSION_MCP and NOTION_API_KEY if set
        mcp_file = os.path.join(project_root, ".mcp.notion.json")
        if os.path.exists(mcp_file):
            try:
                with open(mcp_file, 'r') as f:
                    mcp_data = json.load(f)

                # Access nested env
                if "mcpServers" in mcp_data and "notion" in mcp_data["mcpServers"] and "env" in mcp_data["mcpServers"]["notion"]:

                    env = mcp_data["mcpServers"]["notion"]["env"]

                    # Update Notion-Version
                    notion_version_mcp = os.getenv("NOTION_VERSION_MCP")
                    if notion_version_mcp and notion_version_mcp.strip():
                        env["Notion-Version"] = notion_version_mcp
                        if self.verbose:
                            print(f"VERBOSE: Updated .mcp.notion.json mcpServers.notion.env Notion-Version to {notion_version_mcp}")

                    # Update NOTION_TOKEN
                    notion_api_key = os.getenv("NOTION_API_KEY")
                    if notion_api_key and notion_api_key.strip():
                        env["NOTION_TOKEN"] = notion_api_key
                        if self.verbose:
                            print(f"VERBOSE: Updated .mcp.notion.json mcpServers.notion.env NOTION_TOKEN from NOTION_API_KEY")

                    # Write back if any changes
                    if notion_version_mcp or notion_api_key:
                        with open(mcp_file, 'w') as f:
                            json.dump(mcp_data, f, indent=4)

            except Exception as e:
                self.logger.warning(f"Failed to update .mcp.notion.json: {str(e)}")
        else:
            if self.verbose:
                print("VERBOSE: .mcp.notion.json not found - skipping update")

        if self.verbose or self.milestone:
            print(f"LOG: NotionCronTrigger initialized (debug={config.debug}, verbose={self.verbose}, milestone={self.milestone})")


    def check_worktree_exists(self, worktree_name: str) -> bool:
        """Check if a worktree already exists."""
        worktree_path = Path(self.config.worktree_base_path) / worktree_name
        exists = worktree_path.exists()
        if self.verbose:
            print(f"VERBOSE: Worktree '{worktree_name}' exists: {exists}")
        return exists

    def create_worktree(self, worktree_name: str) -> bool:
        """Create a new worktree using the init_worktree command."""
        if self.config.dry_run:
            self.console.print(
                f"[yellow]DRY RUN: Would create worktree '{worktree_name}'[/yellow]"
            )
            return True

        try:
            # For this project, always use the project root for sparse checkout
            target_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

            request = AgentTemplateRequest(
                agent_name="worktree-creator",
                slash_command="/init_worktree",
                args=[worktree_name, target_directory],
                adw_id=generate_short_id(),
                model="x-ai/grok-4-fast",
                working_dir=os.getcwd(),
            )

            response = execute_template(request)
            if response.success:
                self.stats["worktrees_created"] += 1
                success_panel = Panel(
                    f"✓ Created worktree: {worktree_name} → {target_directory}",
                    title="[bold green]Worktree Created[/bold green]",
                    border_style="green",
                )
                self.console.print(success_panel)
                if self.verbose or self.milestone:
                    print(f"LOG: Worktree '{worktree_name}' created successfully")
                return True
            else:
                error_panel = Panel(
                    f"Failed to create worktree: {worktree_name}",
                    title="[bold red]❌ Worktree Creation Failed[/bold red]",
                    border_style="red",
                )
                self.console.print(error_panel)
                self.stats["errors"] += 1
                return False
        except Exception as e:
            error_panel = Panel(
                f"Error creating worktree: {str(e)}",
                title="[bold red]❌ Worktree Creation Error[/bold red]",
                border_style="red",
            )
            self.console.print(error_panel)
            self.stats["errors"] += 1
            return False

    def delegate_task(self, task: NotionTask, worktree_name: str, adw_id: str):
        # Delegate a Notion task to the appropriate workflow (no agent dependency).
        # Determine workflow and model
        use_full_workflow = (
            task.should_use_full_workflow() or task.prototype is not None
        )
        model = task.get_thinking_model()  # Use thinking model for planning tasks

        if self.config.dry_run:
            workflow_type = (
                "plan-implement-update" if use_full_workflow else "build-update"
            )
            self.console.print(
                f"[yellow]DRY RUN: Would delegate task '{task.title}' to {workflow_type} workflow with {model} model[/yellow]"
            )
            return

        try:
            # Determine which workflow script to use
            if use_full_workflow:
                workflow_script = f"adws/adw_plan_implement_update_notion_task.py"
                workflow_type = "plan-implement-update"
            else:
                workflow_script = f"adws/adw_build_update_notion_task.py"
                workflow_type = "build-update"

            # Build the command to run the workflow
            # Combine title and prompt for better context
            combined_task = (
                f"{task.title}: {task.task_prompt}" if task.task_prompt else task.title
            )
            cmd = [
                "python",
                workflow_script,
                "--adw-id",
                adw_id,
                "--worktree-name",
                worktree_name,
                "--task",
                combined_task,
                "--page-id",
                task.page_id,
                "--model-thinking",
                task.get_thinking_model(),
                "--model-fast",
                task.get_fast_model(),
            ]

            # Add prototype flag if specified
            if task.prototype:
                cmd.extend(["--prototype", task.prototype])

            # Add verbose and milestone if set
            if self.verbose:
                cmd.append("--verbose")
            if self.milestone:
                cmd.append("--milestone")
            # Pass milestone to subprocess for logging

            # Create a panel showing the agent execution details
            exec_details = f"[bold]Page ID:[/bold] {task.page_id}\n"
            exec_details += f"[bold]Title:[/bold] {task.title}\n"
            exec_details += f"[bold]Workflow:[/bold] {workflow_type}\n"
            exec_details += f"[bold]Arguments:[/bold]\n"
            exec_details += f"  • ADW ID: {adw_id}\n"
            exec_details += f"  • Worktree: {worktree_name}\n"
            exec_details += f"  • Task: {task.task_prompt[:50]}{'...' if len(task.task_prompt or '') > 50 else ''}\n"
            exec_details += f"  • Model: {model}\n"

            if task.tags:
                tags_str = ", ".join([f"{k}: {v}" for k, v in task.tags.items()])
                exec_details += f"  • Tags: {tags_str}"

            exec_panel = Panel(
                exec_details,
                title="[bold cyan]🤖 Executing Workflow[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            )
            self.console.print(exec_panel)

            # Run the workflow in a subprocess
            # Use start_new_session=True to detach the process and let it survive parent death
            try:
                result = subprocess.Popen(cmd, start_new_session=True)
                self.stats["tasks_started"] += 1
                # Track PID for monitoring
                self.active_pids.add(result.pid)
                if self.verbose:
                    print(f"VERBOSE: Started PID {result.pid} for task {adw_id}")
            except Exception as spawn_error:
                error_msg = f"Failed to spawn subprocess for task {adw_id}: {str(spawn_error)}"
                self.console.print(Panel(error_msg, title="[bold red]❌ Subprocess Spawn Failed[/bold red]", border_style="red"))
                self.stats["errors"] += 1
                # Rollback status if not dry-run
                if not self.config.dry_run:
                    self.task_manager.update_task_status(task.page_id, "Failed", error_msg)
                return  # Continue to next task

            # Create success panel for task delegation
            delegation_panel = Panel(
                f"✓ Task delegated with ADW ID: {adw_id} (PID: {result.pid})",
                title="[bold green]✅ Notion Task Delegated[/bold green]",
                border_style="green",
            )
            self.console.print(delegation_panel)
            if self.verbose or self.milestone:
                print(f"LOG: Task '{task.title}' delegated (ADW ID: {adw_id}, PID: {result.pid})")

        except Exception as e:
            if self.verbose:
                print(f"VERBOSE: Error delegating task: {str(e)}")
            error_panel = Panel(
                f"Error delegating Notion task: {str(e)}",
                title="[bold red]❌ Delegation Failed[/bold red]",
                border_style="red",
            )
            self.console.print(error_panel)
            self.stats["errors"] += 1

    def monitor_active_pids(self):
        """Check status of active subprocesses and clean up PIDs."""
        import psutil
        if not self.active_pids:
            return
        dead_pids = []
        for pid in self.active_pids:
            try:
                pid_obj = psutil.Process(pid)
                if not pid_obj.is_running():
                    dead_pids.append(pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                dead_pids.append(pid)
        for dead_pid in dead_pids:
            self.active_pids.remove(dead_pid)
            if self.verbose:
                print(f"VERBOSE: Cleaned up dead PID {dead_pid}")

    def process_tasks(self):
        """Main task processing logic for Notion tasks."""
        if self.verbose or self.milestone:
            print(f"LOG: Starting task processing cycle")

        # Monitor active PIDs before new cycle
        self.monitor_active_pids()

        self.stats["checks"] += 1
        self.stats["last_check"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Get eligible tasks from Notion
        eligible_tasks = self.task_manager.get_eligible_tasks(
            status_filter=self.config.status_filter,
            limit=self.config.max_concurrent_tasks,
            debug=self.config.debug,
        )

        if not eligible_tasks:
            if self.verbose or self.milestone:
                print(f"LOG: No eligible tasks found")
            info_panel = Panel(
                "No eligible tasks found in Notion database. Add tasks with '--execute' or '--continue -' triggers.",
                title="[bold yellow]No Notion Tasks[/bold yellow]",
                border_style="yellow",
            )
            self.console.print(info_panel)
            return

        # Report tasks that will be processed
        task_summary_lines = []
        for task in eligible_tasks:
            tags_str = ""
            if task.tags:
                tags_list = [f"{k}: {v}" for k, v in task.tags.items()]
                tags_str = f" [dim]({', '.join(tags_list)})[/dim]"

            task_summary_lines.extend([
                f"  • {task.title}{tags_str}",
                f"    [dim]Status: {task.status} | Trigger: {task.execution_trigger} | Page: {task.page_id[:12]}...[/dim]"
            ])

        if task_summary_lines:
            tasks_panel = Panel(
                "\n".join(task_summary_lines),
                title=f"[bold green]🚀 Processing {len(eligible_tasks)} Notion Task{'s' if len(eligible_tasks) != 1 else ''}[/bold green]",
                border_style="green",
            )
            self.console.print(tasks_panel)

        # Process each task
        for task in eligible_tasks:
            if self.verbose or self.milestone:
                print(f"LOG: Processing task '{task.title}' (page_id={task.page_id[:12]}...)")
            try:
                # Re-verify task is still eligible before claiming
                current_task = self.task_manager.fetch_single_task(task.page_id)
                if not current_task or not current_task.is_eligible_for_processing():
                    if self.verbose:
                        print(f"VERBOSE: Task {task.page_id} no longer eligible, skipping.")
                    self.console.print(f"[yellow]Task {task.page_id} no longer eligible, skipping.[/yellow]")
                    continue

                # Generate ADW ID for this task
                adw_id = generate_short_id()

                # IMMEDIATELY update task status to "In progress" in Notion
                # This marks the task as being picked up BEFORE any processing
                # IMMEDIATELY update task status to "In progress" in Notion to take ownership
                update_content_claim = {
                    "status": "In progress",
                    "adw_id": adw_id,
                    "timestamp": datetime.now().isoformat(),
                    "trigger": task.execution_trigger,
                    "previous_status": task.status,
                    "claimed_by": "cron-trigger",
                }
                claim_success = self.task_manager.update_task_status(
                    task.page_id, "In progress", json.dumps(update_content_claim)
                )
                if not claim_success and not self.config.dry_run:
                    error_panel = Panel(
                        f"[bold red]CRITICAL: Failed to claim task {task.title} to 'In progress' - cannot proceed to avoid duplicates[/bold red]\n"
                        f"Page ID: {task.page_id}",
                        title="[bold red]❌ Task Claim Failed - Aborting[/bold red]",
                        border_style="red",
                    )
                    self.console.print(error_panel)
                    self.stats["errors"] += 1
                    # Do not continue to delegation without claim
                    return
                elif claim_success:
                    self.stats["notion_updates"] += 1
                    claim_panel = Panel(
                        f"[bold green]✓ Ownership claimed: Task '{task.title}' moved to 'In progress'[/bold green]\n"
                        f"ADW ID: {adw_id} | Page ID: {task.page_id}",
                        title="[bold green]✅ Task Ownership Secured[/bold green]",
                        border_style="green",
                    )
                    self.console.print(claim_panel)
                else:
                    self.console.print(
                        f"[yellow]DRY RUN: Would claim task '{task.title}' to 'In progress' with ADW ID {adw_id}[/yellow]"
                    )

                # Proceed only if claimed successfully (status remains 'In progress' until workflow completion)

                # After successful delegation, the workflow will handle final status (Done/Failed)
                # Cron does not rollback unless spawn fails immediately

                # Generate or extract worktree name
                if task.worktree:
                    worktree_name = task.worktree
                else:
                    # Generate worktree name
                    worktree_name = self.task_manager.generate_worktree_name(
                        task.task_prompt or task.title, "task"
                    )
                    if not worktree_name:
                        if self.verbose:
                            print(f"VERBOSE: Failed to generate worktree name for task: {task.title}")
                        self.console.print(
                            f"[red]Failed to generate worktree name for task: {task.title}[/red]"
                        )
                        # Rollback to 'Failed' only if claim succeeded but can't proceed
                        if claim_success and not self.config.dry_run:
                            rollback_content = {
                                "status": "Failed",
                                "adw_id": adw_id,
                                "timestamp": datetime.now().isoformat(),
                                "error": "Failed to generate worktree name",
                                "claimed_by": "cron-trigger",
                            }
                            self.task_manager.update_task_status(
                                task.page_id, "Failed", json.dumps(rollback_content)
                            )
                        continue

                # Check if worktree exists, create if needed
                if not self.check_worktree_exists(worktree_name):
                    info_panel = Panel(
                        f"Worktree '{worktree_name}' doesn't exist, creating...",
                        title="[bold yellow]ℹ️ Creating Worktree[/bold yellow]",
                        border_style="yellow",
                    )
                    self.console.print(info_panel)
                    if not self.create_worktree(worktree_name):
                        # If we updated status but can't proceed, mark as failed
                        if not self.config.dry_run:
                            self.task_manager.update_task_status(
                                task.page_id, "Failed", "Failed to create worktree"
                            )
                        continue  # Skip this task if worktree creation failed

                # Delegate task to appropriate workflow
                self.delegate_task(task, worktree_name, adw_id)

                # Respect max concurrent tasks limit
                if self.stats["tasks_started"] >= self.config.max_concurrent_tasks:
                    if self.verbose or self.milestone:
                        print(f"LOG: Reached max concurrent tasks ({self.config.max_concurrent_tasks})")
                    warning_panel = Panel(
                        f"Reached max concurrent tasks ({self.config.max_concurrent_tasks})",
                        title="[bold yellow]⚠️ Task Limit[/bold yellow]",
                        border_style="yellow",
                    )
                    self.console.print(warning_panel)
                    break

            except Exception as e:
                if self.verbose:
                    print(f"VERBOSE: Error processing task {task.page_id if 'task' in locals() else 'unknown'}: {str(e)}")
                error_panel = Panel(
                    f"Error processing Notion task: {str(e)}",
                    title="[bold red]❌ Task Processing Error[/bold red]",
                    border_style="red",
                )
                self.console.print(error_panel)
                self.stats["errors"] += 1
                continue
        if self.verbose or self.milestone:
            print(f"LOG: Task processing cycle complete (tasks_started={self.stats['tasks_started']}, errors={self.stats['errors']})")

    def create_status_display(self) -> Panel:
        """Create a status display panel for Notion operations."""
        table = Table(show_header=False, box=None)
        table.add_column(style="bold cyan")
        table.add_column()

        table.add_row(
            "Status", "[green]Running[/green]" if self.running else "[red]Stopped[/red]"
        )
        table.add_row("Polling Interval", f"{self.config.polling_interval} seconds")
        table.add_row("Notion Database", f"{self.config.database_id[:12]}...")
        table.add_row("Status Filter", str(self.config.status_filter))
        table.add_row("Max Concurrent", str(self.config.max_concurrent_tasks))
        table.add_row("Dry Run", "Yes" if self.config.dry_run else "No")
        table.add_row("Active PIDs", str(len(self.active_pids)))
        table.add_row("", "")
        table.add_row("Checks", str(self.stats["checks"]))
        table.add_row("Tasks Started", str(self.stats["tasks_started"]))
        table.add_row("Worktrees Created", str(self.stats["worktrees_created"]))
        table.add_row("Notion Updates", str(self.stats["notion_updates"]))
        table.add_row("Errors", str(self.stats["errors"]))
        table.add_row("Last Check", self.stats["last_check"] or "Never")

        return Panel(
            Align.center(table),
            title="[bold blue]🔄 Notion Multi-Agent Cron[/bold blue]",
            border_style="blue",
        )

    def run_once(self):
        """Run the task check once and exit."""
        self.console.print(self.create_status_display())
        self.console.print("\n[yellow]Running single Notion check...[/yellow]\n")
        self.process_tasks()
        self.console.print("\n[green]✅ Single Notion check completed[/green]")

    def run_continuous(self):
        """Run continuously with scheduled checks."""
        # Schedule the task processing
        schedule.every(self.config.polling_interval).seconds.do(self.process_tasks)

        self.console.print(self.create_status_display())
        self.console.print(
            f"\n[green]Started monitoring Notion tasks every {self.config.polling_interval} seconds[/green]"
        )
        self.console.print("[dim]Press Ctrl+C to stop[/dim]\n")

        try:
            while self.running:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            self.running = False
            self.console.print("\n[yellow]Stopping Notion cron trigger...[/yellow]")
            self.console.print(self.create_status_display())
            self.console.print("[green]✅ Notion cron trigger stopped[/green]")

    # Add missing methods from original if needed, but ensure no errors
    def logger(self):
        return self.logger


@click.command()
@click.option(
    "--interval", type=int, default=15, help="Polling interval in seconds (default: 15)"
)
@click.option(
    "--database-id",
    type=str,
    default=DEFAULT_DATABASE_ID,
    help=f"Notion database ID (default: {DEFAULT_DATABASE_ID[:12]}... from env)",
)
@click.option(
    "--dry-run", is_flag=True, help="Run in dry-run mode without making changes"
)
@click.option(
    "--max-tasks", type=int, default=3, help="Maximum concurrent tasks (default: 3)"
)
@click.option(
    "--once", is_flag=True, help="Run once and exit instead of continuous monitoring"
)
@click.option(
    "--status-filter",
    type=str,
    default='["Not started", "HIL Review"]',
    help='Status filter as JSON array (default: ["Not started", "HIL Review"])',
)
@click.option("--debug", is_flag=True, help="Enable debug mode for task fetching")
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
@click.option("--milestone", is_flag=True, help="Display milestone logs")
def main(
    interval: int,
    database_id: str,
    dry_run: bool,
    max_tasks: int,
    once: bool,
    status_filter: str,
    debug: bool,
    verbose: bool,
    milestone: bool,
):
    """Monitor and distribute tasks from the Notion Agentic Prototyper database."""
    console = Console()
    if verbose or milestone:
        print(f"LOG: main() started (verbose={verbose}, milestone={milestone})")

    # Check if database ID is properly configured
    if not database_id:
        console.print(
            Panel(
                f"[bold red]No Notion database ID configured![/bold red]\n\n"
                f"Expected: NOTION_AGENTIC_TASK_TABLE_ID={os.getenv('NOTION_AGENTIC_TASK_TABLE_ID')}\n\n"
                "Please set it in your .env file or provide via --database-id option.",
                title="[bold red]❌ Configuration Error[/bold red]",
                border_style="red",
            )
        )
        sys.exit(1)

    parsed_status_filter = ["Not started", "HIL Review"]

    try:
        # Parse status filter
        parsed_status_filter = json.loads(status_filter)
        if not isinstance(parsed_status_filter, list):
            raise ValueError("Status filter must be a JSON array")
    except (json.JSONDecodeError, ValueError) as e:
        if verbose:
            print(f"VERBOSE: Error parsing status filter: {e}")
        console.print(f"[red]Error parsing status filter: {e}[/red]")
        console.print("[yellow]Using default: ['Not started', 'HIL Review'][/yellow]")
        parsed_status_filter = ["Not started", "HIL Review"]

    if verbose or milestone:
        print(f"LOG: Loaded config (database_id='{database_id[:12]}...', verbose={verbose}, milestone={milestone})")

    # Create configuration
    config = NotionCronConfig(
        database_id=database_id,
        polling_interval=interval,
        dry_run=dry_run,
        max_concurrent_tasks=max_tasks,
        status_filter=parsed_status_filter,
        debug=debug,
        worktree_base_path="trees",
    )

    # Create and run the trigger
    trigger = NotionCronTrigger(config, verbose=verbose, milestone=milestone)

    if once:
        trigger.run_once()
    else:
        trigger.run_continuous()


if __name__ == "__main__":
    main()
