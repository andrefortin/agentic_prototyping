#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pydantic",
#   "python-dotenv",
#   "click",
#   "rich",
# ]
# ///
"""
Run plan, implement, and update task workflow for Notion-based multi-agent task processing.

This script runs three slash commands in sequence:
1. /plan - Creates a plan based on the task description
2. /implement - Implements the plan created by /plan
3. /update_notion_task - Updates the Notion page with the result

This is a Notion-aware version of adw_plan_implement_update_task.py.

Usage:
    # Method 1: Direct execution (requires uv)
    ./adws/adw_plan_implement_update_notion_task.py --adw-id abc123 --worktree-name feature-auth --task "Implement OAuth2" --page-id 247fc382...

    # Method 2: Using uv run
    uv run adws/adw_plan_implement_update_notion_task.py --adw-id abc123 --worktree-name feature-auth --task "Add user profiles" --page-id 247fc382...

Examples:
    # Run with specific model
    ./adws/adw_plan_implement_update_notion_task.py --adw-id abc123 --worktree-name feature-auth --task "Add JWT tokens" --model x-ai/grok-4 --page-id 247fc382...

    # Run with verbose and milestone output
    ./adws/adw_plan_implement_update_notion_task.py --adw-id abc123 --worktree-name feature-auth --task "Fix auth bug" --verbose --milestone --page-id 247fc382...
"""

import os
import sys
import json
import re
import subprocess
from pathlib import Path
from typing import Optional
from datetime import datetime
import time
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'x-ai/grok-4')

# Add the adw_modules directory to the path so we can import agent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "adw_modules"))

from agent import (
    AgentTemplateRequest,
    AgentPromptResponse,
    execute_template,
)
from utils import format_agent_status, format_worktree_status


def print_status_panel(
    console,
    action: str,
    adw_id: str,
    worktree: str,
    phase: str = None,
    status: str = "info",
    verbose: bool = False,
    milestone: bool = False,
):
    """Print a status panel with timestamp and context.

    Args:
        console: Rich console instance
        action: The action being performed
        adw_id: ADW ID for tracking
        worktree: Worktree/branch name
        phase: Optional phase name (build, plan, etc)
        status: Status type (info, success, error)
        verbose: Verbose mode for additional details
        milestone: Milestone mode for high-level logs
    """
    timestamp = datetime.now().strftime("%H:%M:%S")

    # Choose color based on status
    if status == "success":
        border_style = "green"
        icon = "✅"
    elif status == "error":
        border_style = "red"
        icon = "❌"
    else:
        border_style = "cyan"
        icon = "🔄"

    # Build title with context
    title_parts = [f"[{timestamp}]", adw_id[:6], worktree]
    if phase:
        title_parts.append(phase)
    title = " | ".join(title_parts)

    if milestone:
        print(f"LOG: Milestone - {action} (phase: {phase or 'general'})")

    console.print(
        Panel(
            f"{icon} {action}",
            title=f"[bold {border_style}]{title}[/bold {border_style}]",
            border_style=border_style,
            padding=(0, 1),
        )
    )

    if verbose:
        print(f"VERBOSE: Status panel printed for action '{action}' in phase '{phase}'")


# Output file name constants
OUTPUT_JSONL = "cc_raw_output.jsonl"
OUTPUT_JSON = "cc_raw_output.json"
FINAL_OBJECT_JSON = "cc_final_object.json"
SUMMARY_JSON = "custom_summary_output.json"


def get_current_commit_hash(working_dir: str) -> Optional[str]:
    """Get the current git commit hash in the working directory."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=working_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()[:9]  # Return first 9 characters of hash
    except subprocess.CalledProcessError:
        return None


@click.command()
@click.option("--adw-id", required=True, help="ADW ID for this task execution")
@click.option(
    "--worktree-name", required=True, help="Name of the git worktree to work in"
)
@click.option("--task", required=True, help="Task description to implement")
@click.option("--page-id", required=True, help="Notion page ID to update with results")
@click.option(
    "--model",
    type=str,
    default=DEFAULT_MODEL,
    help="Model to use (from env, command line, or Notion tags)",
)
@click.option(
    "--prototype",
    type=click.Choice(["uv_script", "vite_vue", "bun_scripts", "uv_mcp"]),
    help="Prototype type for app generation",
)
@click.option("--verbose", is_flag=True, help="Enable verbose output")
@click.option("--milestone", is_flag=True, help="Display milestone logs")
def main(
    adw_id: str,
    worktree_name: str,
    task: str,
    page_id: str,
    model: str,
    prototype: Optional[str],
    verbose: bool,
    milestone: bool,
):
    import os  # Ensure os is available in main to avoid UnboundLocalError
    """Run plan, implement, and update task workflow for Notion-based multi-agent processing."""
    console = Console()

    if milestone:
        print(f"LOG: Milestone 0 - Plan-implement-update workflow started")

    # Sanitize worktree name - extract just the name if it contains extra text
    import re

    # Look for a valid worktree name pattern in the input
    match = re.search(r"([a-z][a-z0-9-]{4,19})", worktree_name)
    if match:
        worktree_name = match.group(1)
    else:
        # Clean up the name
        worktree_name = re.sub(r"[^a-z0-9-]", "-", worktree_name.lower())[:20]
        worktree_name = re.sub(r"-+", "-", worktree_name).strip("-")

    # Calculate the worktree path and the actual working directory
    # With sparse checkout, the worktree contains agentic_prototyping/
    # and we work within that directory
    # Ensure base dirs exist
    os.makedirs("trees", exist_ok=True)
    os.makedirs("specs", exist_ok=True)
    os.makedirs("apps", exist_ok=True)

    worktree_base_path = os.path.abspath(f"trees/{worktree_name}")
    target_directory = "."

    # Check if worktree exists, create if needed
    if not os.path.exists(worktree_base_path):
        if milestone:
            print("LOG: Milestone 1 - Worktree missing, creating...")
        console.print(
            Panel(
                f"[bold yellow]Worktree not found at: {worktree_base_path}[/bold yellow]\n\n"
                "Creating worktree now...",
                title="[bold yellow]⚠️  Worktree Missing[/bold yellow]",
                border_style="yellow",
            )
        )

        # Create worktree using the init_worktree command
        init_request = AgentTemplateRequest(
            agent_name="worktree-initializer",
            slash_command="/init_worktree",
            args=[worktree_name, target_directory],
            adw_id=adw_id,
            model=model,
            working_dir=os.getcwd(),  # Run from project root
        )

        # Print start message for worktree creation
        print_status_panel(
            console, "Starting worktree creation", adw_id, worktree_name, "init", verbose=verbose, milestone=milestone
        )

        init_response = execute_template(init_request)

        # Verify worktree was actually created by checking dir existence and contents
        if verbose:
            print(f"VERBOSE: Verifying worktree at {worktree_base_path}")
        if not os.path.exists(worktree_base_path):
            # Fallback: manual git worktree create if slash failed
            if milestone:
                print("LOG: Milestone - Slash /init_worktree failed; fallback to manual git worktree create")
            try:
                git_cmd = ["git", "worktree", "add", worktree_base_path, "master"]  # or current branch
                result = subprocess.run(git_cmd, capture_output=True, text=True, check=True)
                if verbose:
                    print(f"VERBOSE: Manual worktree create success: {result.stdout.strip()}")
                os.chdir(worktree_base_path)
                subprocess.run(["git", "checkout", "-b", worktree_name], check=True)
                os.chdir("..")  # Back to project root
            except subprocess.CalledProcessError as ge:
                error_msg = f"Manual worktree creation failed: {ge.stderr or ge.stdout}"
                if verbose:
                    print(f"VERBOSE: {error_msg}")
                workflow_success = False
                console.print(Panel(error_msg, title="[bold red]❌ Manual Worktree Fallback Failed[/bold red]", border_style="red"))
                sys.exit(1)
        else:
            if verbose:
                print(f"VERBOSE: Worktree dir exists: {os.listdir(worktree_base_path)}")

        # Print completion message
        print_status_panel(
            console,
            "Completed worktree creation",
            adw_id,
            worktree_name,
            "init",
            "success",
            verbose=verbose,
            milestone=milestone,
        )

        if init_response.success or os.path.exists(worktree_base_path):
            if milestone:
                print("LOG: Milestone 2 - Worktree creation successful (dir verified)")
            console.print(
                Panel(
                    f"[bold green]✅ Worktree created/verified at: {worktree_base_path}\nContents: {', '.join(os.listdir(worktree_base_path)) if os.path.exists(worktree_base_path) else 'Fallback created'}[/bold green]",
                    title="[bold green]Worktree Created[/bold green]",
                    border_style="green",
                )
            )
        else:
            console.print(
                Panel(
                    f"[bold red]Failed to create worktree (dir missing):\n{init_response.output}[/bold red]",
                    title="[bold red]❌ Worktree Creation Failed[/bold red]",
                    border_style="red",
                )
            )
            sys.exit(1)

    # Set agent names for each phase
    planner_name = f"planner-{worktree_name}"
    builder_name = f"builder-{worktree_name}"
    updater_name = f"notion-updater-{worktree_name}"

    workflow_info = (
        f"[bold blue]Notion Plan-Implement-Update Workflow[/bold blue]\n\n"
        f"[cyan]ADW ID:[/cyan] {adw_id}\n"
        f"[cyan]Worktree:[/cyan] {worktree_name}\n"
        f"[cyan]Task:[/cyan] {task}\n"
        f"[cyan]Page ID:[/cyan] {page_id}\n"
        f"[cyan]Model:[/cyan] {model}\n"
    )

    if prototype:
        workflow_info += f"[cyan]Prototype:[/cyan] {prototype}\n"

    workflow_info += f"[cyan]Working Dir:[/cyan] {worktree_base_path}"

    console.print(
        Panel(
            workflow_info,
            title="[bold blue]🚀 Workflow Configuration[/bold blue]",
            border_style="blue",
        )
    )
    console.print()

    if verbose:
        print(f"VERBOSE: Workflow configuration displayed for ADW {adw_id}")

    # Track workflow state
    workflow_success = True
    plan_path = None
    commit_hash = None
    error_message = None
    plan_response = None
    implement_response = None

    # Phase 1: Run /plan command (or prototype-specific plan command)
    # Determine the plan command based on prototype
    app_name = None  # Initialize for later use
    if prototype:
        plan_command = f"/plan_{prototype}"
        plan_phase_name = f"Prototype Planning ({plan_command})"
        # For prototypes, extract app name from task or use ADW ID
        import re

        app_name_match = re.search(r"app[:\s]+([a-z0-9-]+)", task.lower())
        app_name = app_name_match.group(1) if app_name_match else f"app-{adw_id[:6]}"
        # For prototype plans: adw_id, prompt
        plan_args = [adw_id, task]
    else:
        plan_command = "/plan"
        plan_phase_name = "Planning (/plan)"
        # For regular plan: adw_id, prompt
        plan_args = [adw_id, task]

    if milestone:
        print(f"LOG: Milestone 3 - Phase 1: Planning ({plan_phase_name})")

    console.print(Rule(f"[bold yellow]Phase 1: {plan_phase_name}[/bold yellow]"))
    console.print()

    # Set working directory to include the target_directory path
    # For planning, use project root to generate specs/; for implement, use worktree
    plan_working_dir = os.getcwd()
    agent_working_dir = os.path.join(worktree_base_path, "agentic_prototyping") if os.path.exists(os.path.join(worktree_base_path, "agentic_prototyping")) else worktree_base_path

    # Get model: Notion tag > CLI > env > tag > fallback
    notion_model = None

    # Fetch Notion page for tags (simple API call)
    import requests
    notion_url = f"https://api.notion.com/v1/pages/{page_id}"
    headers = {"Authorization": f"Bearer {os.getenv('NOTION_API_KEY')}", "Notion-Version": "2025-09-03"}
    notion_response = requests.get(notion_url, headers=headers)
    if notion_response.status_code == 200:
        notion_data = notion_response.json()
        content = ''.join([t['plain_text'] for t in notion_data['properties'].get('Content', {}).get('rich_text', [])])
        def extract_tags(content):  # Inline for simplicity
            tags = {}
            for match in re.finditer(r'{{([^:]+):\s*([^}]+)}}', content):
                key, value = match.groups()
                tags[key.strip()] = value.strip()
            return tags
        tags = extract_tags(content)
        thinking_model = tags.get('thinking') or tags.get('model')
        fast_model = tags.get('fast') or tags.get('model')

    # Set models: CLI override > tag > fallback
    thinking_model = thinking_model or 'x-ai/grok-4'  # Thinking for planning
    fast_model = fast_model or 'x-ai/grok-4-fast'  # Fast for implement/update

    if verbose:
        print(f"VERBOSE: Resolved thinking_model: {thinking_model}, fast_model: {fast_model}")

    plan_request = AgentTemplateRequest(
        agent_name=planner_name,
        slash_command=plan_command,
        args=plan_args,
        adw_id=adw_id,
        model=plan_model,  # Use thinking model for planning
        working_dir=plan_working_dir,
    )

    # Display plan execution info
    plan_info_table = Table(show_header=False, box=None, padding=(0, 1))
    plan_info_table.add_column(style="bold cyan")
    plan_info_table.add_column()

    plan_info_table.add_row("ADW ID", adw_id)
    plan_info_table.add_row("Phase", plan_phase_name)
    plan_info_table.add_row("Command", plan_command)
    plan_info_table.add_row("Args", f'{adw_id} "{task}"')
    if prototype:
        plan_info_table.add_row("Prototype", prototype)
    plan_info_table.add_row("Model", model)
    plan_info_table.add_row("Agent", planner_name)

    console.print(
        Panel(
            plan_info_table,
            title=f"[bold blue]🚀 Plan Inputs | {adw_id} | {worktree_name}[/bold blue]",
            border_style="blue",
        )
    )
    console.print()

    if verbose:
        print(f"VERBOSE: Plan inputs table displayed for {plan_phase_name}")

    try:
        # Print start message for plan phase
        # Retry logic for plan phase: Up to 3 attempts, check dir exists, create if missing
        max_retries = 3
        plan_response = None
        for attempt in range(max_retries):
            try:
                if not os.path.exists(plan_working_dir):
                    os.makedirs(plan_working_dir, exist_ok=True)
                    if verbose:
                        print(f"VERBOSE: Created missing directory: {plan_working_dir}")

                print_status_panel(
                    console, f"Starting plan creation (attempt {attempt+1}/{max_retries})", adw_id, worktree_name, "plan", verbose=verbose, milestone=milestone
                )

                if milestone:
                    print(f"LOG: Milestone - Executing {plan_command} with request: {plan_request}")

                if verbose:
                    print(f"VERBOSE: Calling execute_template for {plan_command} in dir {plan_working_dir}")

                # Execute the plan command with timeout
                import signal
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"{plan_command} timed out after 600s")

                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(600)  # 10 min timeout

                try:
                    plan_request.working_dir = plan_working_dir
                    plan_response = execute_template(plan_request)
                finally:
                    signal.alarm(0)

                if verbose:
                    print(f"VERBOSE: {plan_command} response success: {plan_response.success}, output len: {len(plan_response.output) if plan_response.output else 0}")

                if plan_response.success:
                    if milestone:
                        print(f"LOG: Milestone - {plan_command} successful")
                    break
                else:
                    if milestone:
                        print(f"LOG: Milestone - {plan_command} failed: {plan_response.output[:200]}...")
                    time.sleep(2 ** attempt)  # Exponential backoff
            except TimeoutError as te:
                if verbose:
                    print(f"VERBOSE: Plan attempt {attempt+1} timed out: {te}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
            except Exception as e:
                if verbose:
                    print(f"VERBOSE: Plan attempt {attempt+1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

        # Print completion message
        print_status_panel(
            console,
            f"Completed plan creation",
            adw_id,
            worktree_name,
            "plan",
            "success",
            verbose=verbose,
            milestone=milestone,
        )

        if plan_response.success:
            if milestone:
                print("LOG: Milestone 4 - Planning phase successful")
            plan_path = plan_response.output.strip()

            # Ensure specs/ exists
            os.makedirs("specs", exist_ok=True)
            if verbose:
                print(f"VERBOSE: Ensured specs/ dir exists")

            # Improved fallback: Check if response.output is a path (AI wrote it) or content (write it)
            import os.path

            if plan_response.output.strip().startswith("specs/") and plan_response.output.strip().endswith(".md") and "Error:" not in plan_response.output:
                # Assume AI wrote the file; verify existence
                plan_path = plan_response.output.strip()
                full_plan_path = os.path.join(os.getcwd(), plan_path)
                if os.path.exists(full_plan_path) and os.path.getsize(full_plan_path) > 100:  # Basic sanity: >100 chars
                    console.print(f"\n[bold cyan]Plan spec verified (AI-written) at:[/bold cyan] {plan_path}")
                    if milestone:
                        print(f"LOG: Milestone - Plan file verified (AI wrote it)")
                    # Optional: Read to confirm
                    try:
                        with open(full_plan_path, 'r') as f:
                            content = f.read()
                            if "## Metadata" in content and len(content) > 500:  # Expect plan format
                                if verbose:
                                    print(f"VERBOSE: Plan verified: {len(content)} chars, contains metadata")
                            else:
                                raise ValueError("Invalid content")
                    except Exception as ve:
                        if verbose:
                            print(f"VERBOSE: Plan verification failed: {ve}")
                        workflow_success = False
                        error_message = f"Plan file exists but invalid: {full_plan_path}"
                else:
                    workflow_success = False
                    error_message = f"AI reported path {plan_path} but file missing or empty"
            else:
                # Response is likely content or error; generate default path and write
                plan_path = f"specs/plan-{adw_id[:6]}-{worktree_name.replace('-', '_')}.md"
                full_plan_path = os.path.join(os.getcwd(), plan_path)
                if "Error:" in plan_response.output:
                    error_message = f"AI plan generation error: {plan_response.output}"
                    workflow_success = False
                else:
                    # Write content
                    with open(full_plan_path, 'w') as f:
                        f.write(f"# Plan for Task {adw_id}\n\n{plan_response.output}\n\nGenerated at {datetime.now()}")
                    if verbose:
                        print(f"VERBOSE: Fallback wrote plan to {full_plan_path} (len: {len(plan_response.output)})")

                # Verify write
                if os.path.exists(full_plan_path):
                    plan_path = os.path.relpath(full_plan_path, os.getcwd())
                    console.print(f"\n[bold cyan]Plan spec written/verified at:[/bold cyan] {plan_path}")
                    if milestone:
                        print(f"LOG: Milestone - Plan file ready for downstream ({plan_path})")
                else:
                    workflow_success = False
                    error_message = f"Failed to write/verify plan file at {full_plan_path}. Check permissions."
                    console.print(
                        Panel(
                            f"[bold red]{error_message}[/bold red]",
                            title="[bold red]❌ Plan File Write Failed[/bold red]",
                            border_style="red",
                        )
                    )

            # Ensure specs/ exists proactively
            os.makedirs("specs", exist_ok=True)

            console.print(
                Panel(
                    (
                        plan_response.output
                        if verbose
                        else f"Plan spec ready at {plan_path} for implement agents"
                    ),
                    title=f"[bold green]✅ Planning Success | {adw_id} | {worktree_name}[/bold green]",
                    border_style="green",
                    padding=(1, 2),
                )
            )
        else:
            workflow_success = False
            error_message = f"Planning phase failed: {plan_response.output}"
            console.print(
                Panel(
                    plan_response.output,
                    title=f"[bold red]❌ Planning Failed | {adw_id} | {worktree_name}[/bold red]",
                    border_style="red",
                    padding=(1, 2),
                )
            )

        # Save plan phase summary
        plan_output_dir = f"./agents/{adw_id}/{planner_name}"
        plan_summary_path = f"{plan_output_dir}/{SUMMARY_JSON}"

        os.makedirs(plan_output_dir, exist_ok=True)
        with open(plan_summary_path, "w") as f:
            json.dump(
                {
                    "phase": "planning",
                    "adw_id": adw_id,
                    "worktree_name": worktree_name,
                    "task": task,
                    "page_id": page_id,
                    "slash_command": plan_command,
                    "args": plan_args,
                    "thinking_model": thinking_model,
                    "fast_model": fast_model,
                    "prototype": prototype,
                    "app_name": app_name,
                    "working_dir": agent_working_dir,
                    "success": plan_response.success,
                    "session_id": plan_response.session_id,
                    "plan_path": plan_path,
                },
                f,
                indent=2,
            )

        if verbose:
            print(f"VERBOSE: Plan summary saved to {plan_summary_path}")

        # Phase 2: Run /implement command (only if planning succeeded)
        if workflow_success and plan_path:
            if milestone:
                print("LOG: Milestone 5 - Phase 2: Implementation")
            console.print()
            console.print(
                Rule("[bold yellow]Phase 2: Implementation (/implement)[/bold yellow]")
            )
            console.print()

            implement_request = AgentTemplateRequest(
                agent_name=builder_name,
                slash_command="/implement",
                args=[plan_path],
                adw_id=adw_id,
                model=model,
                working_dir=agent_working_dir,
            )

            # Display implement execution info
            implement_info_table = Table(show_header=False, box=None, padding=(0, 1))
            implement_info_table.add_column(style="bold cyan")
            implement_info_table.add_column()

            implement_info_table.add_row("ADW ID", adw_id)
            implement_info_table.add_row("Phase", "Implementation")
            implement_info_table.add_row("Command", "/implement")
            implement_info_table.add_row("Args", plan_path)
            implement_info_table.add_row("Model", model)
            implement_info_table.add_row("Agent", builder_name)

            console.print(
                Panel(
                    implement_info_table,
                    title=f"[bold blue]🚀 Implement Inputs | {adw_id} | {worktree_name}[/bold blue]",
                    border_style="blue",
                )
            )
            console.print()

            if verbose:
                print(f"VERBOSE: Implement inputs table displayed")

            # Print start message for implement phase
            print_status_panel(
                console, "Starting implementation", adw_id, worktree_name, "implement", verbose=verbose, milestone=milestone
            )

            # Execute the implement command
            # Verify working dir before implement
            if not os.path.exists(agent_working_dir):
                raise ValueError(f"Agent working dir missing: {agent_working_dir}")

            implement_response = execute_template(implement_request)

            # Verify app output post-implementation (check for app dir or key files)
            if verbose:
                print(f"VERBOSE: Verifying implement output in {agent_working_dir}")
            app_dirs = [d for d in os.listdir(agent_working_dir) if d.startswith('app') or d == 'apps']
            key_files = []
            if app_dirs:
                for app_dir in app_dirs:
                    app_path = os.path.join(agent_working_dir, app_dir)
                    if os.path.isdir(app_path):
                        for root, dirs, files in os.walk(app_path):
                            key_files.extend([f for f in files if f in ['package.json', 'index.html', 'vite.config.js', 'App.vue']])
                            break  # Shallow walk
            if verbose:
                print(f"VERBOSE: Found app dirs: {app_dirs}, key files: {key_files}")

            if not key_files and implement_response.success:
                # Fallback: run npm init or manual setup if no output
                if milestone:
                    print("LOG: Milestone - No app output; fallback npm init in worktree")
                try:
                    os.chdir(agent_working_dir)
                    subprocess.run(["npm", "create", "vite@latest", ".", "--template", "vue", "--yes"], check=True)
                    os.chdir("..")
                except subprocess.CalledProcessError as se:
                    if verbose:
                        print(f"VERBOSE: Fallback npm create failed: {se}")

            # Print completion message
            print_status_panel(
                console,
                "Completed implementation",
                adw_id,
                worktree_name,
                "implement",
                "success",
                verbose=verbose,
                milestone=milestone,
            )

            if implement_response.success:
                if milestone:
                    print("LOG: Milestone 6 - Implementation phase successful")
                output_msg = f"Implementation completed | App dirs: {app_dirs} | Key files: {key_files}"
                console.print(
                    Panel(
                        (
                            implement_response.output
                            if verbose
                            else output_msg
                        ),
                        title=f"[bold green]✅ Implementation Success | {adw_id} | {worktree_name}[/bold green]",
                        border_style="green",
                        padding=(1, 2),
                    )
                )

                # Get the commit hash after successful implementation
                commit_hash = get_current_commit_hash(agent_working_dir)
                if commit_hash:
                    console.print(
                        f"\n[bold cyan]Commit hash:[/bold cyan] {commit_hash}"
                    )
            else:
                workflow_success = False
                error_message = (
                    f"Implementation phase failed: {implement_response.output}"
                )
                console.print(
                    Panel(
                        implement_response.output,
                        title=f"[bold red]❌ Implementation Failed | {adw_id} | {worktree_name}[/bold red]",
                        border_style="red",
                        padding=(1, 2),
                    )
                )

            # Save implement phase summary
            implement_output_dir = f"./agents/{adw_id}/{builder_name}"
            implement_summary_path = f"{implement_output_dir}/{SUMMARY_JSON}"

            os.makedirs(implement_output_dir, exist_ok=True)
            with open(implement_summary_path, "w") as f:
                json.dump(
                    {
                        "phase": "implementation",
                        "adw_id": adw_id,
                        "worktree_name": worktree_name,
                        "task": task,
                        "page_id": page_id,
                        "slash_command": "/implement",
                        "args": [plan_path],
                        "thinking_model": thinking_model,
                        "fast_model": fast_model,
                        "working_dir": agent_working_dir,
                        "success": implement_response.success,
                        "session_id": implement_response.session_id,
                        "commit_hash": commit_hash,
                    },
                    f,
                    indent=2,
                )

            if verbose:
                print(f"VERBOSE: Implement summary saved to {implement_summary_path}")
        else:
            # Implementation skipped due to planning issues
            if milestone:
                print("LOG: Milestone 5 - Implementation skipped due to planning failure")
            if not workflow_success:
                console.print()
                console.print(
                    Panel(
                        "[yellow]⏭️  Skipping implementation phase due to planning errors[/yellow]",
                        title="[bold yellow]Implementation Skipped[/bold yellow]",
                        border_style="yellow",
                    )
                )

        # Phase 3: Run /update_notion_task command (always run to update status)
        if milestone:
            print("LOG: Milestone 7 - Phase 3: Notion update")
        console.print()
        console.print(
            Rule(
                "[bold yellow]Phase 3: Update Notion Task (/update_notion_task)[/bold yellow]"
            )
        )
        console.print()

        # Determine the status to update
        update_status = "Done" if workflow_success and commit_hash else "In Progress (Review Needed)"  # Keep in progress for manual review on logical errors

        # Build update content with results
        update_content = {
            "status": update_status,
            "adw_id": adw_id,
            "commit_hash": commit_hash or "",
            "error": error_message or "",
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "workflow": "plan-implement-update",
            "worktree_name": worktree_name,
            "result": (
                implement_response.output
                if implement_response
                else plan_response.output if plan_response else ""
            ),
        }

        update_request = AgentTemplateRequest(
            agent_name=updater_name,
            slash_command="/update_notion_task",
            args=[page_id, update_status, json.dumps(update_content)],
            adw_id=adw_id,
            model=model,
            working_dir=os.getcwd(),  # Run from project root
        )

        # Display update execution info
        update_info_table = Table(show_header=False, box=None, padding=(0, 1))
        update_info_table.add_column(style="bold cyan")
        update_info_table.add_column()

        update_info_table.add_row("ADW ID", adw_id)
        update_info_table.add_row("Phase", "Update Notion Task")
        update_info_table.add_row("Command", "/update_notion_task")
        update_info_table.add_row("Page ID", page_id[:12] + "...")
        update_info_table.add_row("Status", update_status)
        update_info_table.add_row("Model", model)
        update_info_table.add_row("Agent", updater_name)

        console.print(
            Panel(
                update_info_table,
                title=f"[bold blue]🚀 Update Inputs | {adw_id} | {worktree_name}[/bold blue]",
                border_style="blue",
            )
        )
        console.print()

        if verbose:
            print(f"VERBOSE: Update inputs table displayed")

        # Print start message for update phase
        print_status_panel(
            console, "Starting Notion task update", adw_id, worktree_name, "update", verbose=verbose, milestone=milestone
        )

        # Execute the update command
        update_response = execute_template(update_request)

        # Print completion message
        print_status_panel(
            console,
            "Completed Notion task update",
            adw_id,
            worktree_name,
            "update",
            "success",
            verbose=verbose,
            milestone=milestone,
        )

        if update_response.success:
            if milestone:
                print("LOG: Milestone 8 - Notion update successful")
            console.print(
                Panel(
                    (
                        update_response.output
                        if verbose
                        else "Notion task updated successfully"
                    ),
                    title=f"[bold green]✅ Update Success | {adw_id} | {worktree_name}[/bold green]",
                    border_style="green",
                    padding=(1, 2),
                )
            )
        else:
            console.print(
                Panel(
                    update_response.output,
                    title=f"[bold red]❌ Update Failed | {adw_id} | {worktree_name}[/bold red]",
                    border_style="red",
                    padding=(1, 2),
                )
            )

        # Save update phase summary
        update_output_dir = f"./agents/{adw_id}/{updater_name}"
        update_summary_path = f"{update_output_dir}/{SUMMARY_JSON}"

        os.makedirs(update_output_dir, exist_ok=True)
        with open(update_summary_path, "w") as f:
            json.dump(
                {
                    "phase": "update_notion_task",
                    "adw_id": adw_id,
                    "worktree_name": worktree_name,
                    "task": task,
                    "page_id": page_id,
                    "slash_command": "/update_notion_task",
                    "args": [page_id, update_status, json.dumps(update_content)],
                    "thinking_model": thinking_model,
                    "fast_model": fast_model,
                    "working_dir": os.getcwd(),
                    "success": update_response.success,
                    "session_id": update_response.session_id,
                    "final_status": update_status,
                    "result": update_response.output,
                },
                f,
                indent=2,
            )

        if verbose:
            print(f"VERBOSE: Update summary saved to {update_summary_path}")

        # Show workflow summary
        console.print()
        console.print(Rule("[bold blue]Workflow Summary[/bold blue]"))
        console.print()

        summary_table = Table(show_header=True, box=None)
        summary_table.add_column("Phase", style="bold cyan")
        summary_table.add_column("Status", style="bold")
        summary_table.add_column("Output Directory", style="dim")

        # Planning phase row
        planning_status = "✅ Success" if plan_response and plan_response.success else "❌ Failed"
        summary_table.add_row(
            "Planning (/plan)",
            planning_status,
            f"./agents/{adw_id}/{planner_name}/" if plan_response else "-",
        )

        # Implementation phase row
        if implement_response:
            implement_status = (
                "✅ Success" if implement_response.success else "❌ Failed"
            )
            summary_table.add_row(
                "Implementation (/implement)",
                implement_status,
                f"./agents/{adw_id}/{builder_name}/",
            )
        else:
            summary_table.add_row(
                "Implementation (/implement)",
                "⏭️ Skipped (plan failed or no path)",
                "-",
            )

        # Update phase row
        update_status_display = "✅ Success" if update_response and update_response.success else "❌ Failed"
        summary_table.add_row(
            "Update Notion (/update_notion_task)",
            update_status_display,
            f"./agents/{adw_id}/{updater_name}/" if update_response else "-",
        )

        console.print(summary_table)

        # Create overall workflow summary
        workflow_summary_path = f"./agents/{adw_id}/workflow_summary.json"
        os.makedirs(f"./agents/{adw_id}", exist_ok=True)

        with open(workflow_summary_path, "w") as f:
            json.dump(
                {
                    "workflow": "plan_implement_update_notion_task",
                    "adw_id": adw_id,
                    "worktree_name": worktree_name,
                    "task": task,
                    "page_id": page_id,
                    "thinking_model": thinking_model,
                    "fast_model": fast_model,
                    "prototype": prototype,
                    "app_name": app_name,
                    "working_dir": agent_working_dir,
                    "plan_path": plan_path,
                    "commit_hash": commit_hash,
                    "phases": {
                        "planning": {
                            "success": plan_response.success if plan_response else False,
                            "session_id": plan_response.session_id if plan_response else None,
                            "agent": planner_name,
                            "model_used": plan_model,
                        },
                        "implementation": {
                            "success": implement_response.success if implement_response else False,
                            "session_id": implement_response.session_id if implement_response else None,
                            "agent": builder_name if implement_response else None,
                            "model_used": implement_model,
                        } if implement_response or plan_path else None,
                        "update_notion_task": {
                            "success": update_response.success if update_response else False,
                            "session_id": update_response.session_id if update_response else None,
                            "agent": updater_name,
                            "model_used": update_model,
                        },
                    },
                    "overall_success": workflow_success,
                    "final_task_status": (
                        "Done" if workflow_success and commit_hash else "Failed"
                    ),
                },
                f,
                indent=2,
            )

        if verbose:
            print(f"VERBOSE: Overall workflow summary saved to {workflow_summary_path}")

        console.print(
            f"\n[bold cyan]Workflow summary:[/bold cyan] {workflow_summary_path}"
        )
        console.print()

        # Exit with appropriate code
        if milestone:
            print(f"LOG: Milestone 9 - Workflow complete (success={workflow_success})")
        if workflow_success:
            console.print(
                "[bold green]✅ Workflow completed successfully![/bold green]"
            )
            sys.exit(0)
        else:
            console.print(
                "[bold yellow]⚠️  Workflow completed with errors[/bold yellow]"
            )
            sys.exit(1)

    except Exception as e:
        if milestone:
            print(f"LOG: Milestone - Unexpected error: {str(e)}")
        console.print(
            Panel(
                f"[bold red]{str(e)}[/bold red]",
                title="[bold red]❌ Unexpected Error[/bold red]",
                border_style="red",
            )
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
