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
Run build and update task workflow for Notion-based multi-agent task processing.

This script runs two slash commands in sequence:
1. /build - Directly implements the task without planning
2. /update_notion_task - Updates the Notion page with the result

This is a Notion-aware version of adw_build_update_task.py.

Usage:
    # Method 1: Direct execution (requires uv)
    ./adws/adw_build_update_notion_task.py --adw-id abc123 --worktree-name feature-auth --task "Fix typo in README" --page-id 247fc382...

    # Method 2: Using uv run
    uv run adws/adw_build_update_notion_task.py --adw-id abc123 --worktree-name feature-auth --task "Add logging statement" --page-id 247fc382...

Examples:
    # Run with specific model
    ./adws/adw_build_update_notion_task.py --adw-id abc123 --worktree-name feature-auth --task "Update version" --model x-ai/grok-4 --page-id 247fc382...

    # Run with verbose and milestone output
    ./adws/adw_build_update_notion_task.py --adw-id abc123 --worktree-name feature-auth --task "Fix import" --verbose --milestone --page-id 247fc382...
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Optional
from datetime import datetime
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule

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
    type=click.Choice(["x-ai/grok-4-fast", "x-ai/grok-4"]),
    default="x-ai/grok-4-fast",
    help="OpenRouter model to use",
)
@click.option("--verbose", is_flag=True, help="Enable verbose output")
@click.option("--milestone", is_flag=True, help="Display milestone logs")
def main(
    adw_id: str,
    worktree_name: str,
    task: str,
    page_id: str,
    model: str,
    verbose: bool,
    milestone: bool,
):
    """Run build and update task workflow for Notion-based multi-agent processing."""
    console = Console()

    if milestone:
        print(f"LOG: Milestone 0 - Build-update workflow started")

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
    worktree_base_path = os.path.abspath(f"trees/{worktree_name}")
    target_directory = "agentic_prototyping"

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

        if init_response.success:
            if milestone:
                print("LOG: Milestone 2 - Worktree creation successful")
            console.print(
                Panel(
                    f"[bold green]✅ Worktree created successfully at: {worktree_base_path}[/bold green]",
                    title="[bold green]Worktree Created[/bold green]",
                    border_style="green",
                )
            )
        else:
            console.print(
                Panel(
                    f"[bold red]Failed to create worktree:\n{init_response.output}[/bold red]",
                    title="[bold red]❌ Worktree Creation Failed[/bold red]",
                    border_style="red",
                )
            )
            sys.exit(1)

    # Set agent names for each phase
    builder_name = f"builder-{worktree_name}"
    updater_name = f"notion-updater-{worktree_name}"

    console.print(
        Panel(
            f"[bold blue]Notion Build-Update Workflow[/bold blue]\n\n"
            f"[cyan]ADW ID:[/cyan] {adw_id}\n"
            f"[cyan]Worktree:[/cyan] {worktree_name}\n"
            f"[cyan]Task:[/cyan] {task}\n"
            f"[cyan]Page ID:[/cyan] {page_id}\n"
            f"[cyan]Model:[/cyan] {model}\n"
            f"[cyan]Working Dir:[/cyan] {os.path.join(worktree_base_path, target_directory)}",
            title="[bold blue]🚀 Workflow Configuration[/bold blue]",
            border_style="blue",
        )
    )
    console.print()

    if verbose:
        print(f"VERBOSE: Workflow configuration displayed for ADW {adw_id}")

    # Track workflow state
    workflow_success = True
    commit_hash = None
    error_message = None

    # Phase 1: Run /build command
    console.print(Rule("[bold yellow]Phase 1: Build (/build)[/bold yellow]"))
    console.print()

    # Set working directory to include the target_directory path
    agent_working_dir = os.path.join(worktree_base_path, target_directory)

    build_request = AgentTemplateRequest(
        agent_name=builder_name,
        slash_command="/build",
        args=[adw_id, task],
        adw_id=adw_id,
        model=model,
        working_dir=agent_working_dir,
    )

    # Display build execution info
    build_info_table = Table(show_header=False, box=None, padding=(0, 1))
    build_info_table.add_column(style="bold cyan")
    build_info_table.add_column()

    build_info_table.add_row("ADW ID", adw_id)
    build_info_table.add_row("Phase", "Build")
    build_info_table.add_row("Command", "/build")
    build_info_table.add_row("Args", f'{adw_id} "{task}"')
    build_info_table.add_row("Model", model)
    build_info_table.add_row("Agent", builder_name)

    console.print(
        Panel(
            build_info_table,
            title=f"[bold blue]🚀 Build Inputs | {adw_id} | {worktree_name}[/bold blue]",
            border_style="blue",
        )
    )
    console.print()

    if verbose:
        print(f"VERBOSE: Build inputs table displayed")

    try:
        # Print start message for build phase
        print_status_panel(
            console, "Starting build process", adw_id, worktree_name, "build", verbose=verbose, milestone=milestone
        )

        # Execute the build command
        build_response = execute_template(build_request)

        # Print completion message
        print_status_panel(
            console,
            "Completed build process",
            adw_id,
            worktree_name,
            "build",
            "success",
            verbose=verbose,
            milestone=milestone,
        )

        if build_response.success:
            if milestone:
                print("LOG: Milestone 3 - Build phase successful")
            console.print(
                Panel(
                    (
                        build_response.output
                        if verbose
                        else "Build completed successfully"
                    ),
                    title=f"[bold green]✅ Build Success | {adw_id} | {worktree_name}[/bold green]",
                    border_style="green",
                    padding=(1, 2),
                )
            )

            # Get the commit hash after successful build
            commit_hash = get_current_commit_hash(agent_working_dir)
            if commit_hash:
                console.print(f"\n[bold cyan]Commit hash:[/bold cyan] {commit_hash}")
        else:
            workflow_success = False
            error_message = f"Build phase failed: {build_response.output}"
            console.print(
                Panel(
                    build_response.output,
                    title=f"[bold red]❌ Build Failed | {adw_id} | {worktree_name}[/bold red]",
                    border_style="red",
                    padding=(1, 2),
                )
            )

        # Save build phase summary
        build_output_dir = f"./agents/{adw_id}/{builder_name}"
        build_summary_path = f"{build_output_dir}/{SUMMARY_JSON}"

        os.makedirs(build_output_dir, exist_ok=True)
        with open(build_summary_path, "w") as f:
            json.dump(
                {
                    "phase": "build",
                    "adw_id": adw_id,
                    "worktree_name": worktree_name,
                    "task": task,
                    "page_id": page_id,
                    "slash_command": "/build",
                    "args": [adw_id, task],
                    "model": model,
                    "working_dir": agent_working_dir,
                    "success": build_response.success,
                    "session_id": build_response.session_id,
                    "commit_hash": commit_hash,
                },
                f,
                indent=2,
            )

        if verbose:
            print(f"VERBOSE: Build summary saved to {build_summary_path}")

        # Phase 2: Run /update_notion_task command (always run to update status)
        console.print()
        console.print(
            Rule(
                "[bold yellow]Phase 2: Update Notion Task (/update_notion_task)[/bold yellow]"
            )
        )
        console.print()

        # Determine the status to update
        update_status = "Done" if workflow_success and commit_hash else "Failed"

        # Build update content with results
        update_content = {
            "status": update_status,
            "adw_id": adw_id,
            "commit_hash": commit_hash or "",
            "error": error_message or "",
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "workflow": "build-update",
            "worktree_name": worktree_name,
            "result": build_response.output,
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
                print("LOG: Milestone 4 - Notion update successful")
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
                    "model": model,
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

        # Build phase row
        build_status_display = "✅ Success" if build_response.success else "❌ Failed"
        summary_table.add_row(
            "Build (/build)",
            build_status_display,
            f"./agents/{adw_id}/{builder_name}/",
        )

        # Update phase row
        update_status_display = "✅ Success" if update_response.success else "❌ Failed"
        summary_table.add_row(
            "Update Notion (/update_notion_task)",
            update_status_display,
            f"./agents/{adw_id}/{updater_name}/",
        )

        console.print(summary_table)

        # Create overall workflow summary
        workflow_summary_path = f"./agents/{adw_id}/workflow_summary.json"
        os.makedirs(f"./agents/{adw_id}", exist_ok=True)

        with open(workflow_summary_path, "w") as f:
            json.dump(
                {
                    "workflow": "build_update_notion_task",
                    "adw_id": adw_id,
                    "worktree_name": worktree_name,
                    "task": task,
                    "page_id": page_id,
                    "model": model,
                    "working_dir": agent_working_dir,
                    "commit_hash": commit_hash,
                    "phases": {
                        "build": {
                            "success": build_response.success,
                            "session_id": build_response.session_id,
                            "agent": builder_name,
                        },
                        "update_notion_task": {
                            "success": update_response.success,
                            "session_id": update_response.session_id,
                            "agent": updater_name,
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
            print(f"LOG: Milestone 5 - Workflow complete (success={workflow_success})")
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
