"""
Data models for the multi-agent task list architecture.

These models define the structure for tasks, worktrees, and workflow states
used throughout the ToDone system.
"""

from typing import List, Optional, Literal, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator

from contextlib import suppress
with suppress(ImportError):
    from dotenv import load_dotenv  # type: ignore
    # Load environment variables
    load_dotenv()


class SystemTag(str, Enum):
    """System-defined tags that control task execution behavior."""

    # Workflow selection tags
    PLAN_IMPLEMENT_UPDATE = "adw_plan_implement_update_task"

    # Model selection tags
    THINKING = "x-ai/grok-4"
    FAST = "x-ai/grok-4-fast"

    @classmethod
    def get_workflow_tags(cls) -> List[str]:
        """Get all workflow-related tags."""
        return [cls.PLAN_IMPLEMENT_UPDATE]

    @classmethod
    def get_model_tags(cls) -> List[str]:
        """Get all model-related tags."""
        return [cls.THINKING, cls.FAST]

    @classmethod
    def extract_model_from_tags(cls, tags: List[str]) -> Optional[str]:
        """Extract the model to use from tags.

        Priority: THINKING > FAST > default (None)
        """
        if cls.THINKING in tags:
            return "x-ai/grok-4"
        elif cls.FAST in tags:
            return "x-ai/grok-4-fast"
        return None

    @classmethod
    def extract_workflow_from_tags(cls, tags: List[str]) -> bool:
        """Check if full workflow should be used based on tags."""
        return cls.PLAN_IMPLEMENT_UPDATE in tags


class Task(BaseModel):
    """Represents a single task in the task list."""

    description: str = Field(..., description="The task description")
    status: Literal["[]", "[⏰]", "[🟡]", "[✅]", "[❌]"] = Field(
        default="[]", description="Current status of the task"
    )
    adw_id: Optional[str] = Field(
        None, description="ADW ID assigned when task is picked up"
    )
    commit_hash: Optional[str] = Field(
        None, description="Git commit hash when task is completed"
    )
    tags: List[str] = Field(
        default_factory=list, description="Optional tags for the task"
    )
    worktree_name: Optional[str] = Field(
        None, description="Associated git worktree name"
    )

    @validator("status")
    def validate_status(cls, v):
        """Ensure status is one of the valid values."""
        valid_statuses = ["[]", "[⏰]", "[🟡]", "[✅]", "[❌]"]
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of {valid_statuses}")
        return v

    def is_eligible_for_pickup(self) -> bool:
        """Check if task can be picked up by an agent."""
        return self.status in ["[]", "[⏰]"]

    def is_completed(self) -> bool:
        """Check if task is in a terminal state."""
        return self.status in ["[✅]", "[❌]"]


class Worktree(BaseModel):
    """Represents a git worktree section in the task list."""

    name: str = Field(..., description="Name of the git worktree")
    tasks: List[Task] = Field(
        default_factory=list, description="Tasks in this worktree"
    )

    def get_eligible_tasks(self) -> List[Task]:
        """Get all tasks eligible for pickup, considering blocking rules."""
        eligible = []

        for i, task in enumerate(self.tasks):
            if task.status == "[]":
                # Non-blocked tasks are always eligible
                eligible.append(task)
            elif task.status == "[⏰]":
                # Blocked tasks are eligible only if all tasks above are successful
                all_above_successful = all(t.status == "[✅]" for t in self.tasks[:i])
                if all_above_successful:
                    eligible.append(task)

        return eligible


class TaskToStart(BaseModel):
    """Task ready to be started by an agent."""

    description: str = Field(..., description="The task description")
    tags: List[str] = Field(
        default_factory=list, description="Optional tags for the task"
    )


class WorktreeTaskGroup(BaseModel):
    """Groups tasks by worktree for processing."""

    worktree_name: str = Field(..., description="Name of the git worktree")
    tasks_to_start: List[TaskToStart] = Field(
        ..., description="Tasks ready to be started in this worktree"
    )


class ProcessTasksResponse(BaseModel):
    """Response from the /process_tasks command."""

    task_groups: List[WorktreeTaskGroup] = Field(
        default_factory=list, description="Tasks grouped by worktree"
    )

    def has_tasks(self) -> bool:
        """Check if there are any tasks to process."""
        return any(len(group.tasks_to_start) > 0 for group in self.task_groups)


class TaskUpdate(BaseModel):
    """Update information for a task after agent processing."""

    adw_id: str = Field(..., description="ADW ID of the task")
    status: Literal["[✅]", "[❌]"] = Field(..., description="Final status of the task")
    commit_hash: Optional[str] = Field(
        None, description="Git commit hash if successful"
    )
    error_message: Optional[str] = Field(None, description="Error message if failed")
    worktree_name: str = Field(..., description="Worktree where task was executed")
    task_description: str = Field(..., description="Original task description")

    @validator("status")
    def validate_final_status(cls, v):
        """Ensure status is a terminal state."""
        if v not in ["[✅]", "[❌]"]:
            raise ValueError("Task update status must be either [✅] or [❌]")
        return v

    @validator("commit_hash")
    def validate_commit_hash(cls, v, values):
        """Ensure commit hash is provided for successful tasks."""
        if values.get("status") == "[✅]" and not v:
            raise ValueError("Commit hash is required for successful tasks")
        return v


class WorkflowState(BaseModel):
    """Tracks the state of a workflow execution."""

    adw_id: str = Field(..., description="Unique ADW ID for this workflow")
    worktree_name: str = Field(..., description="Git worktree name")
    task_description: str = Field(..., description="Task being processed")
    phase: Literal["planning", "implementing", "updating", "completed", "failed"] = (
        Field(..., description="Current phase of the workflow")
    )
    started_at: datetime = Field(
        default_factory=datetime.now, description="Workflow start time"
    )
    completed_at: Optional[datetime] = Field(
        None, description="Workflow completion time"
    )
    plan_path: Optional[str] = Field(None, description="Path to generated plan file")
    error: Optional[str] = Field(None, description="Error message if workflow failed")

    def mark_completed(self, success: bool = True, error: Optional[str] = None):
        """Mark workflow as completed."""
        self.completed_at = datetime.now()
        self.phase = "completed" if success else "failed"
        if error:
            self.error = error


class CronTriggerConfig(BaseModel):
    """Configuration for the cron trigger."""

    polling_interval: int = Field(
        default=5, ge=1, description="Polling interval in seconds"
    )
    dry_run: bool = Field(
        default=False, description="Run in dry-run mode without making changes"
    )
    max_concurrent_tasks: int = Field(
        default=5, ge=1, description="Maximum number of concurrent tasks to process"
    )
    task_file_path: str = Field(
        default="tasks.md", description="Path to the task list file"
    )
    worktree_base_path: str = Field(
        default="trees", description="Base directory for git worktrees"
    )


class WorktreeConfig(BaseModel):
    """Configuration for creating a new worktree."""

    worktree_name: str = Field(..., description="Name of the worktree to create")
    base_branch: str = Field(
        default="main", description="Base branch to create worktree from"
    )
    copy_env: bool = Field(
        default=True, description="Whether to copy .env file to worktree"
    )


# Notion-specific models for the rapid prototyping multi-agent system

class NotionTask(BaseModel):
    """Represents a task from the Notion database."""

    page_id: str = Field(..., description="Notion page ID")
    title: str = Field(..., description="Task title/name")
    status: str = Field(
        "Not started", description="Current task status"
    )  # Flexible string to avoid literal error; validate later if needed
    content_blocks: List[Dict[str, Any]] = Field(
        default_factory=list, description="Page content blocks"
    )
    tags: Dict[str, str] = Field(
        default_factory=dict, description="Extracted tags from content {{key: value}}"
    )
    worktree: Optional[str] = Field(
        None, description="Target worktree name"
    )
    thinking_model: Optional[str] = Field(
        None, description="Model for complex reasoning/planning (defaults to x-ai/grok-4)"
    )
    fast_model: Optional[str] = Field(
        None, description="Model for regular prompting (defaults to x-ai/grok-4-fast)"
    )
    workflow_type: Optional[str] = Field(
        None, description="Workflow to use (build/plan-implement)"
    )
    prototype: Optional[str] = Field(
        None, description="Prototype type for app generation (uv_script/vite_vue/bun_scripts/uv_mcp)"
    )
    last_block_type: Optional[str] = Field(
        None, description="Type of the last content block"
    )
    execution_trigger: Optional[str] = Field(
        None, description="Execution command (execute/continue)"
    )
    task_prompt: Optional[str] = Field(
        None, description="Extracted task prompt for agent processing"
    )
    assigned_to: Optional[str] = Field(
        None, description="User assigned to this task"
    )
    created_time: Optional[datetime] = Field(
        None, description="Task creation timestamp"
    )
    last_edited_time: Optional[datetime] = Field(
        None, description="Last modification timestamp"
    )

    @validator("status")
    def validate_notion_status(cls, v):
        """Ensure status is one of the valid Notion values."""
        valid_statuses = ["Not started", "In progress", "Done", "HIL Review", "Failed"]
        if v not in valid_statuses:
            raise ValueError(f"Notion status must be one of {valid_statuses}")
        return v

    def is_eligible_for_processing(self) -> bool:
        """Check if task can be picked up for processing."""
        return (
            self.status in ["Not started", "HIL Review"] and
            self.execution_trigger in ["execute", "continue"]
        )

    def extract_app_context(self) -> Optional[str]:
        """Extract app context from tags."""
        return self.tags.get("app")

    def get_thinking_model(self) -> str:
        """Get the thinking model, allowing any specified model or defaulting to x-ai/grok-4."""
        model = self.thinking_model or self.tags.get("thinking") or self.tags.get("model") or os.getenv("MODEL_THINKING")
        return model if model else "x-ai/grok-4"

    def get_fast_model(self) -> str:
        """Get the fast model, allowing any specified model or defaulting to x-ai/grok-4-fast."""
        model = self.fast_model or self.tags.get("fast") or self.tags.get("model") or os.getenv("MODEL_FAST")
        return model if model else "x-ai/grok-4-fast"

    def should_use_full_workflow(self) -> bool:
        """Determine if task should use plan-implement-update workflow."""
        return (
            self.workflow_type == "plan" or
            self.tags.get("workflow") == "plan" or
            len(self.task_prompt or "") > 500  # Complex tasks get full workflow
        )


class NotionTaskUpdate(BaseModel):
    """Update payload for Notion task progress."""
    
    page_id: str = Field(..., description="Notion page ID to update")
    status: Optional[str] = Field(None, description="New status value")
    content_blocks: List[Dict[str, Any]] = Field(
        default_factory=list, description="Content blocks to append"
    )
    agent_output: Optional[str] = Field(
        None, description="Agent output to add as JSON block"
    )
    update_type: Literal["status", "content", "progress", "completion", "error"] = Field(
        ..., description="Type of update being made"
    )
    adw_id: Optional[str] = Field(
        None, description="ADW ID for tracking"
    )
    agent_name: Optional[str] = Field(
        None, description="Name of the agent making the update"
    )
    session_id: Optional[str] = Field(
        None, description="Session ID for debugging"
    )
    commit_hash: Optional[str] = Field(
        None, description="Git commit hash for completion updates"
    )
    error_message: Optional[str] = Field(
        None, description="Error message for failure updates"
    )

    @validator("status")
    def validate_update_status(cls, v):
        """Validate status is a valid Notion status."""
        if v is not None:
            valid_statuses = ["Not started", "In progress", "Done", "HIL Review", "Failed"]
            if v not in valid_statuses:
                raise ValueError(f"Status must be one of {valid_statuses}")
        return v


class WorktreeCreationRequest(BaseModel):
    """Request for automatic worktree creation."""
    
    task_description: str = Field(..., description="Task description for context")
    suggested_name: Optional[str] = Field(
        None, description="User-suggested worktree name"
    )
    base_branch: str = Field(
        default="main", description="Base branch for worktree"
    )
    app_context: Optional[str] = Field(
        None, description="App context for worktree creation"
    )
    prefix: Optional[str] = Field(
        None, description="Prefix for worktree name generation"
    )

    def generate_name_args(self) -> List[str]:
        """Generate arguments for /make_worktree_name command."""
        return [
            self.task_description,
            self.app_context or "",
            self.prefix or ""
        ]


class NotionCronConfig(BaseModel):
    """Configuration for Notion-based cron trigger."""

    database_id: str = Field(..., description="Notion database ID")
    polling_interval: int = Field(
        default=15, ge=5, description="Polling interval in seconds"
    )
    max_concurrent_tasks: int = Field(
        default=3, ge=1, le=10, description="Maximum concurrent notion tasks"
    )
    default_model: str = Field(
        default="x-ai/grok-4-fast", description="Default Claude model"
    )
    apps_directory: str = Field(
        default="apps", description="Target apps directory"
    )
    worktree_base_path: str = Field(
        default="trees", description="Base directory for worktrees"
    )
    dry_run: bool = Field(
        default=False, description="Run in dry-run mode without making changes"
    )
    status_filter: List[str] = Field(
        default_factory=lambda: ["Not started", "HIL Review"],
        description="Task statuses to poll for"
    )
    enable_hil_review: bool = Field(
        default=True, description="Enable HIL (Human-in-the-Loop) review support"
    )
    debug: bool = Field(
        default=False, description="Enable debug mode"
    )
    debug: bool = Field(
        default=False, description="Enable debug mode"
    )

    @validator("default_model")
    def validate_default_model(cls, v):
        """Ensure default model is valid."""
        if v not in ["x-ai/grok-4", "x-ai/grok-4-fast"]:
            raise ValueError("Default model must be 'x-ai/grok-4' or 'x-ai/grok-4-fast'")
        return v

    @validator("status_filter")
    def validate_status_filter(cls, v):
        """Validate status filter contains valid statuses."""
        valid_statuses = ["Not started", "In progress", "Done", "HIL Review", "Failed"]
        for status in v:
            if status not in valid_statuses:
                raise ValueError(f"Invalid status in filter: {status}")
        return v


class NotionWorkflowState(BaseModel):
    """Tracks the state of a Notion-based workflow execution."""

    adw_id: str = Field(..., description="Unique ADW ID for this workflow")
    page_id: str = Field(..., description="Notion page ID")
    worktree_name: str = Field(..., description="Git worktree name")
    task_description: str = Field(..., description="Task being processed")
    workflow_type: Literal["build_update", "plan_implement_update"] = Field(
        ..., description="Type of workflow being executed"
    )
    phase: Literal[
        "starting", "planning", "implementing", "updating", "completed", "failed"
    ] = Field(..., description="Current phase of the workflow")
    model: str = Field(..., description="Claude model being used")
    started_at: datetime = Field(
        default_factory=datetime.now, description="Workflow start time"
    )
    completed_at: Optional[datetime] = Field(
        None, description="Workflow completion time"
    )
    plan_path: Optional[str] = Field(None, description="Path to generated plan file")
    commit_hash: Optional[str] = Field(None, description="Final commit hash")
    error: Optional[str] = Field(None, description="Error message if workflow failed")
    notion_updates_count: int = Field(
        default=0, description="Number of updates sent to Notion"
    )
    
    def mark_completed(self, success: bool = True, error: Optional[str] = None, commit_hash: Optional[str] = None):
        """Mark workflow as completed."""
        self.completed_at = datetime.now()
        self.phase = "completed" if success else "failed"
        if error:
            self.error = error
        if commit_hash:
            self.commit_hash = commit_hash

    def get_duration(self) -> Optional[float]:
        """Get workflow duration in seconds."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class NotionAgentMetrics(BaseModel):
    """Metrics for monitoring Notion agent performance."""
    
    tasks_processed: int = Field(default=0, description="Total tasks processed")
    tasks_completed: int = Field(default=0, description="Tasks completed successfully")
    tasks_failed: int = Field(default=0, description="Tasks that failed")
    average_processing_time: float = Field(default=0.0, description="Average time per task")
    notion_api_calls: int = Field(default=0, description="Total Notion API calls made")
    notion_api_errors: int = Field(default=0, description="Notion API call failures")
    worktrees_created: int = Field(default=0, description="Worktrees created")
    last_reset: datetime = Field(
        default_factory=datetime.now, description="Last metrics reset time"
    )
    
    def success_rate(self) -> float:
        """Calculate task success rate."""
        total = self.tasks_processed
        return (self.tasks_completed / total * 100) if total > 0 else 0.0
    
    def api_success_rate(self) -> float:
        """Calculate Notion API success rate."""
        total = self.notion_api_calls
        return ((total - self.notion_api_errors) / total * 100) if total > 0 else 100.0
