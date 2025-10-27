#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "anthropic",
#     "python-dotenv",
# ]
# ///

"""
AI-powered event summarizer for Claude Code hooks.
Generates concise summaries of hook events using Anthropic's Claude.
"""

import os
import json
from contextlib import suppress
from typing import Optional, Dict, Any

from contextlib import suppress
with suppress(ImportError):
    from dotenv import load_dotenv  # type: ignore
    # Load environment variables
    load_dotenv()


def _create_prompt(hook_type: str, payload: Dict[str, Any]) -> str:
    """Create a concise prompt based on hook type and payload."""
    if hook_type == "PreToolUse":
        tool_name = payload.get("tool_name", "Unknown")
        return f"Summarize in 1 sentence: PreToolUse hook for {tool_name} tool"
    elif hook_type == "PostToolUse":
        tool_name = payload.get("tool_name", "Unknown")
        return f"Summarize in 1 sentence: PostToolUse hook for {tool_name} tool"
    elif hook_type == "UserPromptSubmit":
        user_prompt = payload.get("prompt", "")[:100]  # First 100 chars
        return f"Summarize in 1 sentence: User submitted prompt: {user_prompt}"
    else:
        return f"Summarize in 1 sentence: {hook_type} hook event occurred"


def _call_anthropic_api(api_key: str, prompt: str) -> str:
    """Call Anthropic API to generate summary."""
    import anthropic  # type: ignore

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=50,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def generate_event_summary(event_data: Dict[str, Any]) -> Optional[str]:
    """
    Generate a concise summary of a hook event using Claude.

    Args:
        event_data: The event data dictionary containing hook information

    Returns:
        A string summary or None if generation fails
    """

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        import anthropic  # type: ignore

        # Extract key information from event
        hook_type = event_data.get("hook_event_type", "Unknown")
        payload = event_data.get("payload", {})

        prompt = _create_prompt(hook_type, payload)
        return _call_anthropic_api(api_key, prompt)

    except Exception:
        return None


def main():
    """Test the summarizer with sample data."""
    sample_event = {
        "hook_event_type": "PreToolUse",
        "payload": {"tool_name": "Bash", "tool_input": {"command": "ls -la"}},
    }

    if summary := generate_event_summary(sample_event):
        print(f"Summary: {summary}")
    else:
        print("Failed to generate summary")


if __name__ == "__main__":
    main()
