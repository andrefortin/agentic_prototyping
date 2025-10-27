#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "python-dotenv",
# ]
# ///

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from utils.constants import ensure_session_log_dir

from contextlib import suppress
with suppress(ImportError):
    from dotenv import load_dotenv  # type: ignore
    # Load environment variables
    load_dotenv()


def log_session_start(session_id, input_data):
    """Log session start to session directory."""
    # Ensure session log directory exists
    log_dir = ensure_session_log_dir(session_id)
    log_file = log_dir / 'session_start.json'
    
    # Read existing log data or initialize empty list
    if log_file.exists():
        with open(log_file, 'r') as f:
            try:
                log_data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                log_data = []
    else:
        log_data = []
    
    # Append the entire input data
    log_data.append(input_data)
    
    # Write back to file with formatting
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)


def get_additional_context(source):
    """
    Generate additional context based on the session source.
    This can be customized to load project-specific context.
    """
    # Add any project-specific context here
    # For example, you could load CLAUDE.md content, recent issues, etc.
    
    return None


def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--add-context', action='store_true',
                          help='Add additional context to the session')
        args = parser.parse_args()
        
        # Read JSON input from stdin
        input_data = json.loads(sys.stdin.read())
        
        # Extract session_id and source
        session_id = input_data.get('session_id', 'unknown')
        source = input_data.get('source', 'unknown')
        
        # Log the session start
        log_session_start(session_id, input_data)
        
        # Optionally add context
        if args.add_context:
            if additional_context := get_additional_context(source):
                # Use JSON output to provide context
                output = {
                    "hookSpecificOutput": {
                        "hookEventName": "SessionStart",
                        "additionalContext": additional_context
                    }
                }
                print(json.dumps(output))
        
        # Success
        sys.exit(0)
        
    except json.JSONDecodeError:
        # Handle JSON decode errors gracefully
        sys.exit(0)
    except Exception:
        # Handle any other errors gracefully
        sys.exit(0)


if __name__ == '__main__':
    main()