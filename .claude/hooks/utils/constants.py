from pathlib import Path

def ensure_session_log_dir(session_id):
    """Ensure the session log directory exists and return its path."""
    log_dir = Path.home() / '.claude' / 'logs' / session_id
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir