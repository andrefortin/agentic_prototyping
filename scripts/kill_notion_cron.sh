#!/bin/bash

# Script to kill any running adw_trigger_cron_notion_tasks.py processes

echo "🔍 Searching for adw_trigger_cron_notion_tasks.py processes..."

# Find processes matching the pattern
pids=$(ps aux | grep -E "python.*adws/adw_triggers/adw_trigger_cron_notion_tasks\.py" | grep -v grep | awk '{print $2}')

if [ -z "$pids" ]; then
    echo "✓ No adw_trigger_cron_notion_tasks.py processes found."
    exit 0
fi

# Show processes that will be killed
echo -e "\nFound processes to kill:"
ps aux | grep -E "python.*adws/adw_triggers/adw_trigger_cron_notion_tasks\.py" | grep -v grep

# Kill the processes
echo -e "\nKilling processes..."
echo "$pids" | while read -r pid; do
    if [ -n "$pid" ]; then
        kill -9 "$pid" 2>/dev/null && echo "  ✓ Killed PID: $pid" || echo "  ✗ Failed to kill PID: $pid"
    fi
done

echo -e "\n✅ Done!"