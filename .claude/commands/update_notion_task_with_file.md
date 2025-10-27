# Update Notion Task with File

Update a Notion task page by appending content from an agent output file as appropriately formatted blocks.

## Variables

page_id: $1
agent_output_file: $2
block_type: $3
status: $4

## Instructions

You are an agent that reads agent output files and posts their content to Notion task pages for live progress tracking. Follow these steps:

1. **Read the File**: Read the content from `agent_output_file` path. Handle these common file types:

   - **JSON files** (.json): Parse and pretty-format the JSON
   - **Log files** (.log, .txt): Read as plain text
   - **JSONL files** (.jsonl): Read and format each JSON line

2. **Create Appropriate Notion Blocks**: Based on `block_type`:

   - **"code"**: Use code blocks with syntax highlighting (JSON gets "json" language)
   - **"toggle"**: Use collapsible toggle blocks for large content (>1000 chars)
   - **"callout"**: Use callout blocks with appropriate styling
   - **"paragraph"**: Use regular paragraph blocks

3. **Add Context Header**: Before the file content, add a paragraph block with metadata:

   ```text
   📊 Agent Output - {phase} Phase
   ADW ID: {adw_id} | Agent: {agent_name}
   File: {relative_file_path}
   Generated: {current_timestamp}
   ```

4. **Append to Notion Page**: Use the Notion API to append the formatted content blocks to the specified page.

5. **Update Status** (if provided): If `status` parameter is given, also update the page's Status property.

6. **Handle Large Content**: If content is over 2000 characters, use toggle blocks or split into multiple code blocks to stay within Notion's limits.

Return confirmation with the number of blocks added and the page URL.

## File Types Supported

- **JSON Files**: Agent output in JSON format (\*.json)
- **Log Files**: Plain text logs (_.log,_.txt)
- **Markdown Files**: Formatted markdown content (\*.md)
- **JSONL Files**: JSON Lines format for streaming data (\*.jsonl)

## Block Type Options

- `code` - Format content as code block (good for JSON, logs)
- `paragraph` - Format as regular text paragraphs
- `callout` - Format as highlighted callout block
- `quote` - Format as blockquote
- `toggle` - Format as collapsible toggle block (good for long content)

## Agent Output File Paths

Expected file locations based on ADW structure:

- `./agents/{adw_id}/{agent_name}/cc_raw_output.json` - Raw Claude output
- `./agents/{adw_id}/{agent_name}/cc_final_object.json` - Final result object
- `./agents/{adw_id}/{agent_name}/custom_summary_output.json` - Phase summary
- `./agents/{adw_id}/workflow_summary.json` - Overall workflow summary

## JSON Content Formatting

For JSON files, create structured code blocks with:

- Language set to "json" for syntax highlighting
- Pretty-printed formatting (indented)
- Metadata header with file info

### Example JSON Block

```json
{
  "object": "block",
  "type": "code",
  "code": {
    "rich_text": [
      {
        "type": "text",
        "text": {
          "content": "{\n  \"adw_id\": \"abc123\",\n  \"phase\": \"build\",\n  \"success\": true,\n  \"timestamp\": \"2024-01-15T14:30:00Z\"\n}"
        }
      }
    ],
    "language": "json",
    "caption": [
      {
        "type": "text",
        "text": {
          "content": "Agent Output - Build Phase (abc123)"
        }
      }
    ]
  }
}
```

## Toggle Block for Large Content

For large outputs (>1000 chars), use toggle blocks:

```json
{
  "object": "block",
  "type": "toggle",
  "toggle": {
    "rich_text": [
      {
        "type": "text",
        "text": {
          "content": "Agent Output Details (Click to expand)"
        }
      }
    ],
    "children": [
      {
        "object": "block",
        "type": "code",
        "code": {
          "rich_text": [
            {
              "type": "text",
              "text": {
                "content": "... large content here ..."
              }
            }
          ],
          "language": "json"
        }
      }
    ]
  }
}
```

## Phase-Specific Formatting

### Build Phase Output

- Use code blocks for build logs
- Add callout for success/failure status
- Include commit hash if available

### Planning Phase Output

- Use toggle blocks for plan details
- Add paragraph summary
- Include plan file path

### Update Phase Output

- Use callout blocks for status changes
- Include final task status
- Add workflow completion summary

## Content Preprocessing

Before adding to Notion:

1. **Truncate**: Limit single blocks to 2000 characters max
2. **Sanitize**: Remove or escape special characters that break Notion formatting
3. **Structure**: Break large content into multiple blocks
4. **Metadata**: Add context headers with agent info

## Example Usage

Add build phase output:

```bash
/update_notion_task_with_file 247fc382-ac73-8063-9b44-e33188d4d791 ./agents/abc123/builder-feature-auth/custom_summary_output.json code "In progress"
```

Add final workflow summary:

```bash
/update_notion_task_with_file 247fc382-ac73-8063-9b44-e33188d4d791 ./agents/abc123/workflow_summary.json toggle "Done"
```

## Error Handling

### File Not Found

- Return clear error message with file path
- Suggest checking agent output directory
- List common file locations

### File Format Issues

- Attempt to parse as plain text if JSON parsing fails
- Warn about malformed content
- Include partial content if possible

### Notion API Errors

- Retry block creation up to 3 times
- Break large content into smaller blocks if size limit exceeded
- Log errors with file path and content preview

## Metadata Headers

Add context headers before file content:

### For JSON Files

```text
=� Agent Output - {phase} Phase
ADW ID: {adw_id} | Agent: {agent_name}
File: {relative_file_path}
Generated: {file_timestamp}
```

### For Log Files

```text
=� Process Logs - {phase} Phase
ADW ID: {adw_id} | Agent: {agent_name}
File: {relative_file_path}
Generated: {file_timestamp}
```

## Content Size Limits

- **Single Block**: 2000 characters max
- **Page Total**: 100 blocks max per update
- **File Size**: 10MB max file size for processing

If content exceeds limits:

1. Truncate with "... (content truncated)" indicator
2. Split into multiple blocks/toggle sections
3. Suggest viewing full output in agent directory

## Security Considerations

- Validate file paths to prevent directory traversal
- Sanitize file content to prevent injection attacks
- Limit file size to prevent memory issues
- Log all file access attempts for audit

## Success Response

Return confirmation with:

- Page ID updated
- File path processed
- Number of blocks added
- New status (if updated)
- Processing timestamp
