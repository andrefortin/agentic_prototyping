# Update Notion Task

Update a Notion task page with new status and optional content blocks.

## Variables

page_id: $1
status: $2
update_content: $3

## Instructions

You are an agent that updates Notion task pages with status changes and progress updates. Use direct API calls via Bash tool with curl, as MCP may not support updates reliably. IMPORTANT: Use the NOTION_API_KEY environment variable directly for authentication. Do not validate or check the token prefix (accept 'secret*' or 'ntn_' formats as valid). If the token is present, use it for API calls without extra validation. BASE_URL: https://api.notion.com/v1. Notion-Version: 2025-09-03. For updates, use standard Notion endpoints: PATCH /v1/pages/{page_id} for properties, POST /v1/blocks/{page_id}/children for appending blocks. 'data_sources' is not a standard endpoint and should not be used. For databases, use /v1/databases for querying/updating database schema, but task updates are on pages. Follow these steps:

1. **Update Page Status**: Use Bash tool to run curl -X PATCH "https://api.notion.com/v1/pages/${page_id}" -H "Authorization: Bearer $NOTION_API_KEY" -H "Notion-Version: 2025-09-03" -H "Content-Type: application/json" -d '{"properties": {"Status": {"select": {"name": status}}}}'. Valid statuses: "Not started", "In progress", "Done", "HIL Review", "Failed". Verify the response has "Status": {"select": {"name": status}} and no error.

2. **Add Content Block** (if update_content provided): Parse update_content as JSON if possible, else treat as text. Use Bash tool to append blocks via curl -X POST "https://api.notion.com/v1/blocks/${page_id}/children" -H "Authorization: Bearer $NOTION_API_KEY" -H "Notion-Version: 2025-09-03" -H "Content-Type: application/json" -d '[block_json_array]'. Append blocks with update information (e.g., callout for status). Block types:

   - **Status changes**: Callout blocks with icons (🚀 In progress, ✅ Done, ❌ Failed, 👤 HIL Review)
   - **Progress updates**: Paragraph blocks
   - **Error messages**: Callout blocks (red). Notion-Version: "2025-09-03" for appends. Limit to 1-3 blocks for brevity.

Note: For getting tasks (reads), you may use direct curl if needed, but focus on updates. Notion-Version "2025-09-03" is used by default.

3. **Include Metadata**: In the content block, include:

   - Current timestamp in ISO format
   - ADW ID if available (parse from update_content)
   - Agent name/session info if available
   - Any relevant context about the update

4. **Handle Status-Specific Updates**:
   - **"In progress"**: Add callout with 🚀 icon indicating task started
   - **"Done"**: Add callout with ✅ icon and include commit hash if provided (parse from update_content)
   - **"Failed"**: Add callout with ❌ icon and error details (from update_content)
   - **"HIL Review"**: Add callout with 👤 icon requesting human review

Return confirmation of the update with the page URL (https://www.notion.so/${page_id without hyphens}?pvs=21). If API fails, retry up to 3 times with 2s backoff. Log curl outputs via Bash for debugging but do not output full token.

## Valid Status Values

- `Not started` - Task is ready for processing
- `In progress` - Task is currently being worked on
- `Done` - Task has been completed successfully
- `HIL Review` - Task needs human review before proceeding
- `Failed` - Task processing failed

## Content Block Types

When adding content, use appropriate Notion block types:

- **Text Updates**: Use paragraph blocks for general updates
- **Agent Output**: Use code blocks for structured agent output
- **Status Changes**: Use callout blocks for important status updates
- **JSON Data**: Use code blocks with language "json" for structured data

## Update Content Format

The update_content parameter can include:

- Simple text for paragraph blocks
- Structured content with block type specifications
- JSON objects for formatted data blocks

## Timestamp Format

All updates include human readable date and time.

## Agent Metadata

Include agent information with updates:

- ADW ID for tracking
- Agent name/type
- Workflow phase (if applicable)
- Session ID for debugging
- Worktree name

## Content Block Examples

### Status Update Block

```json
{
  "object": "block",
  "type": "callout",
  "callout": {
    "rich_text": [
      {
        "type": "text",
        "text": {
          "content": "Status updated to: In progress"
        }
      }
    ],
    "icon": {
      "emoji": "🚀"
    }
  }
}
```

### Agent Output Block

```json
{
  "object": "block",
  "type": "code",
  "code": {
    "rich_text": [
      {
        "type": "text",
        "text": {
          "content": "{\"adw_id\": \"abc123\", \"phase\": \"build\", \"success\": true}"
        }
      }
    ],
    "language": "json"
  }
}
```

### Progress Update Block

```json
{
  "object": "block",
  "type": "paragraph",
  "paragraph": {
    "rich_text": [
      {
        "type": "text",
        "text": {
          "content": "Agent abc123 started build phase at 2024-01-15T14:30:00Z"
        }
      }
    ]
  }
}
```

## Error Handling

- If API calls fail, retry up to 3 times with exponential backoff
- Log all errors for debugging purposes

- **Invalid Page ID**: Return clear error message with validation details
- **Invalid Status**: List valid status options in error message
- **API Failures**: Retry up to 3 times with exponential backoff
- **Permission Issues**: Check database access and provide guidance

## Success Response

Return confirmation with:

- Updated page ID
- New status value
- Number of content blocks added
- Timestamp of update

### In Progress (🚀 In progress)

- Add start callout block
- Include timestamp
- Summarize work to be done

### Completing Task (✅ Done)

- Add success callout block
- Include commit hash if available
- Add completion timestamp
- Summarize work done

### Failing Task (❌ Failed)

- Add error callout block with failure icon
- Include error details and troubleshooting info
- Add timestamp and agent information
- Preserve error logs for debugging

### HIL Review (👤 HIL Review)

- Add review request block
- Include context about why review is needed
- Add instructions for reviewer
