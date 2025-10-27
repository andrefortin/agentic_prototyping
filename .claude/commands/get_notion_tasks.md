# Get Notion Tasks

Query the Notion database for eligible tasks and return them in a structured format for processing.

## Variables

database_id: $1
status_filter: $2
limit: $3

## Default Variable Values

- database_id: (REQUIRED) Must be provided. This should be the Notion database ID from the `NOTION_AGENTIC_TASK_TABLE_ID` environment variable.
- status_filter: (OPTIONAL) `["Not started", "HIL Review"]` (eligible statuses)
- limit: (OPTIONAL) `10` (maximum tasks to return)

## Instructions

You are an agent that queries Notion databases to find tasks ready for processing. IMPORTANT: Use the NOTION_API_KEY environment variable directly for authentication. Do not validate or check the token prefix (accept 'secret_' or 'ntn_' formats as valid). If the token is present, use it for API calls without extra validation. Use the MCP server \"notion\" for all API calls (no direct HTTP requests; use MCP tools like notion_query_database for reads, notion_retrieve_page for page details, and notion_retrieve_block_children for blocks). The MCP handles the BASE_URL as \"https://api.notion.com/v1\" and uses Notion-Version \"2025-09-03\" (latest, compatible for both reads and writes). For queries, use standard Notion endpoints: POST /v1/databases/{database_id}/query. Follow these steps:

1. **Query the Database**: First, use MCP tool notion_retrieve_database on the provided database_id to get the schema and extract the first data_source_id from the 'data_sources' array. Then, use MCP tool notion_query_database with the extracted data_source_id for querying tasks, filter by status in status_filter (default: ["Not started", "HIL Review"]), and limit to `limit` (default: 10). Sort by creation date ascending for oldest first. If no data_sources, fall back to querying the database directly with notion_query_database using the database_id as if it were the data_source. Normalize IDs if hyphenated.

2. **For Each Task Retrieved**:

   **MANDATORY**: Always include ALL tasks matching the status filter (e.g., "Not started", "HIL Review"). Do NOT skip based on content triggers - eligibility is handled downstream.

   - Use MCP tool notion_retrieve_page on the page_id to get properties (title, status).
   - Use MCP tool notion_retrieve_block_children on the page_id to get ALL content blocks (recursive if needed; include full structure with rich_text arrays).
   - Combine ALL block texts into task_prompt (full concatenated string).
   - Parse `{{key: value}}` tags from the full task_prompt.
   - Set execution_trigger to "execute" if any block contains "execute" or prototype tags are present; otherwise null.
   - **CRITICAL**: Always populate 'content_blocks' with the COMPLETE array of block objects from notion_retrieve_block_children. Do not return empty [] - this is essential for downstream parsing.

3. **Extract Information**:

   - **Page ID**: The Notion page identifier
   - **Title**: From the page properties (title property)
   - **Status**: Current status value from properties
   - **Tags**: Parse content for `{{worktree: name}}`, `{{model: x-ai/grok-4}}`, `{{workflow: plan}}`, `{{app: appname}}`, `{{prototype: type}}` patterns
   - **Execution Trigger**: "execute" or "continue" based on last block
   - **Task Prompt**: Full combined content string for "execute", or just the continue prompt string for "continue -"

4. **Return JSON Response**: Format as an array of eligible tasks with all extracted information. For content_blocks, return the full block structure from notion_retrieve_block_children.

## Task Inclusion Rules

- **INCLUDE ALWAYS**: All tasks where status matches the status_filter (e.g., "Not started", "HIL Review").
- **EXCLUDE ONLY**: Tasks with status "In progress" or explicitly filtered out by status_filter.
- **Do NOT filter** based on content triggers ("execute", "continue -", or tags) - return everything for downstream processing.
- **Full Data**: For each task, ensure:
  - content_blocks: Full array of ALL blocks (use notion_retrieve_block_children with page_size=100, start if paginated).
  - task_prompt: Concatenated plain_text from ALL rich_text in ALL blocks.
  - tags: Parsed from task_prompt.
  - execution_trigger: "execute" if "execute" appears anywhere in task_prompt or if "prototype" tag exists; "continue" if "continue -" appears; otherwise null.

## Response Format

Always return valid JSON array, even if empty []. If no tasks found or query fails, return [{"debug": "No tasks matched filter; queried database_id=[ID], status_filter=[FILTER], limit=[LIMIT]. Check MCP 'notion' server and NOTION_API_KEY."}].

For each task, populate content_blocks fully - empty arrays indicate missing data and will cause parsing failures downstream.

**DEBUG MODE**: If --debug flag is provided, include "debug" field with query details (e.g., number of results, any API errors).

## Tag Extraction

Extract tags from the combined task_prompt string using the pattern `{{key: value}}`:

- `{{worktree: feature-auth}}` - Target worktree name
- `{{model: x-ai/grok-4}}` - X-AI model preference (grok-4/grok-4-fast)
- `{{workflow: plan|build}}` - Force plan-implement workflow
- `{{app: sentiment_analysis}}` - Target app directory
- `{{prototype: uv_script|vite_vue|bun_scripts|uv_mcp}}` - Prototype type for app generation

## Execution Trigger Detection

Check the last content block's rich_text for execution commands:

- `execute` - Process all page content as task prompt, combine texts into a single string for the task_prompt field
- `continue - <specific prompt>` - Process only the continue block text, combine into a single string for the task_prompt field

**CRITICAL**: Output MUST be valid JSON only. Do not include any text before or after the JSON array. No tool calls in the final output - perform all MCP calls and then output the JSON.

## Error Handling

- If database_id is invalid, return [{"error": "Invalid database ID"}]
- If no tasks are eligible, return []
- If MCP/Notion API fails, retry up to 3 times with exponential backoff
- Log all errors for debugging purposes

## Concurrent Access Control

- Filter out tasks with status "In progress"
- Skip tasks modified in the last 30 seconds (use last_edited_time from properties)
- Return tasks in order of creation (oldest first)

## Response Format

**CRITICAL**: Even on errors, return ONLY valid JSON array (e.g., [{"error": "..."}] or []). No markdown, no explanations, no text outside JSON. Prefix with nothing - just the array.

IMPORTANT: task_prompt should be all of the content blocks' texts combined into a single string. This is critical for the agent to process the task. content_blocks should be the full list of block objects.

Return JSON array with the following structure:

```json
[
  {
    "page_id": "notion-page-id",
    "title": "Task title from properties",
    "status": "Not started",
    "content_blocks": [
      {
        "object": "block",
        "type": "paragraph",
        "paragraph": {
          "rich_text": [
            {
              "type": "text",
              "text": {
                "content": "Task description here"
              }
            }
          ]
        }
      }
    ],
    "tags": {
      "worktree": "feature-auth",
      "model": "x-ai/grok-4",
      "workflow": "plan",
      "prototype": "vite_vue"
    },
    "execution_trigger": "execute",
    "task_prompt": "Extracted task description for agent processing"
  }
]
```
