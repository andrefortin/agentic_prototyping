# Make Worktree Name

Generate a unique worktree name from task description.

## Variables

task_description: $1
prefix: $2

## Instructions

IMPORTANT: You MUST output ONLY the worktree name (e.g., "feat-hello-api"). Nothing else. No explanations, no checking messages, no JSON - just the final name on a single line.

1. Extract key terms from task_description
2. Generate name (5-20 chars, lowercase, hyphens only)
3. Silently check uniqueness against existing worktrees (DO NOT output the checking process)
4. Output ONLY the final worktree name on a single line

## Naming Rules

### Prefixes (use provided or detect)

- `feat` - New features
- `fix` - Bug fixes
- `refact` - Refactoring
- `test` - Testing
- `docs` - Documentation
- `task` - Default

### Action Abbreviations

- implement → impl
- fix/resolve → fix
- add/create → add
- update/modify → upd
- remove/delete → rm
- refactor → refact

### Examples

- "Implement user authentication" → `feat-user-auth`
- "Fix login validation bug" → `fix-login-val`
- "Add payment gateway" → `add-payment-gw`
- "Refactor API service" → `refact-api-svc`

### Uniqueness

If name exists, append number: `feat-auth-2`

## Output

IMPORTANT: Return ONLY the generated name as plain text: feat-user-auth
