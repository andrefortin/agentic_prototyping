# Multi-Agent Notion-Based Rapid Prototyping System

A sophisticated multi-agent system designed for **rapid application prototyping**. Simply describe your app idea in Notion, add a prototype tag, and watch as AI agents automatically generate fully-functional applications complete with proper project structure, dependencies, and best practices. The system monitors Notion databases continuously and delegates tasks to specialized AI agents using isolated git worktrees for parallel development.

## 🏗️ System Architecture

### Overview

This system operates as a continuous rapid prototyping service that:

1. **Monitors Notion** for prototype requests marked as "Not started" or "HIL Review" with `execute` triggers
2. **Detects prototype tags** like `{{prototype: vite_vue}}` and routes to specialized planning agents
3. **Claims tasks instantly** by updating status to "In progress" to prevent duplicate work
4. **Creates isolated worktrees** for each prototype to enable parallel development
5. **Generates comprehensive plans** using framework-specific `/plan_[prototype]` commands
6. **Implements complete applications** following the generated plans and best practices
7. **Updates Notion** with results, commit hashes, and any errors upon completion

**Rapid Prototyping Workflow:**

```mermaid
graph LR
    A[Notion Task] --> B{Prototype Tag?}
    B -->|Yes| C[plan_prototype]
    B -->|No| D[plan or build]
    C --> E[implement]
    D --> E
    E --> F[update_notion_task]
```

### Core Components

```text
agentic_prototyping/
├── adws/                         # AI Developer Workflows
│   ├── adw_modules/              # Shared modules
│   │   ├── agent.py              # Agent execution framework
│   │   ├── data_models.py        # Pydantic models for Notion/workflow data
│   │   └── utils.py              # Utility functions
│   ├── adw_triggers/             # Cron-based triggers
│   │   └── adw_trigger_cron_notion_tasks.py  # Main Notion monitor (polls every 15s)
│   ├── adw_build_update_notion_task.py       # Simple build workflow
│   └── adw_plan_implement_update_notion_task.py  # Complex planning workflow + prototypes
├── .claude/
│   ├── commands/                 # Slash command definitions
│   │   ├── plan.md               # /plan - general task planning
│   │   ├── plan_uv_script.md     # /plan_uv_script - Python UV scripts
│   │   ├── plan_bun_scripts.md   # /plan_bun_scripts - TypeScript Bun apps
│   │   ├── plan_vite_vue.md      # /plan_vite_vue - Vue.js web applications
│   │   ├── plan_uv_mcp.md        # /plan_uv_mcp - MCP servers
│   │   ├── implement.md          # /implement - execute generated plans
│   │   ├── build.md              # /build - direct implementation
│   │   ├── get_notion_tasks.md   # /get_notion_tasks - query Notion
│   │   ├── update_notion_task.md # /update_notion_task - update status
│   │   ├── init_worktree.md      # /init_worktree - create worktrees
│   │   └── make_worktree_name.md # /make_worktree_name - generate names
│   └── hooks/                    # Event hooks for customization
├── apps/                         # Generated applications (prototype output)
├── specs/                        # Generated implementation plans
└── trees/                        # Git worktrees (isolated dev environments)
```

## ⚙️ How It Works

### 1. Rapid Prototyping Detection & Task Claiming

The monitoring service (`adw_trigger_cron_notion_tasks.py`) continuously scans Notion:

```python
# Polls every 15 seconds by default
schedule.every(15).seconds.do(self.process_tasks)

# Detects prototype tags and routes to specialized workflows
use_full_workflow = task.should_use_full_workflow() or task.prototype is not None

# Immediately claims tasks to prevent duplicate processing
if task.is_eligible_for_processing():
    adw_id = generate_short_id()
    self.task_manager.update_task_status(
        task.page_id, "In progress", json.dumps({
            "adw_id": adw_id,
            "timestamp": datetime.now().isoformat()
        })
    )
```

**Task Eligibility Criteria:**

- Status: "Not started" or "HIL Review"
- Execution trigger: `execute` or `continue - [additional prompt]`
- Content: Task description with optional prototype tags

**Prototype Detection:**
The system automatically detects `{{prototype: type}}` tags and routes to specialized workflows:

- `{{prototype: uv_script}}` → `/plan_uv_script` command
- `{{prototype: vite_vue}}` → `/plan_vite_vue` command
- `{{prototype: bun_scripts}}` → `/plan_bun_scripts` command
- `{{prototype: uv_mcp}}` → `/plan_uv_mcp` command

### 2. Workflow Routing

The system automatically routes tasks to appropriate workflows:

#### Prototype Workflow (`adw_plan_implement_update_notion_task.py`)

**Triggered by**: `{{prototype: type}}` tags + `--execute`
**Commands**: `/plan_[prototype]` → `/implement` → `/update_notion_task`
**Purpose**: Generate complete applications from scratch (requires explicit --execute for prototypes)
**Examples**:

- `{{prototype: vite_vue}}`: Creates full Vue.js web application
- `{{prototype: uv_script}}`: Generates Python CLI tool with dependencies
- `{{prototype: bun_scripts}}`: Builds TypeScript application with Bun runtime

#### Complex Planning Workflow (`adw_plan_implement_update_notion_task.py`)

**Triggered by**: `{{workflow: plan}}` tag or complex tasks
**Commands**: `/plan` → `/implement` → `/update_notion_task`
**Purpose**: Multi-phase implementation with architectural planning
**Examples**: "Design and implement a user authentication system"

#### Simple Build Workflow (`adw_build_update_notion_task.py`)

**Triggered by**: Simple tasks without special tags
**Commands**: `/build` → `/update_notion_task`
**Purpose**: Direct implementation for straightforward changes
**Examples**: "Add a timestamp utility function", "Fix login bug"

### 3. Worktree Management

Each task gets an isolated git worktree with sparse checkout:

```python
# Generate descriptive worktree name
worktree_name = "feat-timestamp-util"

# Create worktree with sparse checkout
/init_worktree feat-timestamp-util agentic_prototyping
```

Benefits:

- **Parallel development**: Multiple agents work simultaneously
- **No conflicts**: Each task has its own branch
- **Clean history**: One branch per feature
- **Easy review**: Clear separation of concerns

### 4. Agent Execution

Agents are spawned as detached subprocesses:

```python
subprocess.Popen(cmd, start_new_session=True)
```

This allows:

- Parent process can continue monitoring
- Agents survive if monitor restarts
- True parallel execution
- No blocking on long-running tasks

## 🔧 Slash Commands

### Prototype Planning Commands

The system includes specialized planning commands for each prototype type that understand framework-specific patterns and best practices:

#### `/plan_uv_script`

Creates comprehensive plans for Python UV scripts with inline dependencies.

**Arguments:**

1. `adw_id` - Workflow tracking ID
2. `prompt` - Application description and requirements

**Output:** Detailed plan in `specs/plan-{app_name}-uv-script.md`

**Specializations:**

- UV dependency management with inline `/// script` blocks
- CLI argument parsing patterns
- Python packaging and distribution
- Error handling and logging standards

#### `/plan_vite_vue`

Generates plans for modern Vue.js applications with TypeScript and Vite.

**Arguments:**

1. `adw_id` - Workflow tracking ID
2. `prompt` - Application description and requirements

**Output:** Detailed plan in `specs/plan-{app_name}-vite-vue.md`

**Specializations:**

- Vue 3 Composition API patterns
- TypeScript integration and type safety
- Vite build optimization
- Component architecture and state management
- Responsive design principles

#### `/plan_bun_scripts`

Creates plans for TypeScript applications powered by Bun runtime.

**Arguments:**

1. `adw_id` - Workflow tracking ID
2. `prompt` - Application description and requirements

**Output:** Detailed plan in `specs/plan-{app_name}-bun-scripts.md`

**Specializations:**

- Native TypeScript execution
- Bun's built-in APIs (SQLite, file system, shell)
- Performance optimization patterns
- Testing with Bun's test runner

#### `/plan_uv_mcp`

Designs Model Context Protocol servers for AI tool integration.

**Arguments:**

1. `adw_id` - Workflow tracking ID
2. `prompt` - MCP server description and tool requirements

**Output:** Detailed plan in `specs/plan-{app_name}-uv-mcp.md`

**Specializations:**

- MCP protocol compliance
- Tool schema definition and validation
- Secure parameter handling
- Python async/await patterns for MCP

### Task Management Commands

#### `/get_notion_tasks`

Queries Notion database for tasks matching criteria.

**Arguments:**

1. `database_id` - Notion database ID (from env: `NOTION_AGENTIC_TASK_TABLE_ID`)
2. `status_filter` - JSON array of statuses (e.g., `["Not started", "HIL Review"]`)
3. `limit` - Maximum tasks to return

**Example Output:**

```json
[
  {
    "page_id": "247fc382-ac73-...",
    "title": "Add Timestamp Utility",
    "status": "Not started",
    "execution_trigger": "execute",
    "task_prompt": "Create a utility function...",
    "tags": { "model": "x-ai/grok-4-fast", "workflow": "build" }
  }
]
```

#### `/update_notion_task`

Updates task status and adds implementation details.

**Arguments:**

1. `page_id` - Notion page ID
2. `status` - New status ("In progress", "Done", "Failed")
3. `update_content` - JSON with details (commit hash, errors, etc.)

### Development Commands

#### `/build`

Implements features based on task description.

**Arguments:**

1. `task_description` - What to build
2. `working_directory` - Where to make changes

**Behavior:**

- Analyzes codebase structure
- Implements requested features
- Follows existing patterns
- Runs tests if configured

#### `/plan`

Plans implementation approach for complex tasks.

**Arguments:**

1. `task_description` - What needs to be done

**Output:**

- Structured implementation plan
- File modifications needed
- Testing approach
- Potential challenges

#### `/implement`

Executes a plan created by `/plan`.

**Arguments:**

1. `plan` - The plan to execute

### Infrastructure Commands

#### `/init_worktree`

Creates a new git worktree with sparse checkout.

**Arguments:**

1. `worktree_name` - Branch/directory name
2. `target_directory` - Directory to include in sparse checkout

#### `/make_worktree_name`

Generates descriptive worktree names.

**Arguments:**

1. `task_description` - Task being worked on
2. `prefix` - Optional prefix

**Output:** `feat-auth-middleware` or similar

## ⚙️ Task Configuration

### Notion Task Properties

Tasks in Notion can include special tags to control behavior:

```markdown
Title: Implement user authentication

--execute

{{model: x-ai/grok-4}} # Use grok-4
{{workflow: plan}} # Use plan-implement workflow
{{worktree: feat-auth}} # Specific worktree name
{{prototype: vite_vue}} # Generate a Vue.js app prototype
```

## 🚀 Rapid Prototyping with Specialized Templates

The prototype system enables instant application generation from natural language descriptions. Simply add a `{{prototype: type}}` tag to your Notion task and the system automatically generates complete, production-ready applications.

### Available Prototype Types

#### 1. **`uv_script`** - Python CLI Tools & Utilities

**Perfect for**: Command-line tools, automation scripts, data processing utilities, API clients
**Features**:

- Single-file Python script with inline dependency management
- Executable with `./app.py` (no pip install needed)
- Built-in argument parsing, logging, and error handling
- UV's lightning-fast dependency resolution

**Example Task:**

```markdown
Title: Create GitHub Issue Tracker

Build a CLI tool to track GitHub issues with:

- List issues for a repository
- Create new issues with templates
- Update issue status and labels
- Export to CSV format

{{prototype: uv_script}}
{{app: gh-tracker}}

execute
```

#### 2. **`vite_vue`** - Modern Web Applications

**Perfect for**: Dashboards, SPAs, admin panels, interactive web apps
**Features**:

- Vue 3 + TypeScript + Vite for blazing-fast development
- Responsive design with modern CSS frameworks
- Hot module replacement and instant builds
- Production-ready build pipeline with Bun

**Example Task:**

```markdown
Title: Build Personal Finance Dashboard

Create a financial tracking app with:

- Transaction import from CSV/bank APIs
- Expense categorization and budgeting
- Interactive charts and analytics
- Data export functionality

{{prototype: vite_vue}}
{{app: finance-dashboard}}

execute
```

#### 3. **`bun_scripts`** - TypeScript Backend Services

**Perfect for**: APIs, microservices, automation tools, build scripts
**Features**:

- Native TypeScript execution (no compilation step)
- Built-in SQLite database support
- File system and shell integration
- Lightning-fast startup and execution

**Example Task:**

```markdown
Title: Content Management API

Build a headless CMS API with:

- RESTful endpoints for articles/media
- SQLite database with migrations
- File upload and processing
- Authentication and permissions

{{prototype: bun_scripts}}
{{app: headless-cms}}

execute
```

#### 4. **`uv_mcp`** - AI Tool Integration Servers

**Perfect for**: Claude integrations, custom AI tools, automation bridges
**Features**:

- Model Context Protocol (MCP) server implementation
- Exposes custom tools to AI models like Claude
- Python-based with UV dependency management
- Secure tool execution and parameter validation

**Example Task:**

```markdown
Title: Database Query MCP Server

Create an MCP server that lets Claude:

- Connect to PostgreSQL/MySQL databases
- Execute safe SELECT queries
- Generate database schemas and documentation
- Export query results to various formats

{{prototype: uv_mcp}}
{{app: db-query-mcp}}

execute
```

### How Prototype Generation Works

When a prototype tag is detected:

1. **Specialized Planning**: Routes to `/plan_[prototype]` command with framework expertise
2. **Project Scaffolding**: Creates complete project structure in `apps/[app_name]/`
3. **Dependency Setup**: Configures package managers, dependencies, and dev tools
4. **Best Practices**: Implements framework conventions, security patterns, and optimization
5. **Documentation**: Generates README, usage instructions, and deployment guides

### Prototype Task Structure

**Required Elements:**

- **Title**: Descriptive name for your application
- **Description**: What the app should do (features, requirements, use cases)
- **Prototype tag**: `{{prototype: type}}` to specify technology stack
- **Execution trigger**: `--execute` to start the generation process

**Optional Elements:**

- **App name**: `{{app: my-app-name}}` for custom directory naming
- **Model preference**: `{{model: x-ai/grok-4}}` for complex applications
- **Additional constraints**: Technology preferences, specific libraries, etc.

**Complete Example:**

```markdown
Title: Real-time Chat Application

Build a modern chat application featuring:

- WebSocket-based real-time messaging
- User authentication and profiles
- Multiple chat rooms/channels
- Message history with search
- File sharing capabilities
- Mobile-responsive design

Requirements:

- Support 100+ concurrent users
- Modern, clean UI design
- Docker deployment ready

{{prototype: vite_vue}}
{{app: realtime-chat}}
{{model: x-ai/grok-4}}

execute
```

### Status Lifecycle

1. **Not started** → Task awaiting processing
2. **In progress** → Claimed by agent (includes ADW ID)
3. **Done** → Successfully completed (includes commit hash)
4. **Failed** → Error occurred (includes error details)
5. **HIL Review** → Human review needed

### Continue Prompts

Tasks can be iteratively refined:

```markdown
Status: HIL Review

--continue - Add input validation and error handling
```

The system will:

1. Pick up the task again
2. Read previous implementation
3. Apply requested changes
4. Update Notion with new results

## 🚀 Quick Start Guide

### Prerequisites

1. **Set up Notion Database**:

   - Create a Notion database for tasks with properties: Title, Status, Content
   - Get your database ID from the URL: `https://notion.so/yourworkspace/DATABASE_ID?v=...`

2. **Environment Setup** (.env file):

```bash
# Notion database ID for task tracking
NOTION_AGENTIC_TASK_TABLE_ID=your-database-id

# MCP configuration for Notion access (configured separately)
```

3. **Install System Dependencies**:

```bash
# Python dependencies with UV
uv pip install -r requirements.txt

# Additional prototype dependencies
curl -fsSL https://bun.sh/install | bash  # For Bun prototypes
curl -LsSf https://astral.sh/uv/install.sh | sh  # For UV prototypes
```

### Creating Your First Prototype

#### 1. Add Task to Notion

Create a new task in your Notion database:

```markdown
Title: Personal Task Manager

Build a simple task management app with:

- Add, edit, delete tasks
- Mark tasks as complete
- Filter by status (all, active, completed)
- Local storage persistence

{{prototype: vite_vue}}
{{app: task-manager}}

execute
```

#### 2. Start the Monitor

```bash
# Start continuous monitoring (polls every 15 seconds)
./adws/adw_triggers/adw_trigger_cron_notion_tasks.py

# OR run once to process current tasks
./adws/adw_triggers/adw_trigger_cron_notion_tasks.py --once
```

#### 3. Watch the Magic

The system will:

1. **Detect** your prototype task
2. **Claim** it (status changes to "In progress")
3. **Generate** a comprehensive plan using `/plan_vite_vue`
4. **Implement** the complete Vue.js application
5. **Update** Notion with results and commit hash

#### 4. Access Your Application

```bash
cd apps/task-manager
bun run dev  # Start development server
```

### Monitor Configuration

#### Basic Usage

```bash
# Default configuration (15-second polling)
./adws/adw_triggers/adw_trigger_cron_notion_tasks.py

# Custom polling interval
./adws/adw_triggers/adw_trigger_cron_notion_tasks.py --interval 30

# Limit concurrent prototypes
./adws/adw_triggers/adw_trigger_cron_notion_tasks.py --max-tasks 2

# Dry run to test configuration
./adws/adw_triggers/adw_trigger_cron_notion_tasks.py --dry-run
```

#### Advanced Options

- `--interval SECONDS`: Polling frequency (default: 15)
- `--database-id ID`: Override Notion database ID
- `--max-tasks N`: Maximum concurrent prototypes (default: 3)
- `--status-filter JSON`: Custom status filter (default: `["Not started", "HIL Review"]`)
- `--dry-run`: Preview mode without making changes
- `--once`: Process current tasks and exit

### Monitoring

The system provides real-time feedback:

1. **Console Output**: Colorized status updates
2. **Live Feed**: `live_feed_from_adw_trigger_cron_notion_tasks.txt`
3. **Notion Updates**: Real-time status changes
4. **Git History**: Commits in worktree branches

## 📋 Best Practices for Rapid Prototyping

### Writing Effective Prototype Descriptions

#### ✅ Excellent Prototype Tasks

**Comprehensive Feature List:**

```markdown
Title: E-commerce Product Catalog

Build a product browsing system with:

- Product grid with search and filtering
- Category-based navigation
- Product detail pages with image galleries
- Shopping cart functionality
- Responsive design for mobile/desktop
- Local storage for cart persistence

{{prototype: vite_vue}}
{{app: product-catalog}}

execute
```

**Clear Technical Requirements:**

```markdown
Title: Log Analysis Tool

Create a CLI tool that:

- Parses Apache/Nginx log files
- Generates traffic statistics and reports
- Filters by date range, IP, status codes
- Exports results to JSON/CSV formats
- Handles large files efficiently (streaming)

Technical requirements:

- Accept log file paths as arguments
- Support both common log formats
- Include error handling for malformed lines

{{prototype: uv_script}}
{{app: log-analyzer}}

execute
```

#### ❌ Avoid These Common Mistakes

**Too Vague:**

```markdown
Title: Build a website
Make me a website with some features
{{prototype: vite_vue}}
execute
```

**No Specific Features:**

```markdown
Title: API Server
Build an API for my app
{{prototype: bun_scripts}}
execute
```

### Choosing the Right Prototype Type

#### When to Use Each Prototype

**`uv_script`** - Choose for:

- ✅ Command-line tools and utilities
- ✅ Data processing and automation scripts
- ✅ API clients and integrations
- ✅ System administration tools
- ❌ Web applications or servers
- ❌ Real-time or interactive applications

**`vite_vue`** - Choose for:

- ✅ Interactive web applications
- ✅ Dashboards and admin interfaces
- ✅ SPAs with complex user interactions
- ✅ Data visualization applications
- ❌ Command-line tools
- ❌ Background services or APIs

**`bun_scripts`** - Choose for:

- ✅ REST APIs and web services
- ✅ Database-driven applications
- ✅ Build tools and automation
- ✅ Real-time applications (WebSockets)
- ❌ Browser-based interfaces
- ❌ Simple one-off scripts

**`uv_mcp`** - Choose for:

- ✅ Custom AI tool integrations
- ✅ Bridging external APIs to Claude
- ✅ Database query interfaces for AI
- ✅ Specialized data processing tools
- ❌ General-purpose applications
- ❌ User-facing interfaces

### Iteration and Refinement

Use the "continue" pattern for iterative improvements:

**Initial Task:**

```markdown
Title: Task Tracker Dashboard
{{prototype: vite_vue}}
execute
```

**After Review (Status: HIL Review):**

```markdown
--continue - Add user authentication with login/logout
--continue - Implement drag-and-drop task reordering
--continue - Add dark mode toggle and theme persistence
```

### Performance and Scaling

#### For Production-Ready Prototypes

**Include Performance Requirements:**

```markdown
Requirements:

- Support 1000+ concurrent users
- Sub-200ms API response times
- Mobile-first responsive design
- SEO-friendly routing

{{model: x-ai/grok-4}} # Use for complex optimization
```

**Specify Deployment Needs:**

```markdown
Deployment requirements:

- Docker container ready
- Environment variable configuration
- Health check endpoints
- Logging and monitoring hooks
```

### Worktree Management

Worktrees are created in `../trees/[worktree-name]/`. Each contains:

- Sparse checkout of project directory
- Isolated branch for changes
- Complete git history

Clean up old worktrees:

```bash
git worktree list
git worktree remove ../trees/old-feature
```

## 🎨 Customization

### Adding New Commands

1. Create command definition in `.claude/commands/`:

```markdown
# .claude/commands/my_command.md

## My Custom Command

Description of what the command does...

### Arguments

1. `arg1` - Description
2. `arg2` - Description

### Usage

Explain when and how to use this command...
```

2. Implement command handler in `adws/`:

```python
def execute_my_command(args):
    # Implementation
    pass
```

### Custom Workflows

Create new workflow by combining existing commands:

```python
# adws/adw_my_custom_workflow.py

# Phase 1: Planning
execute_template(AgentTemplateRequest(
    slash_command="/plan",
    args=[task_description]
))

# Phase 2: Implementation
execute_template(AgentTemplateRequest(
    slash_command="/my_command",
    args=[custom_args]
))

# Phase 3: Update Notion
execute_template(AgentTemplateRequest(
    slash_command="/update_notion_task",
    args=[page_id, status, details]
))
```

## 🔍 Troubleshooting Rapid Prototyping

### Common Prototype Issues

1. **Prototype tasks not being detected**

   - ✅ Verify `{{prototype: type}}` tag is properly formatted
   - ✅ Check task has `execute` trigger in content
   - ✅ Ensure status is "Not started" or "HIL Review"
   - ✅ Confirm `NOTION_AGENTIC_TASK_TABLE_ID` environment variable is set

2. **Worktree creation fails**

   - ✅ Ensure git repository is clean with no uncommitted changes
   - ✅ Check that worktree name doesn't already exist in `trees/`
   - ✅ Verify sufficient disk space for new worktree
   - ✅ Confirm you're in the project root directory

3. **Plan generation errors**

   - ✅ Make sure task description is detailed enough
   - ✅ Verify prototype type is one of: `uv_script`, `vite_vue`, `bun_scripts`, `uv_mcp`
   - ✅ Check that `specs/` directory exists and is writable

4. **Application scaffolding fails**

   - **For `vite_vue`**: Ensure Bun is installed (`curl -fsSL https://bun.sh/install | bash`)
   - **For `uv_script`**: Verify UV is installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
   - **For `bun_scripts`**: Confirm Bun runtime is available in PATH
   - **For `uv_mcp`**: Check Python 3.10+ is installed

5. **Notion updates failing**
   - ✅ Verify MCP Notion server is running and configured
   - ✅ Check Notion API permissions for database access
   - ✅ Review `.mcp.json` configuration file
   - ✅ Test Notion connection with `/get_notion_tasks` manually

### Debugging Prototype Generation

#### Enable Verbose Mode

```bash
# Run with detailed output
./adws/adw_triggers/adw_trigger_cron_notion_tasks.py --dry-run

# Check individual prototype planning
cd trees/your-worktree/agentic_prototyping
claude /plan_vite_vue your-adw-id "your task description"
```

#### Check Generated Files

```bash
# Verify plan was created
ls -la specs/plan-*

# Check application structure
ls -la apps/your-app-name/

# Review implementation logs
ls -la agents/your-adw-id/
```

#### Test Prototype Applications

```bash
# UV scripts
cd apps/your-app && uv run ./main.py --help

# Vite Vue apps
cd apps/your-app && bun install && bun run dev

# Bun scripts
cd apps/your-app && bun run index.ts

# MCP servers
cd apps/your-app && uv run mcp-server
```

### Performance Optimization

#### For High-Volume Prototyping

```bash
# Increase concurrent prototype limit
./adws/adw_triggers/adw_trigger_cron_notion_tasks.py --max-tasks 5

# Reduce polling interval for faster pickup
./adws/adw_triggers/adw_trigger_cron_notion_tasks.py --interval 10

# Use THINKING model for complex prototypes
# Add {{model: x-ai/grok-4}} tag to Notion tasks
```

#### Worktree Cleanup

```bash
# List all worktrees
git worktree list

# Remove old prototype worktrees
git worktree remove trees/old-prototype-name

# Prune removed worktrees
git worktree prune
```

### Debug Mode

Enable verbose logging:

```bash
# Set in environment
export DEBUG=true

# Or in Python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 💡 Architecture Benefits

1. **Scalability**: Add more agents by increasing `--max-tasks`
2. **Reliability**: Detached processes survive monitor restarts
3. **Traceability**: Every change linked to Notion task and git commit
4. **Flexibility**: Easy to add new workflows and commands
5. **Isolation**: Worktrees prevent interference between tasks

## 🤝 Contributing

1. **Adding Commands**: Place in `.claude/commands/`
2. **New Workflows**: Add to `adws/` directory
3. **Data Models**: Extend `adw_modules/data_models.py`
4. **Utilities**: Add to `adw_modules/utils.py`

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

Built with:

- Claude (Anthropic) for AI agents
- Notion API for task management
- Git worktrees for isolation
- Python + UV for orchestration
