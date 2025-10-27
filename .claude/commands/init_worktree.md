# Initialize Worktree with Sparse Checkout

Create a new git worktree for an agent to work in isolation, with only the specified directory checked out.

## Variables

worktree_name: $1 (IMPORTANT: first argument)
target_directory: $2 (IMPORTANT: second argument)

## Instructions

Execute ALL the following steps in sequence WITHOUT using a todo list:

1. IMPORTANT Parse the variables - the first will be the worktree_name while the second (after the first space) will be the target_directory
2. **List existing worktrees** to check for name conflicts
3. Create a new git worktree in the `trees/<worktree_name>` directory with sparse checkout
4. Configure sparse checkout to only include `<target_directory>`
5. Base the worktree on the main branch
6. Copy the `.env` file from the root directory to the worktree (if it exists)
7. Create an initial commit in the worktree to establish the branch
8. Report the successful creation of the worktree

## Git Worktree Setup with Sparse Checkout

Execute these steps in order:

1. **List existing worktrees to avoid collisions**:

   ```bash
   git worktree list
   ```

   **WARNING**: DO NOT use any of the existing worktree names shown above! Each worktree must have a unique name.

   - Check if the proposed `<worktree_name>` appears in the list
   - If it does, STOP and report: "Worktree name '<worktree_name>' is already in use. Please choose a different name."

2. **Create the trees directory** if it doesn't exist:

   ```bash
   mkdir -p trees
   ```

3. **Check if worktree directory already exists**:

   - If `trees/<worktree_name>` already exists, report that it exists and stop
   - Otherwise, proceed with creation

4. **Create the git worktree without checkout**:

   ```bash
   git worktree add --no-checkout trees/<worktree_name> -b <worktree_name>
   ```

5. **Configure sparse checkout for the target directory**:

   ```bash
   # Initialize sparse checkout (use -C to avoid cd)
   git -C trees/<worktree_name> sparse-checkout init --cone

   # Set sparse checkout to only include the target directory
   git -C trees/<worktree_name> sparse-checkout set <target_directory>

   # Now checkout the files
   git -C trees/<worktree_name> checkout
   ```

6. **Copy environment file** (if exists):

   ```bash
   if [ -f .env ]; then
     cp .env trees/<worktree_name>/<target_directory>/.env
     echo "Copied .env file"
   else
     echo ".env file not found in root, skipping"
   fi
   ```

7. **Create initial commit with no changes**:

   ```bash
   git -C trees/<worktree_name> commit --allow-empty -m "Initial worktree setup for <worktree_name> with sparse checkout of <target_directory>"
   ```

## Error Handling

- **If the worktree name is already in use**, report this immediately and exit: "Worktree name '<worktree_name>' is already in use. Please choose a different name."
- If the worktree directory already exists, report this and exit gracefully
- If git worktree creation fails, report the error
- If sparse-checkout configuration fails, report the error
- If .env doesn't exist in root or target directory, continue without error (it's optional)

## Verification

After setup, verify the sparse checkout is working:

```bash
ls -la trees/<worktree_name>  # Should only show <target_directory> directory (plus .git)
git -C trees/<worktree_name> sparse-checkout list  # Should show: <target_directory>
```

## Report

Report one of the following:

- Success: "Worktree '<worktree_name>' created successfully at trees/<worktree_name> with only <target_directory> checked out"
- Already exists: "Worktree '<worktree_name>' already exists at trees/<worktree_name>"
- Error: "Failed to create worktree: <error message>"

## Notes

- Git worktrees with sparse checkout provide double isolation:
  - **Worktree isolation**: Separate branch and working directory
  - **Sparse checkout**: Only the relevant app directory is present
- This reduces clutter and prevents accidental modifications to other apps
- The agent only sees and works with `<target_directory>`
- Full repository history is still available but only the specified directory is in the working tree
- Each worktree maintains its own sparse-checkout configuration
