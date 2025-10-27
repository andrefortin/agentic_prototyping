#!/bin/bash

# Script to clean up git worktrees and their associated branches
# Worktrees are stored in trees/* directory

set -e

echo "🧹 Git Worktree Cleanup Summary"
echo "================================"

# First, prune any stale worktree entries
git worktree prune 2>/dev/null

# Check if trees directory exists
if [ ! -d "trees" ]; then
    echo "✓ No trees directory found. Nothing to clean."
    exit 0
fi

# Get list of worktrees in trees directory
worktrees=$(git worktree list --porcelain | grep "^worktree.*trees/" | cut -d' ' -f2 | xargs -n1 basename 2>/dev/null || true)
orphaned_dirs=$(ls -A trees 2>/dev/null || true)

# Get ALL branches that were created by worktrees (including those without directories)
# Look for branches from: worktrees, directories in trees/, and any branch containing the project name
worktree_branches=$(git worktree list --porcelain | grep "^branch" | sed 's/^branch refs\/heads\///' | grep -v "^main$" | grep -v "^master$" || true)
dir_branches=""
if [ -n "$orphaned_dirs" ]; then
    for dir in $orphaned_dirs; do
        dir_branches="$dir_branches$dir "
    done
fi

# Also find branches that contain common worktree patterns
pattern_branches=$(git branch --format="%(refname:short)" | grep -E "(feat-|fix-|task-|agentic_prototyping)" | grep -v "^main$" | grep -v "^master$" || true)

# Combine all branches and remove duplicates
all_branches=$(echo "$worktree_branches $dir_branches $pattern_branches" | tr ' ' '\n' | sort -u | tr '\n' ' ')

# Build summary
echo -e "\nThis will:"

if [ -n "$worktrees" ]; then
    echo "• Remove git worktrees and branches:"
    echo "$worktrees" | while read -r worktree; do
        [ -n "$worktree" ] && echo "  - trees/$worktree (and branch: $worktree)"
    done
fi

if [ -n "$orphaned_dirs" ]; then
    echo "• Remove ALL directories in trees/:"
    for dir in $orphaned_dirs; do
        echo "  - trees/$dir"
    done
fi

if [ -n "$all_branches" ]; then
    echo "• Remove branches (worktree-related and pattern-matched):"
    for branch in $all_branches; do
        [ -n "$branch" ] && echo "  - $branch"
    done
fi

[ -z "$worktrees" ] && [ -z "$orphaned_dirs" ] && [ -z "$all_branches" ] && echo "• Nothing to remove"

# Single confirmation
read -p $'\nProceed? (y/N): ' -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Remove worktrees and branches
if [ -n "$worktrees" ]; then
    echo "$worktrees" | while read -r worktree; do
        if [ -n "$worktree" ]; then
            git worktree remove "trees/$worktree" --force 2>/dev/null || true
            git branch -D "$worktree" 2>/dev/null || true
        fi
    done
fi

# Remove ALL directories in trees/ (both orphaned and any remaining)
if [ -n "$orphaned_dirs" ]; then
    for dir in $orphaned_dirs; do
        rm -rf "trees/$dir" 2>/dev/null || true
        # Also try to delete any branch with this name
        git branch -D "$dir" 2>/dev/null || true
    done
fi

# Remove any remaining branches that match the pattern
if [ -n "$all_branches" ]; then
    for branch in $all_branches; do
        git branch -D "$branch" 2>/dev/null || true
    done
fi

# Final cleanup
git worktree prune

# Remove trees directory if empty
if [ -d "trees" ] && [ -z "$(ls -A trees 2>/dev/null)" ]; then
    rmdir trees
fi

echo "✅ Cleanup complete!"