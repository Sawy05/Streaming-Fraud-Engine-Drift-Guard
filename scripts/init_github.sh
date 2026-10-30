#!/usr/bin/env bash
set -euo pipefail

REPO_SLUG="Sawy05/streaming-fraud-mlops"
DEFAULT_BRANCH="main"

echo "=================================================================="
echo " Initializing Git & Publishing to GitHub: ${REPO_SLUG}"
echo "=================================================================="

if ! command -v git >/dev/null 2>&1; then
    echo "[ERROR] git is not installed or not on PATH." >&2
    exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
    echo "[ERROR] GitHub CLI (gh) is not installed. Install from https://cli.github.com/ and run 'gh auth login'." >&2
    exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
    echo "[ERROR] GitHub CLI (gh) is not authenticated. Run 'gh auth login' first." >&2
    exit 1
fi

if [ ! -d ".git" ]; then
    echo "[INFO] Initializing local Git repository on branch '${DEFAULT_BRANCH}'..."
    git init -b "${DEFAULT_BRANCH}"
else
    git branch -M "${DEFAULT_BRANCH}"
fi

if ! git config user.name >/dev/null 2>&1; then
    git config user.name "Sawy05"
fi
if ! git config user.email >/dev/null 2>&1; then
    git config user.email "Sawy05@users.noreply.github.com"
fi

echo "[INFO] Staging project files..."
git add .

if git diff --cached --quiet; then
    echo "[INFO] No uncommitted changes to commit."
else
    git commit -m "feat: initial release of real-time streaming fraud engine & drift guard MLOps pipeline"
fi

if gh repo view "${REPO_SLUG}" >/dev/null 2>&1; then
    echo "[INFO] Remote repository '${REPO_SLUG}' already exists."
    if ! git remote get-url origin >/dev/null 2>&1; then
        git remote add origin "https://github.com/${REPO_SLUG}.git"
    fi
    git push -u origin "${DEFAULT_BRANCH}"
else
    echo "[INFO] Creating public GitHub repository '${REPO_SLUG}' and pushing '${DEFAULT_BRANCH}'..."
    gh repo create "${REPO_SLUG}" --public --source=. --remote=origin --push
fi

echo "[SUCCESS] Repository published to https://github.com/${REPO_SLUG}"
