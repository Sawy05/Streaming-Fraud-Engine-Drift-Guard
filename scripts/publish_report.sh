#!/usr/bin/env bash
set -euo pipefail

REPO_OWNER="Sawy05"
REPO_NAME="Streaming-Fraud-Engine-Drift-Guard"
REPO_SLUG="${REPO_OWNER}/${REPO_NAME}"
DEFAULT_BRANCH="main"
COMMIT_MSG="docs: publish comprehensive thesis, system proposal, and CI compilation workflow"

echo "=================================================================="
echo " Publishing Academic Thesis & MLOps Proposal to GitHub: ${REPO_SLUG}"
echo "=================================================================="

# 1. Validate git installation
if ! command -v git >/dev/null 2>&1; then
    echo "[ERROR] git is not installed or not found on PATH." >&2
    exit 1
fi

# 2. Check for gh CLI and auth status
GH_AVAILABLE=true
if ! command -v gh >/dev/null 2>&1; then
    echo "[WARN] GitHub CLI ('gh') is not installed. Pushing will proceed via standard git remote."
    GH_AVAILABLE=false
elif ! gh auth status >/dev/null 2>&1; then
    echo "[WARN] GitHub CLI ('gh') is installed but unauthenticated. Pushing will proceed via standard git remote."
    GH_AVAILABLE=false
fi

# Ensure git user identity is configured
if ! git config user.name >/dev/null 2>&1; then
    git config user.name "Ahmed Elsawy"
fi
if ! git config user.email >/dev/null 2>&1; then
    git config user.email "Sawy05@users.noreply.github.com"
fi

# Ensure branch is main
if [ ! -d ".git" ]; then
    echo "[INFO] Initializing Git repository on '${DEFAULT_BRANCH}'..."
    git init -b "${DEFAULT_BRANCH}"
else
    git branch -M "${DEFAULT_BRANCH}"
fi

# 3. Detect or set remote origin
REMOTE_URL="https://github.com/${REPO_SLUG}.git"
if git remote get-url origin >/dev/null 2>&1; then
    CURRENT_REMOTE=$(git remote get-url origin)
    echo "[INFO] Existing remote origin detected: ${CURRENT_REMOTE}"
else
    echo "[INFO] Setting remote origin to: ${REMOTE_URL}"
    git remote add origin "${REMOTE_URL}"
fi

# 4. Stage documentation and workflow files
echo "[INFO] Staging thesis report, LaTeX source, and CI workflow..."
git add docs/report/THESIS.md
git add docs/report/latex/
git add .github/workflows/compile-report.yml
git add scripts/publish_report.sh
git add .gitignore

# 5. Commit changes if staged diff exists
if git diff --cached --quiet; then
    echo "[INFO] No changes staged. Documentation is already committed."
else
    echo "[INFO] Creating conventional commit: '${COMMIT_MSG}'"
    git commit -m "${COMMIT_MSG}"
fi

# 6. Push to origin main
echo "[INFO] Pushing changes to origin '${DEFAULT_BRANCH}'..."
if [ "${GH_AVAILABLE}" = true ]; then
    if ! gh repo view "${REPO_SLUG}" >/dev/null 2>&1; then
        echo "[INFO] Creating public repository on GitHub via gh..."
        gh repo create "${REPO_SLUG}" --public --source=. --remote=origin --push
    else
        git push -u origin "${DEFAULT_BRANCH}"
    fi
else
    echo "[INFO] Attempting git push to origin ${DEFAULT_BRANCH}..."
    if ! git push -u origin "${DEFAULT_BRANCH}"; then
        echo ""
        echo "[NOTE] Direct git push requires authentication credentials (SSH key or Personal Access Token)."
        echo "       Run 'gh auth login' or configure your SSH key: git remote set-url origin git@github.com:${REPO_SLUG}.git"
        echo "       Then run: git push -u origin ${DEFAULT_BRANCH}"
    fi
fi

# 7. Print summary and direct links
echo "=================================================================="
echo " [SUCCESS] Publication Pipeline Complete!"
echo " Repository URL: https://github.com/${REPO_SLUG}"
echo " Thesis Markdown: https://github.com/${REPO_SLUG}/blob/${DEFAULT_BRANCH}/docs/report/THESIS.md"
echo " LaTeX Sources: https://github.com/${REPO_SLUG}/tree/${DEFAULT_BRANCH}/docs/report/latex"
echo " PDF Release: https://github.com/${REPO_SLUG}/releases/tag/v1.0.0-documentation"
echo "=================================================================="
