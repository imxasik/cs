#!/usr/bin/env bash
# Quick commit & push helper (Termux-friendly).
#
#   ./git.sh                     -> prompts for a commit message
#   ./git.sh "fixed cone bug"    -> uses the given message
#
# Pushes the CURRENT branch (not hardcoded to main).
set -e
cd "$(dirname "$0")"

BRANCH=$(git rev-parse --abbrev-ref HEAD)

git add -A

if git diff --cached --quiet; then
  echo "Nothing to commit — working tree clean."
  exit 0
fi

if [ -n "$1" ]; then
  commit_msg="$1"
else
  echo "Enter commit message (or press Enter for default):"
  read -r commit_msg
  [ -z "$commit_msg" ] && commit_msg="Auto update from Termux"
fi

git commit -m "$commit_msg"
git push origin "$BRANCH"
echo "✓ Pushed '$commit_msg' to origin/$BRANCH"
