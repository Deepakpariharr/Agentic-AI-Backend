#!/usr/bin/env bash
# Run the test suite with coverage, then check that the lines changed on this
# branch are covered. Same checks the "PR Checks" workflow runs on GitHub.
#
#   scripts/run_tests.sh                        # compare against origin/master
#   BASE_BRANCH=origin/qa scripts/run_tests.sh  # compare against another branch
#   DIFF_COVER_MIN=90 scripts/run_tests.sh      # stricter bar for changed lines
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

BASE_BRANCH="${BASE_BRANCH:-origin/master}"
DIFF_COVER_MIN="${DIFF_COVER_MIN:-80}"

PYTHON=".venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"

# Keep the base branch current so the diff only contains *your* changes.
git fetch --quiet origin "${BASE_BRANCH#origin/}" || echo "⚠️  Could not fetch $BASE_BRANCH, using local copy"

echo "▶ Running tests with coverage"
"$PYTHON" -m pytest

echo
echo "▶ Coverage of lines changed vs $BASE_BRANCH (minimum ${DIFF_COVER_MIN}%)"
"$PYTHON" -m diff_cover.diff_cover_tool coverage.xml \
    --compare-branch="$BASE_BRANCH" \
    --fail-under="$DIFF_COVER_MIN"
