#!/usr/bin/env bash
# Run the test suite with coverage, then check that the lines changed on this
# branch are covered, using only local git refs.
#
#   scripts/run_tests.sh              # compare against local main
#   BASE_BRANCH=qa scripts/run_tests.sh  # compare against another local branch
#   DIFF_COVER_MIN=90 scripts/run_tests.sh      # stricter bar for changed lines
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

BASE_BRANCH="${BASE_BRANCH:-main}"
DIFF_COVER_MIN="${DIFF_COVER_MIN:-80}"

PYTHON=".venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"

echo "▶ Running tests with coverage"
"$PYTHON" -m pytest

echo
echo "▶ Coverage of lines changed vs $BASE_BRANCH (minimum ${DIFF_COVER_MIN}%)"
"$PYTHON" -m diff_cover.diff_cover_tool coverage.xml \
    --compare-branch="$BASE_BRANCH" \
    --fail-under="$DIFF_COVER_MIN"
