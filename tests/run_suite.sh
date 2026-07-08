#!/bin/bash
# Script to run the test suite and linting in Docker
set -e
set -o pipefail

# Change to the project root directory
cd "$(dirname "$0")/.."

# Activate virtual environment if running locally and it exists
if [ "$CI" != "true" ]; then
  if [ -d ".venv" ]; then
    source .venv/bin/activate
  elif [ -d "venv" ]; then
    source venv/bin/activate
  fi
fi


if [ "$SKIP_LINT" != "1" ]; then
  echo "--- Running Ruff (check) ---"
  python3 -m ruff check .

  echo "--- Running Ruff (format) ---"
  python3 -m ruff format --check .

  echo "--- Running Pylint ---"
  python3 -m pylint modules whisper_pro_asr.py tests
  
  echo "--- Running Yamllint ---"
  yamllint .
else
  echo "--- Skipping Linting (LINT job already completed) ---"
fi

echo ""
echo "--- Running Pytest with Coverage ---"
# We output XML (for PR display) and terminal report
python3 -m pytest --cov=. --cov-report=xml:coverage.xml --cov-report=term-missing --junitxml=pytest.xml | tee coverage_output.txt

echo ""
echo "--- Verifying Per-File Coverage (Threshold: 90%) ---"
python3 tests/check_coverage.py
# Only generate the badge if not running in CI and runtime deps are available
if [ "$CI" != "true" ] && python3 -c "import coverage_badge, pkg_resources" &>/dev/null; then
  echo ""
  echo "--- Generating Coverage Badge ---"
  # Create assets directory if it doesn't exist
  mkdir -p assets
  python3 -m coverage_badge -o assets/coverage.svg -f
else
  echo ""
  echo "--- Skipping Coverage Badge Generation (not required or dependencies missing) ---"
fi

echo ""
echo "--- Test Suite Completed Successfully ---"
