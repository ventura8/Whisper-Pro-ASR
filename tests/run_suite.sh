#!/bin/bash
# Script to run the test suite and linting in Docker
set -e
set -o pipefail

# Change to the project root directory
cd "$(dirname "$0")/.."

if [ "$SKIP_LINT" != "1" ]; then
  echo "--- Running AutoPEP8 ---"
  python3 -m autopep8 --in-place --recursive --max-line-length 100 modules tests whisper_server.py scripts

  echo "--- Running Pylint ---"
  python3 -m pylint modules tests whisper_server.py --recursive=y
  
  echo "--- Running Yamllint ---"
  yamllint .
else
  echo "--- Skipping Linting (LINT job already completed) ---"
fi

echo ""
echo "--- Running Pytest with Coverage ---"
# We output XML (for PR display) and terminal report
python3 -m pytest --cov=modules --cov=whisper_server --cov-report=xml:coverage.xml --cov-report=term-missing | tee coverage_output.txt

echo ""
echo "--- Verifying Per-File Coverage (Threshold: 90%) ---"
python3 tests/check_coverage.py

echo ""
echo "--- Generating Coverage Badge ---"
# Create assets directory if it doesn't exist
mkdir -p assets
python3 -m coverage_badge -o assets/coverage.svg -f

echo ""
echo "--- Test Suite Completed Successfully ---"
