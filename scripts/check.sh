#!/bin/bash
# Run all checks from CI workflow locally

set -e

echo "Running ruff check..."
ruff check src tests

echo "Running ruff format check..."
ruff format --check src tests

echo "Running mypy..."
mypy src

echo "Running pytest with coverage..."
pytest --cov=src/qrate --cov-report=term-missing --cov-fail-under=80

echo "✅ All checks passed!"

