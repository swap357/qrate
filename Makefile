.PHONY: help install dev-install test lint format type-check clean build publish

help:
	@echo "Python Package Development Commands"
	@echo "==================================="
	@echo "install        Install package for production use"
	@echo "dev-install    Install package with dev dependencies"
	@echo "test           Run test suite with coverage"
	@echo "lint           Run linter (ruff)"
	@echo "format         Format code (black, isort)"
	@echo "type-check     Run type checker (mypy)"
	@echo "clean          Remove build artifacts and cache files"
	@echo "build          Build distribution packages"
	@echo "publish        Publish to PyPI (requires credentials)"

install:
	pip install .

dev-install:
	pip install -e ".[dev]"

test:
	pytest --cov=src/PACKAGE_NAME --cov-report=term-missing --cov-report=html

lint:
	ruff check src tests

format:
	black src tests
	isort src tests

type-check:
	mypy src

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

publish: build
	twine check dist/*
	twine upload dist/*

# Development convenience targets
all-checks: format lint type-check test
	@echo "✅ All checks passed!"

watch-test:
	pytest-watch --clear --runner "pytest --cov=src/PACKAGE_NAME"


