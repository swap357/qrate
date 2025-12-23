# Setup Guide

Quick guide to get started with development.

## Prerequisites

- Python 3.11 or higher
- pip (usually comes with Python)
- git

## Initial Setup

### 1. Clone the Repository

```bash
git clone https://github.com/USERNAME/PACKAGE_NAME.git
cd PACKAGE_NAME
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate     # On Windows
```

### 3. Install Package

```bash
# Install in development mode with all dev dependencies
pip install -e ".[dev]"

# Or just development dependencies
pip install -e ".[dev,docs]"
```

### 4. Verify Installation

```bash
# Check version
python -c "import PACKAGE_NAME; print(PACKAGE_NAME.__version__)"

# Run tests
make test
```

## Development Workflow

### Daily Commands

```bash
# Activate virtual environment (if not already active)
source .venv/bin/activate

# Run tests
make test

# Format code before committing
make format

# Check code quality
make lint

# Type check
make type-check

# All quality checks
make all-checks
```

### Making Changes

1. **Create a branch:**
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes** in `src/PACKAGE_NAME/`

3. **Add tests** in `tests/`

4. **Run checks:**
   ```bash
   make all-checks
   ```

5. **Commit:**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

6. **Push and create PR:**
   ```bash
   git push origin feature/my-new-feature
   ```

### Running Examples

```bash
# Make sure virtual environment is active
python examples/basic_example.py

# Check generated output
ls examples/output/
```

## Testing

### Quick Test

```bash
pytest
```

### With Coverage

```bash
# Generate coverage report
pytest --cov=src/PACKAGE_NAME --cov-report=html

# Open report in browser
open htmlcov/index.html  # macOS
# xdg-open htmlcov/index.html  # Linux
# start htmlcov/index.html     # Windows
```

### Specific Tests

```bash
# Single file
pytest tests/test_core.py

# Single test
pytest tests/test_core.py::test_function_name

# With verbose output
pytest -v

# Stop on first failure
pytest -x
```

## Code Quality Tools

### Linting (Ruff)

```bash
# Check for issues
ruff check src tests

# Auto-fix issues
ruff check --fix src tests
```

### Formatting (Black + isort)

```bash
# Format all code
make format

# Or separately
black src tests
isort src tests

# Check without changing
black --check src tests
```

### Type Checking (mypy)

```bash
mypy src
```

## Building for Distribution

### Local Build

```bash
# Build wheel and source distribution
make build

# Output in dist/
ls dist/
```

### Install from Local Build

```bash
# Install the wheel
pip install dist/PACKAGE_NAME-0.1.0-py3-none-any.whl

# Test it works
python -c "import PACKAGE_NAME; print(PACKAGE_NAME.__version__)"
```

## Publishing to PyPI

### First Time Setup

1. **Create PyPI accounts:**
   - [PyPI](https://pypi.org/account/register/)
   - [TestPyPI](https://test.pypi.org/account/register/) (for testing)

2. **Install publishing tools:**
   ```bash
   pip install build twine
   ```

3. **Configure credentials:**
   ```bash
   # Create ~/.pypirc (or use keyring)
   [pypi]
   username = __token__
   password = pypi-YOUR-TOKEN-HERE
   
   [testpypi]
   username = __token__
   password = pypi-YOUR-TESTPYPI-TOKEN
   ```

### Publishing Process

1. **Update version** in `pyproject.toml`

2. **Update CHANGELOG.md**

3. **Create git tag:**
   ```bash
   git tag -a v0.1.0 -m "Release version 0.1.0"
   git push origin v0.1.0
   ```

4. **Test on TestPyPI:**
   ```bash
   make build
   twine upload --repository testpypi dist/*
   
   # Test installation
   pip install --index-url https://test.pypi.org/simple/ PACKAGE_NAME
   ```

5. **Publish to PyPI:**
   ```bash
   twine upload dist/*
   ```

6. **Verify:**
   ```bash
   pip install PACKAGE_NAME
   ```

## Troubleshooting

### Import Errors

```bash
# Reinstall in development mode
pip install -e ".[dev]"

# Or refresh installation
pip install --force-reinstall -e ".[dev]"
```

### Test Failures

```bash
# Clear pytest cache
rm -rf .pytest_cache

# Run with verbose output
pytest -vv

# Run with print statements visible
pytest -s
```

### Type Check Errors

```bash
# Check mypy configuration
mypy --config-file pyproject.toml src

# Ignore specific errors (use sparingly)
# Add to pyproject.toml [[tool.mypy.overrides]] sections
```

### Coverage Not Working

```bash
# Reinstall with coverage
pip install pytest-cov

# Check coverage configuration in pyproject.toml
```

## Editor Setup

### VS Code

Create `.vscode/settings.json`:
```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"]
}
```

### PyCharm

1. Set Python interpreter to `.venv/bin/python`
2. Enable pytest in Settings → Tools → Python Integrated Tools
3. Set Black as formatter in Settings → Tools → Black
4. Enable mypy in Settings → Tools → Python Integrated Tools → Type Checking

## Tips

- **Always activate virtual environment** before working
- **Run tests frequently** with `make test`
- **Format before committing** with `make format`
- **Keep dependencies updated** with `pip list --outdated`
- **Use make help** to see all available commands

## Next Steps

1. Read [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines
2. Check [examples/](examples/) for usage examples
3. Read main [README.md](README.md) for API documentation
4. Browse [tests/](tests/) to understand testing patterns

## Getting Help

- Check existing [Issues](https://github.com/USERNAME/PACKAGE_NAME/issues)
- Read the [Documentation](https://PACKAGE_NAME.readthedocs.io)
- Ask in [Discussions](https://github.com/USERNAME/PACKAGE_NAME/discussions)

Happy coding! 🚀


