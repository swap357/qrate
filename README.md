# Python Package Template

A Python package template with modern tooling and practices.

## Quick Start

### 1. Use This Template

1. Click "Use this template" on GitHub
2. Clone your new repository
3. Run the setup script to customize for your project:

```bash
cd your-new-project
python setup_template.py
```

This will prompt you for:
- Package name
- Project description
- Author name and email
- License type

### 2. Manual Setup

If you prefer manual setup:

1. Replace all instances of `PACKAGE_NAME` with your package name
2. Update author information in `pyproject.toml`
3. Update package description and keywords
4. Rename `src/PACKAGE_NAME/` to `src/your_package/`

### 3. Set Up Development Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests to verify setup
make test
```

## Development Workflow

### Common Commands

```bash
make help          # Show all available commands
make test          # Run tests with coverage
make lint          # Check code quality
make format        # Format code
make type-check    # Run type checker
make clean         # Remove build artifacts
make build         # Build distribution packages
```

### Before Committing

Always run these checks:

```bash
make format lint type-check test
```

## Project Structure

```
your-package/
├── src/
│   └── your_package/
│       ├── __init__.py      # Package exports and version
│       ├── core.py          # Core functionality
│       └── utils.py         # Utility functions
├── tests/
│   ├── __init__.py
│   ├── test_core.py
│   └── test_utils.py
├── examples/
│   ├── README.md
│   └── basic_example.py
├── docs/                    # Optional: Sphinx documentation
├── .github/                 # Optional: CI/CD workflows
├── pyproject.toml           # Project configuration
├── Makefile                 # Development commands
├── MANIFEST.in              # Additional files to include in distribution
├── .gitignore              # Git ignore patterns
├── LICENSE                  # Project license
├── README.md               # Main documentation
├── CONTRIBUTING.md         # Contribution guidelines
├── CHANGELOG.md            # Version history
└── SETUP.md                # Setup and development guide
```

## Configuration Files

### pyproject.toml

Central configuration for:
- Project metadata (name, version, description, authors)
- Dependencies (runtime and development)
- Build system (hatchling)
- Tool configurations (ruff, black, isort, mypy, pytest, coverage)

### Makefile

Convenient shortcuts for development tasks. All tools are properly configured in `pyproject.toml`.

### MANIFEST.in

Specifies additional files to include in source distributions.

## Publishing to PyPI

### First Time Setup

1. Create accounts on [PyPI](https://pypi.org/) and [TestPyPI](https://test.pypi.org/)
2. Create API tokens for both services
3. Configure credentials (use keyring or `.pypirc`)

### Publishing Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create git tag:
   ```bash
   git tag -a v0.1.0 -m "Release version 0.1.0"
   git push origin v0.1.0
   ```
4. Build and publish:
   ```bash
   make build
   # Test on TestPyPI first
   twine upload --repository testpypi dist/*
   # Then publish to PyPI
   twine upload dist/*
   ```

## Documentation

### Code Documentation

- Use Google-style docstrings
- Add type hints to all functions
- Include examples in docstrings

### User Documentation

- Keep README.md updated with usage examples
- Use CONTRIBUTING.md for development guidelines
- Track changes in CHANGELOG.md
- Add setup instructions in SETUP.md

## Testing

### Writing Tests

```python
def test_feature():
    """Test description following AAA pattern."""
    # Arrange
    input_data = "test"
    
    # Act
    result = my_function(input_data)
    
    # Assert
    assert result == expected_output
```

### Running Tests

```bash
# All tests
pytest

# With coverage report
pytest --cov=src/your_package --cov-report=html

# Specific test file
pytest tests/test_core.py

# Specific test function
pytest tests/test_core.py::test_feature
```

## Code Quality

### Linting (Ruff)

Fast Python linter combining multiple tools:
```bash
ruff check src tests
```

### Formatting (Black + isort)

Automatic code formatting:
```bash
black src tests
isort src tests
```

### Type Checking (mypy)

Static type analysis:
```bash
mypy src
```

## Best Practices

1. **Type Hints**: Add type hints to all public functions
2. **Documentation**: Write clear docstrings with examples
3. **Testing**: Aim for >90% code coverage
4. **Code Style**: Follow PEP 8 and use provided formatters
5. **Commits**: Use conventional commit messages
6. **Dependencies**: Pin versions in production, ranges in libraries

## Customization

### Adding Dependencies

**Runtime dependencies** (required for users):
```toml
dependencies = [
    "numpy>=1.24.0",
    "requests>=2.31.0",
]
```

**Development dependencies** (for contributors):
```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
]
```

### Adding Scripts/CLI

Add console scripts in `pyproject.toml`:
```toml
[project.scripts]
my-command = "your_package.cli:main"
```

## CI/CD

Optional: Add GitHub Actions for automated testing and deployment.

Create `.github/workflows/test.yml`:
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -e ".[dev]"
      - run: make test lint type-check
```

## License

Choose a license appropriate for your project:
- MIT: Permissive, allows commercial use
- Apache 2.0: Permissive with patent grant
- GPL: Copyleft, requires derivative works to be open source
- BSD: Similar to MIT, commonly used

Update `LICENSE` file and `pyproject.toml` accordingly.

## Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [PEP 621](https://peps.python.org/pep-0621/) - Storing project metadata in pyproject.toml
- [Hatchling](https://hatch.pypa.io/latest/) - Modern build backend
- [pytest](https://docs.pytest.org/) - Testing framework
- [ruff](https://docs.astral.sh/ruff/) - Fast Python linter

---

Built with ♥️ by [@swap357](https://github.com/swap357)


