# Contributing to [Package Name]

Thank you for your interest in contributing! This document provides guidelines and information for contributors.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Assume good intentions

## Development Setup

1. **Fork the repository**
   - Click "Fork" on GitHub
   - Clone your fork locally

2. **Set up development environment:**
   ```bash
   git clone https://github.com/yourusername/PACKAGE_NAME.git
   cd PACKAGE_NAME
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. **Verify setup:**
   ```bash
   make test
   ```

## Development Principles

This project follows these design principles:

1. **Make Illegal States Unrepresentable** - Encode invariants in types/data so bugs can't exist
2. **Minimal Surface, Maximal Clarity** - Consolidate files when possible; make APIs obvious and hard to misuse
3. **DRY & Orthogonality** - Reuse existing functions; each piece does one job cleanly
4. **Code is for Readers** - Prefer small, easily understandable code; optimize for readability and intent
5. **Iterative Design** - Take small steps; structure code so it can evolve or be safely removed
6. **Abstraction with Taste** - Abstract when it reduces complexity, duplicate when abstraction obscures intent

## Contribution Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test improvements

### 2. Make Your Changes

- Write clear, documented code
- Follow existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Run Quality Checks

```bash
# Format code
make format

# Check linting
make lint

# Type check
make type-check

# Run tests
make test

# Or run all checks at once
make all-checks
```

All checks must pass before submitting.

### 4. Commit Your Changes

Write clear commit messages following [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new feature X

- Implement feature X with Y approach
- Add tests for feature X
- Update documentation
```

Commit prefixes:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions/changes
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `chore:` - Maintenance tasks
- `ci:` - CI/CD changes

### 5. Submit a Pull Request

1. Push to your fork: `git push origin your-branch-name`
2. Open a pull request against `main`
3. Fill out the PR template with:
   - Clear description of changes
   - Related issue numbers
   - Testing performed
   - Screenshots (if applicable)
4. Wait for review and address feedback
5. Once approved, maintainers will merge

## Code Style

### Python Style

- Follow [PEP 8](https://pep8.org/)
- Use type hints for all public functions
- Maximum line length: 100 characters
- Use descriptive variable names
- Prefer explicit over implicit

### Documentation

Write docstrings for all public functions/classes using Google style:

```python
def my_function(x: int, y: int) -> int:
    """Calculate the sum of two numbers.
    
    This function demonstrates proper documentation style.
    
    Args:
        x: First input value
        y: Second input value
        
    Returns:
        The sum of x and y
        
    Raises:
        ValueError: If inputs are negative
        
    Examples:
        >>> my_function(2, 3)
        5
    """
    if x < 0 or y < 0:
        raise ValueError("Inputs must be non-negative")
    return x + y
```

### Testing

- Write tests for all new functionality
- Aim for >90% code coverage
- Use descriptive test names
- Follow AAA pattern: Arrange, Act, Assert

Example:
```python
def test_feature_behavior():
    """Test that feature behaves correctly with valid input."""
    # Arrange
    input_data = create_test_input()
    expected_output = "expected_result"
    
    # Act
    result = my_feature(input_data)
    
    # Assert
    assert result == expected_output
```

## Adding New Features

### Core Functionality

When adding to core modules:

1. **Discuss first** - Open an issue to discuss the feature
2. **Ensure fit** - Make sure it aligns with project goals
3. **Keep API minimal** - Add only what's necessary
4. **Test thoroughly** - Add comprehensive tests
5. **Document well** - Include docstrings and examples

### Examples

New examples are always welcome!

1. Create a new file in `examples/`
2. Add a clear docstring explaining what it demonstrates
3. Use `examples/output/` for generated files
4. Keep examples focused and educational
5. Update `examples/README.md`

## Testing

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src/PACKAGE_NAME --cov-report=html

# Specific test file
pytest tests/test_feature.py

# Specific test
pytest tests/test_feature.py::test_specific_case

# Watch mode (requires pytest-watch)
make watch-test
```

### Writing Good Tests

- Test one behavior per test function
- Use fixtures for common setup
- Mock external dependencies
- Test edge cases and error conditions
- Keep tests independent

## Reporting Issues

### Bug Reports

Include:
- Python version (`python --version`)
- Operating system
- Package version
- Minimal code to reproduce
- Expected vs actual behavior
- Full error messages/stack traces

### Feature Requests

Include:
- Clear description of the feature
- Use cases and examples
- How it fits with existing functionality
- Willingness to implement it yourself

### Questions

- Check existing issues and discussions first
- Search documentation
- Provide context about what you're trying to achieve

## Code Review Process

### For Contributors

- Be patient - maintainers review in their spare time
- Respond to feedback promptly
- Ask questions if feedback is unclear
- Keep PRs focused and reasonably sized

### For Reviewers

- Be respectful and constructive
- Explain the "why" behind requested changes
- Approve when it's good enough, not perfect
- Thank contributors for their time

## Release Process

Maintainers follow this process for releases:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` with changes
3. Create git tag: `git tag -a v0.x.0 -m "Release v0.x.0"`
4. Push tag: `git push origin v0.x.0`
5. Build and publish: `make publish`
6. Create GitHub release with notes

## Getting Help

- 💬 **Discussions** - Ask questions, share ideas
- 🐛 **Issues** - Report bugs, request features
- 📖 **Documentation** - Check README and docs/
- 💼 **Email** - Reach maintainers directly (see pyproject.toml)

## Recognition

Contributors are recognized in:
- `CHANGELOG.md` for their contributions
- GitHub contributors page
- Release notes

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

---

Thank you for contributing! Your efforts make this project better for everyone. 🎉


