# Examples

This directory contains example scripts demonstrating how to use PACKAGE_NAME.

## Running Examples

Make sure you have the package installed:

```bash
# From project root
pip install -e ".[dev]"

# Run examples
python examples/basic_example.py
```

## Examples

### 1. Basic Example (`basic_example.py`)

Demonstrates basic usage of the main functionality.

```bash
python examples/basic_example.py
```

**Concepts:** Basic API usage, simple operations

---

## Creating Your Own Examples

1. Create a new Python file in this directory
2. Add a clear docstring explaining what it demonstrates
3. Include comments to explain key concepts
4. Use the `output/` directory for any generated files
5. Update this README with a description

Example template:

```python
"""Example: Description of what this demonstrates.

This example shows how to use feature X with Y approach.
"""

from PACKAGE_NAME import main_function


def main():
    """Run the example."""
    result = main_function(10)
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
```

## Output

Generated files are saved to `examples/output/` and are ignored by git.

## Tips

- Keep examples focused on one concept
- Add comments explaining non-obvious code
- Make examples runnable standalone
- Use realistic but simple use cases
- Show best practices


