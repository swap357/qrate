#!/usr/bin/env python3
"""Setup script to customize the template for a new project.

This script helps you quickly customize the template by replacing
placeholders with your actual project information.
"""

import re
import sys
from pathlib import Path


def get_input(prompt: str, default: str = "") -> str:
    """Get input from user with optional default."""
    if default:
        prompt = f"{prompt} [{default}]"
    value = input(f"{prompt}: ").strip()
    return value or default


def replace_in_file(file_path: Path, replacements: dict[str, str]) -> None:
    """Replace placeholders in a file."""
    try:
        content = file_path.read_text()
        for old, new in replacements.items():
            content = content.replace(old, new)
        file_path.write_text(content)
        print(f"✓ Updated {file_path}")
    except Exception as e:
        print(f"✗ Error updating {file_path}: {e}")


def rename_directory(old_path: Path, new_name: str) -> Path:
    """Rename a directory."""
    new_path = old_path.parent / new_name
    if new_path.exists():
        print(f"⚠ Directory {new_path} already exists, skipping rename")
        return old_path
    old_path.rename(new_path)
    print(f"✓ Renamed {old_path} → {new_path}")
    return new_path


def main() -> None:
    """Run the setup script."""
    print("=" * 60)
    print("Python Package Template Setup")
    print("=" * 60)
    print("\nThis script will customize the template for your project.")
    print("Press Enter to use default values.\n")
    
    # Get project information
    package_name = get_input("Package name (lowercase, underscores)", "my_package")
    description = get_input("Package description", "A Python package")
    author_name = get_input("Author name", "Your Name")
    author_email = get_input("Author email", "your.email@example.com")
    github_username = get_input("GitHub username", "yourusername")
    license_type = get_input("License", "MIT")
    
    # Prepare replacements
    replacements = {
        "PACKAGE_NAME": package_name,
        "A brief description of your package": description,
        "Your Name": author_name,
        "your.email@example.com": author_email,
        "USERNAME": github_username,
        "[fullname]": author_name,
        "[year]": "2024",  # You might want to get current year
    }
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for key, value in replacements.items():
        print(f"{key:30} → {value}")
    print("=" * 60)
    
    confirm = get_input("\nProceed with setup? (yes/no)", "yes").lower()
    if confirm not in ("yes", "y"):
        print("Setup cancelled.")
        return
    
    print("\nSetting up project...")
    
    # Get project root
    root = Path(__file__).parent
    
    # Files to update
    files_to_update = [
        "pyproject.toml",
        "Makefile",
        "README.md",
        "CONTRIBUTING.md",
        "SETUP.md",
        "CHANGELOG.md",
        "LICENSE",
        "src/PACKAGE_NAME/__init__.py",
        "src/PACKAGE_NAME/core.py",
        "src/PACKAGE_NAME/utils.py",
        "tests/test_core.py",
        "tests/test_utils.py",
        "examples/basic_example.py",
        "examples/README.md",
    ]
    
    # Update files
    print("\nUpdating files...")
    for file_path_str in files_to_update:
        file_path = root / file_path_str
        if file_path.exists():
            replace_in_file(file_path, replacements)
    
    # Rename package directory
    print("\nRenaming package directory...")
    old_package_dir = root / "src" / "PACKAGE_NAME"
    if old_package_dir.exists():
        rename_directory(old_package_dir, package_name)
    
    print("\n" + "=" * 60)
    print("Setup complete! 🎉")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review and customize generated files")
    print("2. Create virtual environment: python -m venv .venv")
    print("3. Activate it: source .venv/bin/activate")
    print("4. Install: pip install -e \".[dev]\"")
    print("5. Run tests: make test")
    print("6. Start coding!")
    print("\nHappy developing! 🚀\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        sys.exit(1)


