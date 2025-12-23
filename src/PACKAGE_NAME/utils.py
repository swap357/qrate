"""Utility functions for PACKAGE_NAME.

This module contains helper functions and utilities.
"""

from typing import Any


def helper_function(data: list[Any]) -> int:
    """A helper function demonstrating utilities.
    
    Args:
        data: A list of items to process
        
    Returns:
        The length of the data
        
    Examples:
        >>> helper_function([1, 2, 3])
        3
        >>> helper_function([])
        0
    """
    return len(data)


def validate_input(value: Any) -> bool:
    """Validate input data.
    
    This is a placeholder validation function. Replace with your
    actual validation logic.
    
    Args:
        value: The value to validate
        
    Returns:
        True if valid, False otherwise
        
    Examples:
        >>> validate_input("test")
        True
        >>> validate_input(None)
        False
    """
    return value is not None


def format_output(data: dict[str, Any]) -> str:
    """Format data for output.
    
    Args:
        data: Dictionary to format
        
    Returns:
        Formatted string representation
        
    Examples:
        >>> format_output({"key": "value"})
        'key: value'
    """
    return "\n".join(f"{k}: {v}" for k, v in data.items())


