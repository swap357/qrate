"""Core functionality for PACKAGE_NAME.

This module contains the main functionality of the package.
"""


def main_function(value: int) -> int:
    """Demonstrate a main function with proper documentation.
    
    This is a placeholder for your actual functionality. Replace with
    your implementation.
    
    Args:
        value: An input value to process
        
    Returns:
        The processed result
        
    Raises:
        ValueError: If value is negative
        
    Examples:
        >>> main_function(10)
        20
        >>> main_function(0)
        0
    """
    if value < 0:
        raise ValueError("Value must be non-negative")
    
    return value * 2


class ExampleClass:
    """Example class demonstrating structure and documentation.
    
    This class shows how to properly document and structure your code.
    Replace with your actual implementation.
    
    Attributes:
        name: The name of the instance
        value: An integer value
    """
    
    def __init__(self, name: str, value: int = 0) -> None:
        """Initialize the ExampleClass.
        
        Args:
            name: The name to assign
            value: Initial value (default: 0)
        """
        self.name = name
        self.value = value
    
    def increment(self, amount: int = 1) -> int:
        """Increment the internal value.
        
        Args:
            amount: Amount to increment by (default: 1)
            
        Returns:
            The new value after incrementing
            
        Examples:
            >>> obj = ExampleClass("test", 10)
            >>> obj.increment(5)
            15
            >>> obj.value
            15
        """
        self.value += amount
        return self.value
    
    def __repr__(self) -> str:
        """Return string representation."""
        return f"ExampleClass(name={self.name!r}, value={self.value})"


