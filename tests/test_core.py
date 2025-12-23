"""Tests for core functionality."""

import pytest

from PACKAGE_NAME.core import ExampleClass, main_function


def test_main_function_positive():
    """Test main_function with positive input."""
    # Arrange
    input_value = 10
    expected = 20
    
    # Act
    result = main_function(input_value)
    
    # Assert
    assert result == expected


def test_main_function_zero():
    """Test main_function with zero input."""
    assert main_function(0) == 0


def test_main_function_negative():
    """Test main_function raises ValueError for negative input."""
    with pytest.raises(ValueError, match="non-negative"):
        main_function(-1)


class TestExampleClass:
    """Tests for ExampleClass."""
    
    def test_initialization(self):
        """Test class initialization."""
        # Arrange & Act
        obj = ExampleClass("test", 10)
        
        # Assert
        assert obj.name == "test"
        assert obj.value == 10
    
    def test_initialization_default_value(self):
        """Test class initialization with default value."""
        obj = ExampleClass("test")
        assert obj.value == 0
    
    def test_increment(self):
        """Test increment method."""
        # Arrange
        obj = ExampleClass("test", 10)
        
        # Act
        result = obj.increment(5)
        
        # Assert
        assert result == 15
        assert obj.value == 15
    
    def test_increment_default(self):
        """Test increment with default amount."""
        obj = ExampleClass("test", 10)
        result = obj.increment()
        assert result == 11
    
    def test_repr(self):
        """Test string representation."""
        obj = ExampleClass("test", 10)
        assert repr(obj) == "ExampleClass(name='test', value=10)"


