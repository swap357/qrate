"""Tests for utility functions."""

from PACKAGE_NAME.utils import format_output, helper_function, validate_input


def test_helper_function_with_data():
    """Test helper_function with non-empty list."""
    # Arrange
    data = [1, 2, 3, 4, 5]
    
    # Act
    result = helper_function(data)
    
    # Assert
    assert result == 5


def test_helper_function_empty():
    """Test helper_function with empty list."""
    assert helper_function([]) == 0


def test_validate_input_valid():
    """Test validate_input with valid data."""
    assert validate_input("test") is True
    assert validate_input(123) is True
    assert validate_input([]) is True


def test_validate_input_invalid():
    """Test validate_input with None."""
    assert validate_input(None) is False


def test_format_output():
    """Test format_output with dictionary."""
    # Arrange
    data = {"key1": "value1", "key2": "value2"}
    
    # Act
    result = format_output(data)
    
    # Assert
    assert "key1: value1" in result
    assert "key2: value2" in result


def test_format_output_empty():
    """Test format_output with empty dictionary."""
    assert format_output({}) == ""


def test_format_output_single_item():
    """Test format_output with single item."""
    result = format_output({"key": "value"})
    assert result == "key: value"


