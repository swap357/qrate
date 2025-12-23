"""Basic example demonstrating core functionality.

This example shows how to use the main features of PACKAGE_NAME.
"""

from PACKAGE_NAME import main_function
from PACKAGE_NAME.core import ExampleClass
from PACKAGE_NAME.utils import helper_function


def main() -> None:
    """Run the basic example."""
    print("=" * 60)
    print("PACKAGE_NAME - Basic Example")
    print("=" * 60)
    
    # Example 1: Using main_function
    print("\n1. Using main_function:")
    value = 10
    result = main_function(value)
    print(f"   Input: {value}")
    print(f"   Output: {result}")
    
    # Example 2: Using ExampleClass
    print("\n2. Using ExampleClass:")
    obj = ExampleClass("demo", 5)
    print(f"   Created: {obj}")
    obj.increment(3)
    print(f"   After increment: {obj}")
    
    # Example 3: Using helper_function
    print("\n3. Using helper_function:")
    data = [1, 2, 3, 4, 5]
    length = helper_function(data)
    print(f"   Data: {data}")
    print(f"   Length: {length}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()


