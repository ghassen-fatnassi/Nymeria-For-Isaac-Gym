import numpy as np
from collections import OrderedDict
import argparse
import sys

# Define a threshold for array size to decide between full print and summary
# You can adjust this number based on what you consider "large"
ARRAY_SUMMARY_THRESHOLD = 1000 # If total elements > 1000, print summary

def print_recursive_beautiful(data, indent=0):
    """
    Recursively prints the content of nested OrderedDicts, lists,
    and NumPy arrays with indentation, summarizing large arrays.
    """
    indent_str = "  " * indent

    if isinstance(data, OrderedDict):
        if indent == 0:
            print("{OrderedDict:") # Top level
        else:
             print("OrderedDict")
        for i, (key, value) in enumerate(data.items()):
            print(f"{indent_str}  '{key}':", end=" ")
            print_recursive_beautiful(value, indent + 1)
            # Add a comma only if the value printed on the same line
            if not isinstance(value, (OrderedDict, dict, list, tuple, np.ndarray)) and i < len(data) - 1:
                 print(",")
        if indent == 0:
             print("}") # Closing brace for the top-level OrderedDict

    elif isinstance(data, dict):
        print("dict")
        for i, (key, value) in enumerate(data.items()):
            print(f"{indent_str}  '{key}':", end=" ")
            print_recursive_beautiful(value, indent + 1)
            if not isinstance(value, (OrderedDict, dict, list, tuple, np.ndarray)) and i < len(data) - 1:
                 print(",")


    elif isinstance(data, np.ndarray):
        total_elements = data.size
        if total_elements > ARRAY_SUMMARY_THRESHOLD:
            # Print summary for large arrays
            print(f"NumPy array (shape: {data.shape}, dtype: {data.dtype}, size: {total_elements} elements)")
            if data.size > 0: # Avoid min/max on empty arrays
                 try:
                    print(f"{indent_str}    min: {np.min(data)}, max: {np.max(data)}, mean: {np.mean(data)}")
                 except TypeError: # Handle non-numeric dtypes
                     print(f"{indent_str}    contains non-numeric data")

        else:
            # Print full content for small arrays
            print(f"NumPy array (shape: {data.shape}, dtype: {data.dtype})")
            # Use np.array_str for full content of small arrays
            array_str = np.array_str(data, max_rows=sys.maxsize, threshold=sys.maxsize)
            indented_array_str = "\n".join([f"{indent_str}    {line}" for line in array_str.splitlines()])
            print(indented_array_str, end="") # end="" to avoid double newline

    elif isinstance(data, list):
        print("[")
        for i, item in enumerate(data):
            print(f"{indent_str}  ", end="")
            print_recursive_beautiful(item, indent + 1)
            if i < len(data) - 1:
                print(",")
        print(f"\n{indent_str}]", end="") # Newline and indent for closing bracket

    elif isinstance(data, tuple):
        print("(")
        for i, item in enumerate(data):
            print(f"{indent_str}  ", end="")
            print_recursive_beautiful(item, indent + 1)
            if i < len(data) - 1:
                print(",")
        print(f"\n{indent_str})", end="") # Newline and indent for closing parenthesis


    elif isinstance(data, str):
        # Handle multi-line strings
        if "\n" in data:
            print("str")
            print(f'{indent_str}    """{data}"""', end="")
        else:
             print(f"str")
             print(f"'{data}'", end="") # Print string value

    elif isinstance(data, bool):
        print("bool")
        print(f"value: {data}", end="") # Print boolean value

    # Handle numeric types (int, float), None, etc.
    else:
        print(type(data).__name__)
        print(f"value: {data}", end="")

    # Add a newline after printing a value, unless it's a nested structure
    if not isinstance(data, (OrderedDict, dict, list, tuple, np.ndarray)):
         print("")


def print_proto_motion_npy_beautiful(npy_file_path):
    """
    Prints the content of a .npy file that is expected to contain
    ProtoMotion-formatted data as an OrderedDict, recursively and ordered,
    with summary for large NumPy arrays.

    Args:
        npy_file_path (str): The path to the .npy file.
    """
    try:
        # Use .item() to get the scalar object (the OrderedDict) from the 0-d array
        data = np.load(npy_file_path, allow_pickle=True).item()
        print_recursive_beautiful(data)
        print("\n") # Add a final newline for cleanliness

    except FileNotFoundError:
        print(f"Error: File not found at {npy_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Print content of a ProtoMotion .npy file beautifully (summarizing large arrays).')
    parser.add_argument('npy_file', type=str, help='Path to the ProtoMotion .npy file')
    # Optional argument to print full arrays if needed (can be added later)
    # parser.add_argument('--full', action='store_true', help='Print full content of large arrays (can be very verbose)')


    args = parser.parse_args()

    # if args.full:
    #     ARRAY_SUMMARY_THRESHOLD = -1 # Set threshold low to print all

    print_proto_motion_npy_beautiful(args.npy_file)