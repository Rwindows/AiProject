import argparse
import copy
import json
import os
from typing import Dict, List, Union


def load_fhir_json(file_path: str) -> Dict:
    """Load FHIR JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_fhir_json(data: Dict, output_path: str) -> None:
    """Save modified FHIR JSON to a file."""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def find_observation_by_display(data: Dict, display_value: str) -> Union[Dict, None]:
    """Find an observation resource by its display name (case-insensitive and partial matching)."""
    display_value = display_value.lower()
    for entry in data.get('entry', []):
        resource = entry.get('resource', {})
        if resource.get('resourceType') == 'Observation':
            codings = resource.get('code', {}).get('coding', [])
            for coding in codings:
                if coding.get('display', '').lower().find(display_value) != -1:
                    return resource
    return None


def get_test_code_from_observation(observation: Dict) -> str:
    """Extract the test code from an observation."""
    if not observation:
        return "unknown"

    codings = observation.get('code', {}).get('coding', [])
    for coding in codings:
        if coding.get('system') == 'http://terminology.labcorp.com/CodeSystem/lcls-test-code':
            return coding.get('code', 'unknown')
    return "unknown"


def update_observation_value(observation: Dict, new_value: float) -> Dict:
    """Update only the value of an observation, keeping other fields intact."""
    observation = copy.deepcopy(observation)

    # Update only the value
    if 'valueQuantity' in observation:
        observation['valueQuantity']['value'] = new_value

    return observation


def parse_range(range_str):
    """Parse a range string in the format 'min-max'."""
    if '-' in range_str:
        parts = range_str.split('-')
        if len(parts) == 2:
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                raise ValueError(f"Invalid range format: {range_str}. Expected format: min-max (e.g., '232-1245')")

    raise ValueError(f"Invalid range format: {range_str}. Expected format: min-max (e.g., '232-1245')")


def generate_boundary_values(min_value, max_value, steps=7):
    """
    Generate boundary test values for a specified range, focusing on exact boundary values.

    Args:
        min_value: The minimum value of the range
        max_value: The maximum value of the range
        steps: Number of steps to generate (minimum 6 for essential boundary points)

    Returns:
        List of test values
    """
    # Ensure a minimum number of steps for key boundary values
    steps = max(steps, 6)

    # Generate the critical boundary values
    values = [
        min_value - 1,  # Exactly 1 unit below minimum
        min_value,  # Exactly at minimum
        min_value + 1,  # Exactly 1 unit above minimum
    ]

    # Calculate how many in-between values to add
    n_between = steps - 6
    if n_between > 0:
        # Add evenly spaced values between min+1 and max-1
        step_size = (max_value - min_value - 2) / (n_between + 1)
        for i in range(1, n_between + 1):
            values.append(round(min_value + 1 + i * step_size))

    # Add the upper boundary values
    values.extend([
        max_value - 1,  # Exactly 1 unit below maximum
        max_value,  # Exactly at maximum
        max_value + 1  # Exactly 1 unit above maximum
    ])

    return values


def create_test_files(original_data: Dict, display_value: str, test_values: List[float], output_dir: str) -> None:
    """Create multiple test files with different values for the specified vitamin by display name."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find the observation
    observation = find_observation_by_display(original_data, display_value)
    if not observation:
        print(f"Warning: Observation with display name containing '{display_value}' not found")
        return

    # Get the full display name and code for more descriptive filenames
    display_name = "Unknown"
    test_code = get_test_code_from_observation(observation)

    for coding in observation.get('code', {}).get('coding', []):
        if coding.get('system') == 'http://terminology.labcorp.com/CodeSystem/lcls-test-code':
            display_name = coding.get('display', 'Unknown')
            break

    # Create a sanitized name for filenames
    safe_name = display_name.replace(' ', '_').replace(',', '').replace('-', '_').replace('(', '').replace(')', '')

    # Create a variant file for each test value
    for value in test_values:
        variant_data = copy.deepcopy(original_data)

        # Find and update the observation in the variant data
        variant_obs = find_observation_by_display(variant_data, display_value)
        updated_obs = update_observation_value(variant_obs, value)

        # Replace the observation in the variant data
        for entry in variant_data.get('entry', []):
            resource = entry.get('resource', {})
            if (resource.get('resourceType') == 'Observation' and
                    resource.get('id') == observation.get('id')):
                entry['resource'] = updated_obs

        # Save the variant file
        output_file = f"{output_dir}/{safe_name}_value_{value}.json"
        save_fhir_json(variant_data, output_file)
        print(f"Created {output_file} (Code: {test_code})")


def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description='Generate FHIR test files with different vitamin values')

    parser.add_argument('--input', '-i', required=True, help='Input FHIR JSON file path')
    parser.add_argument('--output', '-o', default='test_files', help='Output directory for test files')
    parser.add_argument('--display', '-d', required=True,
                        help='Display value to match (e.g., "Vitamin B12", "Folate", "Vitamin D")')
    parser.add_argument('--range', '-r', required=True, help='Value range for boundary tests (e.g., "232-1245")')
    parser.add_argument('--steps', '-s', type=int, default=7,
                        help='Number of test values to generate (default: 7, minimum 6)')

    args = parser.parse_args()

    print(f"Loading FHIR data from {args.input}")
    try:
        data = load_fhir_json(args.input)
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found")
        return
    except json.JSONDecodeError:
        print(f"Error: '{args.input}' is not a valid JSON file")
        return

    # Parse the value range
    try:
        min_val, max_val = parse_range(args.range)
        print(f"Using value range {min_val} to {max_val}")
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Generate test values
    test_values = generate_boundary_values(min_val, max_val, args.steps)
    print(f"Generating {len(test_values)} test values: {test_values}")

    # Create test files
    create_test_files(data, args.display, test_values, args.output)

    print(f"All test files have been created in {args.output}/")


if __name__ == "__main__":
    # Check if command line arguments are provided
    import sys

    if len(sys.argv) == 1:
        # No arguments provided, show usage instructions
        print("No arguments provided. Use the following format:")
        print(
            "python fhir_boundary_analysis.py --input \"path/to/fhir.json\" --display \"Vitamin Name\" --range \"min-max\" [--steps N] [--output output_dir]")
        print("\nExamples:")
        print(
            "python fhir_boundary_analysis.py --input \"patient_data.json\" --display \"Vitamin B12\" --range \"232-1245\"")
        print(
            "python fhir_boundary_analysis.py --input \"lab_results.json\" --display \"Vitamin D, 25-Hydroxy\" --range \"30-100\" --steps 9")
        print("\nUse --help for more information.")
    else:
        main()
