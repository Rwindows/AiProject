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


def determine_interpretation(value: float, reference_range: Dict) -> str:
    """
    Determine the appropriate interpretation based on the value and reference range.
    Returns "Low", "Normal", or "High".
    """
    # Extract range values
    low_value = None
    high_value = None

    if 'low' in reference_range:
        low_value = reference_range['low'].get('value')
    if 'high' in reference_range:
        high_value = reference_range['high'].get('value')

    # If text is like ">3.0 ng/mL", parse it
    if not low_value and not high_value and 'text' in reference_range:
        text = reference_range['text']
        if text.startswith('>'):
            try:
                # Extract the number after '>' and before the unit
                min_value_str = text.split('>')[1].split(' ')[0]
                low_value = float(min_value_str)
                # No upper bound
            except (IndexError, ValueError):
                pass

    # Determine interpretation
    if low_value is not None and value < low_value:
        return "Low"
    elif high_value is not None and value > high_value:
        return "High"
    else:
        return "Normal"


def update_observation_value_and_interpretation(observation: Dict, new_value: float,
                                                interp_code=None, interp_display=None) -> Dict:
    """
    Update the value and interpretation of an observation based on the new value.
    If interp_code and interp_display are provided, they override the automatically determined interpretation.
    """
    observation = copy.deepcopy(observation)

    # Update the value
    if 'valueQuantity' in observation:
        observation['valueQuantity']['value'] = new_value

    # If we need to update the interpretation
    if interp_code is not None or interp_display is not None:
        # First, determine the natural interpretation
        natural_interp = None
        if 'referenceRange' in observation and observation['referenceRange']:
            reference_range = observation['referenceRange'][0]
            natural_interp = determine_interpretation(new_value, reference_range)

        # Create the new interpretation coding object
        new_interpretation = {
            "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
        }

        # Set the code (either custom or based on natural interpretation)
        if interp_code is not None:
            new_interpretation["code"] = interp_code
        else:
            # Default codes based on natural interpretation
            default_codes = {"Normal": "N", "Low": "L", "High": "H"}
            new_interpretation["code"] = default_codes.get(natural_interp, "N")

        # Set the display (either custom or based on natural interpretation)
        if interp_display is not None:
            new_interpretation["display"] = interp_display
        else:
            new_interpretation["display"] = natural_interp

        # Update interpretation in the observation
        if 'interpretation' in observation:
            if observation['interpretation'] and len(observation['interpretation']) > 0:
                if 'coding' in observation['interpretation'][0]:
                    observation['interpretation'][0]['coding'][0] = new_interpretation
                else:
                    observation['interpretation'][0]['coding'] = [new_interpretation]
            else:
                observation['interpretation'] = [{"coding": [new_interpretation]}]
        else:
            observation['interpretation'] = [{"coding": [new_interpretation]}]

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


def create_test_files(original_data: Dict, display_value: str, test_values: List[float],
                      output_dir: str, interp_code=None, interp_display=None) -> None:
    """
    Create multiple test files with different values for the specified vitamin by display name.

    Args:
        original_data: The original FHIR bundle data
        display_value: Display value to match in observations
        test_values: List of test values to generate
        output_dir: Output directory for generated files
        interp_code: Optional override for interpretation code
        interp_display: Optional override for interpretation display text
    """
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

        # Update with custom interpretation if provided
        updated_obs = update_observation_value_and_interpretation(
            variant_obs,
            value,
            interp_code,
            interp_display
        )

        # Replace the observation in the variant data
        for entry in variant_data.get('entry', []):
            resource = entry.get('resource', {})
            if (resource.get('resourceType') == 'Observation' and
                    resource.get('id') == observation.get('id')):
                entry['resource'] = updated_obs

        # Get interpretation for filename
        interpretation_code = "unknown"
        interpretation_display = "unknown"
        if (updated_obs.get('interpretation') and
                updated_obs['interpretation'][0].get('coding') and
                updated_obs['interpretation'][0]['coding'][0]):
            coding = updated_obs['interpretation'][0]['coding'][0]
            interpretation_code = coding.get('code', 'unknown')
            interpretation_display = coding.get('display', 'unknown')

        # Save the variant file with interpretation in the filename
        output_file = f"{output_dir}/{safe_name}_value_{value}_interp_{interpretation_code}.json"
        save_fhir_json(variant_data, output_file)
        print(
            f"Created {output_file} (Code: {test_code}, Interpretation: {interpretation_code}, Display: {interpretation_display})")


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
    parser.add_argument('--interp-code', help='Override interpretation code (e.g., "L", "DEFICIENT", "1")')
    parser.add_argument('--interp-display', help='Override interpretation display text (e.g., "Low", "Deficient")')

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

    # Create test files with optional custom interpretation
    create_test_files(
        data,
        args.display,
        test_values,
        args.output,
        args.interp_code,
        args.interp_display
    )

    # Summary message
    print(f"All test files have been created in {args.output}/")
    if args.interp_code or args.interp_display:
        interp_msg = "Used custom interpretation"
        if args.interp_code:
            interp_msg += f", code: {args.interp_code}"
        if args.interp_display:
            interp_msg += f", display: {args.interp_display}"
        print(interp_msg)


if __name__ == "__main__":
    # Check if command line arguments are provided
    import sys

    if len(sys.argv) == 1:
        # No arguments provided, show usage instructions
        print("No arguments provided. Use the following format:")
        print(
            "python TestScenario.py --input \"path/to/fhir.json\" --display \"Vitamin Name\" --range \"min-max\" [--steps N] [--output output_dir] [--interp-code CODE] [--interp-display TEXT]")
        print("\nExamples:")
        print(
            "python TestScenario.py --input \"patient_data.json\" --display \"Vitamin B12\" --range \"232-1245\"")
        print(
            "python TestScenario.py --input \"lab_results.json\" --display \"Vitamin D, 25-Hydroxy\" --range \"30-100\" --interp-code \"DEFICIENT\"")
        print("\nUse --help for more information.")
    else:
        main()
