import argparse
import base64
import copy
import json
import os
import sys
from typing import Dict, List, Union

# Check for pandas before importing
try:
    import pandas as pd

    # Check for openpyxl dependency
    try:
        # Try to import openpyxl
        import openpyxl
    except ImportError:
        print("Error: openpyxl is not installed. This is required for Excel file handling.")
        print("Please install it using:")
        print("pip install openpyxl")
        print("\nOr install pandas with Excel support:")
        print("pip install pandas openpyxl")
        sys.exit(1)
except ImportError:
    print("Error: pandas is not installed. Please install it using:")
    print("pip install pandas openpyxl requests")  # Include openpyxl and requests by default
    print("\nIf pip is not found, you may need to install it first:")
    print("- For Windows: python -m ensurepip --upgrade")
    print("- For Linux/Mac: python3 -m ensurepip --upgrade")
    print("Or visit https://pip.pypa.io/en/stable/installation/ for instructions")
    sys.exit(1)

# Check for requests before importing
try:
    import requests
except ImportError:
    print("Error: requests is not installed. Please install it using:")
    print("pip install requests")
    print("\nIf pip is not found, you may need to install it first:")
    print("- For Windows: python -m ensurepip --upgrade")
    print("- For Linux/Mac: python3 -m ensurepip --upgrade")
    sys.exit(1)


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
                min_val = float(parts[0].strip())
                max_val = float(parts[1].strip())

                # Convert to int if they're whole numbers to avoid .0 in output
                if min_val.is_integer():
                    min_val = int(min_val)
                if max_val.is_integer():
                    max_val = int(max_val)

                return min_val, max_val
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
            value = min_value + 1 + i * step_size
            # Convert to int if it's a whole number
            if value.is_integer():
                values.append(int(value))
            else:
                values.append(round(value, 2))  # Round to 2 decimal places for readability

    # Add the upper boundary values
    values.extend([
        max_value - 1,  # Exactly 1 unit below maximum
        max_value,  # Exactly at maximum
        max_value + 1  # Exactly 1 unit above maximum
    ])

    # Convert integer values to int type to avoid .0 suffix
    final_values = []
    for value in values:
        if isinstance(value, float) and value.is_integer():
            final_values.append(int(value))
        else:
            final_values.append(value)

    return final_values


def fetch_library_data(library_id, base_url="https://cdsa-cds-api-execution-dev.ald6h.cws.labcorp.com/Library/"):
    """
    Fetch library data from the specified URL, extract and decode the ELM JSON content.

    Args:
        library_id: The library ID to fetch
        base_url: Base URL for the library API

    Returns:
        The decoded ELM JSON if successful, None otherwise
    """
    print(f"Getting the base_url {base_url}...")

    url = f"{base_url}{library_id}"

    try:
        print(f"Fetching library data from {url}...")
        response = requests.get(url)

        # Check if request was successful
        if response.status_code != 200:
            print(f"Error: Failed to fetch data. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return None

        # Parse the response JSON
        try:
            data = response.json()
        except json.JSONDecodeError:
            print(f"Error: Response is not valid JSON: {response.text[:200]}...")
            return None

        # Find the contentType: "application/elm+json" entry
        elm_content = None
        if 'content' in data:
            for content_item in data.get('content', []):
                if content_item.get('contentType') == 'application/elm+json':
                    elm_content = content_item.get('data')
                    break

        if not elm_content:
            print("Error: No 'application/elm+json' content found in the response")
            return None

        # Decode base64 content
        try:
            decoded_bytes = base64.b64decode(elm_content)
            elm_json = json.loads(decoded_bytes.decode('utf-8'))

            # Save the decoded ELM JSON to a file for reference
            elm_filename = f"library_elm_{library_id}.json"
            with open(elm_filename, 'w') as f:
                json.dump(elm_json, f, indent=2)

            print(f"Decoded ELM JSON saved to {elm_filename}")
            return elm_json

        except Exception as e:
            print(f"Error decoding base64 content: {e}")
            return None

    except Exception as e:
        print(f"Error fetching library data: {e}")
        return None


def create_test_files(original_data: Dict, display_value: str, test_values: List[float],
                      output_dir: str, library_elm_json=None, interp_code=None, interp_display=None) -> None:
    """
    Create multiple test files with different values for the specified vitamin by display name.

    Args:
        original_data: The original FHIR bundle data
        display_value: Display value to match in observations
        test_values: List of test values to generate
        output_dir: Output directory for generated files
        library_elm_json: Optional ELM JSON to add to each test file
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

        # Ensure value is appropriately formatted
        numeric_value = value
        if isinstance(numeric_value, float) and numeric_value.is_integer():
            numeric_value = int(numeric_value)

        # Update with custom interpretation if provided
        updated_obs = update_observation_value_and_interpretation(
            variant_obs,
            numeric_value,
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

        # Add ELM JSON if provided
        if library_elm_json:
            variant_data['elmJson'] = library_elm_json

        # Save the variant file with interpretation in the filename
        output_file = f"{output_dir}/{safe_name}_value_{value}_interp_{interpretation_code}.json"
        save_fhir_json(variant_data, output_file)
        print(
            f"Created {output_file} (Code: {test_code}, Interpretation: {interpretation_code}, Display: {interpretation_display})")


def process_excel_input(excel_path, fhir_input, output_dir,
                        base_url="https://cdsa-cds-api-execution-dev.ald6h.cws.labcorp.com/Library/"):
    """
    Process test scenarios defined in an Excel file.

    The Excel file should have columns:
    - display: Display value to match (e.g., "Vitamin B12", "Folate")
    - range: Value range for boundary tests (e.g., "232-1245")
    - library_id: Optional library ID to fetch ELM JSON from
    - steps: Number of test values to generate (optional)
    - interp_code: Override interpretation code (optional)
    - interp_display: Override interpretation display text (optional)
    """
    try:
        # Try to load the Excel file with explicit engine specification
        try:
            df = pd.read_excel(excel_path, engine='openpyxl')
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            print("Make sure you have openpyxl installed: pip install openpyxl")
            return

        # Check for required columns
        required_columns = ['display', 'range']
        for col in required_columns:
            if col not in df.columns:
                print(f"Error: Required column '{col}' not found in Excel file")
                return

        # Load the FHIR data once
        print(f"Loading FHIR data from {fhir_input}")
        try:
            data = load_fhir_json(fhir_input)
        except FileNotFoundError:
            print(f"Error: Input file '{fhir_input}' not found")
            return
        except json.JSONDecodeError:
            print(f"Error: '{fhir_input}' is not a valid JSON file")
            return

        # Process each row in the Excel file
        successful_scenarios = 0
        total_scenarios = len(df)

        print(f"Found {total_scenarios} test scenarios in Excel file")

        for index, row in df.iterrows():
            try:
                display = row['display']
                # Format range_str properly to remove trailing .0 from integer values
                range_value = row['range']
                if isinstance(range_value, float) and range_value.is_integer():
                    range_str = str(int(range_value))
                else:
                    range_str = str(range_value)

                # Get optional parameters with defaults, handling float values correctly
                steps = 7
                if 'steps' in df.columns and not pd.isna(row['steps']):
                    steps_value = row['steps']
                    if isinstance(steps_value, float) and steps_value.is_integer():
                        steps = int(steps_value)
                    else:
                        steps = int(steps_value)

                interp_code = row['interp_code'] if 'interp_code' in df.columns and not pd.isna(
                    row['interp_code']) else None
                interp_display = row['interp_display'] if 'interp_display' in df.columns and not pd.isna(
                    row['interp_display']) else None

                # Check if we have a library ID to fetch
                library_id = None
                if 'library_id' in df.columns and not pd.isna(row['library_id']):
                    library_id_value = row['library_id']
                    # Format library_id properly to remove trailing .0 from integer values
                    if isinstance(library_id_value, float) and library_id_value.is_integer():
                        library_id = str(int(library_id_value))
                    else:
                        library_id = str(library_id_value)
                library_elm_json = None

                print(f"\nProcessing scenario {index + 1}/{total_scenarios}: '{display}' with range {range_str}")

                # If we have a library ID, fetch the ELM JSON
                if library_id:
                    print(f"Fetching library data for ID: {library_id}")
                    library_elm_json = fetch_library_data(library_id, base_url)
                    if not library_elm_json:
                        print(f"Warning: Failed to fetch library data for ID {library_id}, continuing without it")

                # Parse the value range
                try:
                    min_val, max_val = parse_range(range_str)
                    print(f"Using value range {min_val} to {max_val}")
                except ValueError as e:
                    print(f"Error in row {index + 1}: {e}")
                    continue

                # Generate test values
                test_values = generate_boundary_values(min_val, max_val, steps)
                print(f"Generating {len(test_values)} test values: {test_values}")

                # Create test files
                create_test_files(
                    data,
                    display,
                    test_values,
                    output_dir,
                    library_elm_json,
                    interp_code,
                    interp_display
                )

                successful_scenarios += 1

            except Exception as e:
                print(f"Error processing row {index + 1}: {e}")

        print(f"\nCompleted {successful_scenarios}/{total_scenarios} test scenarios")
        print(f"All test files have been created in {output_dir}/")

    except Exception as e:
        print(f"Error processing Excel file: {e}")
        print("Make sure you have all required dependencies installed:")
        print("pip install pandas openpyxl requests")


def create_example_excel():
    """Create an example Excel file with test scenarios"""
    try:
        # Try to import openpyxl directly to check if it's installed
        try:
            import openpyxl
        except ImportError:
            print("Error: openpyxl is not installed. This is required for Excel file handling.")
            print("Please install it using:")
            print("pip install openpyxl")
            return

        # Example data
        data = {
            'display': ['Vitamin B12', 'Folate (Folic Acid)', 'Vitamin D, 25-Hydroxy'],
            'range': ['232-1245', '3-20', '30-100'],
            'steps': [7, 7, 7],
            'library_id': ['24833304', None, '24833304'],  # Example library IDs
            'interp_code': [None, None, 'DEFICIENT'],
            'interp_display': [None, None, 'Deficient']
        }

        # Create DataFrame
        df = pd.DataFrame(data)

        # Save to Excel with explicit engine specification
        output_file = 'test_scenarios_example.xlsx'
        df.to_excel(output_file, index=False, engine='openpyxl')

        print(f"Created example Excel file: {output_file}")
        print("Use this file with: python TestScenario.py --excel test_scenarios_example.xlsx --input file.json")

    except Exception as e:
        print(f"Error creating example Excel: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install pandas openpyxl")


def main():
    """Main function with command line arguments for both Excel and direct arguments."""
    parser = argparse.ArgumentParser(description='Generate FHIR test files with different vitamin values')

    # Create a mutually exclusive group for Excel input vs direct arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--excel', '-e', help='Excel file containing test scenarios')
    input_group.add_argument('--display', '-d', help='Display value to match (e.g., "Vitamin B12", "Folate")')

    # Common arguments
    parser.add_argument('--input', '-i', required=True, help='Input FHIR JSON file path')
    parser.add_argument('--output', '-o', default='test_files', help='Output directory for test files')
    parser.add_argument('--base-url', default="https://cdsa-cds-api-execution-dev.ald6h.cws.labcorp.com/Library/",
                        help='Base URL for library API (default: %(default)s)')

    # Direct argument mode arguments
    parser.add_argument('--range', '-r', help='Value range for boundary tests (e.g., "232-1245")')
    parser.add_argument('--steps', '-s', type=int, default=7,
                        help='Number of test values to generate (default: 7, minimum 6)')
    parser.add_argument('--library-id', help='Library ID to fetch ELM JSON from')
    parser.add_argument('--interp-code', help='Override interpretation code (e.g., "L", "DEFICIENT", "1")')
    parser.add_argument('--interp-display', help='Override interpretation display text (e.g., "Low", "Deficient")')

    args = parser.parse_args()

    # If Excel mode is selected
    if args.excel:
        process_excel_input(args.excel, args.input, args.output, args.base_url)
    else:
        # Original direct argument mode
        if not args.range:
            print("Error: --range is required when not using Excel input")
            return

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

        # Fetch library data if library ID is provided
        library_elm_json = None
        if args.library_id:
            print(f"Fetching library data for ID: {args.library_id}")
            library_elm_json = fetch_library_data(args.library_id, args.base_url)
            if not library_elm_json:
                print(f"Warning: Failed to fetch library data for ID {args.library_id}, continuing without it")

        # Create test files with optional custom interpretation and library data
        create_test_files(
            data,
            args.display,
            test_values,
            args.output,
            library_elm_json,
            args.interp_code,
            args.interp_display
        )

        # Summary message
        print(f"All test files have been created in {args.output}/")
        if args.library_id:
            print(f"Added library data from ID: {args.library_id}")
        if args.interp_code or args.interp_display:
            interp_msg = "Used custom interpretation"
            if args.interp_code:
                interp_msg += f", code: {args.interp_code}"
            if args.interp_display:
                interp_msg += f", display: {args.interp_display}"
            print(interp_msg)


if __name__ == "__main__":
    # Check if command line arguments are provided
    if len(sys.argv) == 1:
        # No arguments provided, show usage instructions
        print("No arguments provided. Use one of the following formats:")
        print("\nUsing Excel input:")
        print(
            "python TestScenario.py --excel \"path/to/scenarios.xlsx\" --input \"path/to/fhir.json\" [--output output_dir] [--base-url URL]")
        print("\nUsing command line arguments:")
        print(
            "python TestScenario.py --input \"path/to/fhir.json\" --display \"Vitamin Name\" --range \"min-max\" [--steps N] [--output output_dir] [--library-id ID] [--interp-code CODE] [--interp-display TEXT]")
        print("\nExamples:")
        print("python TestScenario.py --excel \"test_scenarios.xlsx\" --input \"patient_data.json\"")
        print(
            "python TestScenario.py --input \"patient_data.json\" --display \"Vitamin B12\" --range \"232-1245\" --library-id 24833304")
        print("\nTo create an example Excel file:")
        print("python TestScenario.py --create-example-excel")
        print("\nUse --help for more information.")
        print("\nRequired packages:")
        print("- pandas: for data handling")
        print("- openpyxl: for Excel file support")
        print("- requests: for API calls")
        print("Install with: pip install pandas openpyxl requests")
    elif len(sys.argv) == 2 and sys.argv[1] == "--create-example-excel":
        create_example_excel()
    else:
        main()
