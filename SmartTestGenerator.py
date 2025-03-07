import argparse
import copy
import json
import logging
import os
import time
from typing import Dict

import pandas as pd
import requests
import urllib3

# Suppress insecure request warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ClaudeTestGenerator:
    """Test scenario generator powered by Claude API."""

    def __init__(self, api_key, model="claude-3-sonnet-20240229"):
        """Initialize the Claude test generator."""
        self.api_key = api_key
        self.model = model
        self.claude_api_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

    def analyze_observation(self, observation: Dict) -> Dict:
        """Analyze observation using Claude API and suggest optimal test values."""
        # Extract observation details for analysis
        observation_info = self._extract_observation_info(observation)

        # Create prompt for Claude
        system_prompt = "You are a medical test expert specializing in laboratory result analysis. Your task is to analyze laboratory test observations and recommend appropriate test values for validation."

        user_prompt = f"""Analyze this laboratory observation:
        
        {json.dumps(observation_info, indent=2)}
        
        Based on this observation:
        1. What are the 5-7 most medically significant boundary test values to generate?
        2. What edge cases should be tested based on clinical relevance?
        3. Are there any specific interpretation codes/displays that should be tested?
        
        Format your response as JSON with these keys:
        - test_values: [array of numeric values]
        - edge_cases: [array of objects with "value" and "reason" keys]
        - custom_interpretations: [array of objects with "value", "code", "display" keys]
        """

        # Make API request to Claude
        try:
            payload = {
                "model": self.model,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 1000
            }

            # Add exponential backoff for retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        self.claude_api_url,
                        headers=self.headers,
                        json=payload,
                        timeout=30,  # Set a reasonable timeout
                        verify=False  # Disable SSL verification
                    )
                    response.raise_for_status()  # Raise exception for HTTP errors
                    break
                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.info(f"Retrying Claude API request in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        raise e

            # Parse Claude's response
            response_data = response.json()
            content = response_data.get("content", [])

            # Extract text from content blocks
            response_text = ""
            for block in content:
                if block.get("type") == "text":
                    response_text += block.get("text", "")

            # Try to find and parse JSON in the response
            try:
                # Find JSON block
                if '{' in response_text and '}' in response_text:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    json_str = response_text[json_start:json_end]
                    suggestions = json.loads(json_str)
                else:
                    logger.warning("No JSON found in Claude's response, using default values")
                    suggestions = self._create_default_suggestions(observation_info)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON in Claude's response, using default values")
                suggestions = self._create_default_suggestions(observation_info)

            return suggestions

        except Exception as e:
            logger.error(f"Error calling Claude API: {str(e)}")
            # Return default values if API call fails
            return self._create_default_suggestions(observation_info)

    def _create_default_suggestions(self, observation_info: Dict) -> Dict:
        """Create default suggestions if API fails."""
        logger.info("Creating default test value suggestions")

        # Get reference range information
        low_value = observation_info.get("reference_range", {}).get("low")
        high_value = observation_info.get("reference_range", {}).get("high")
        current_value = observation_info.get("current_value")

        suggestions = {
            "test_values": [],
            "edge_cases": [],
            "custom_interpretations": []
        }

        # Generate test values based on reference range
        if low_value is not None and high_value is not None:
            # Both low and high boundaries
            range_diff = high_value - low_value

            suggestions["test_values"] = [
                low_value - 1,  # Just below minimum
                low_value,  # At minimum
                low_value + range_diff * 0.1,  # Just above minimum
                low_value + range_diff * 0.5,  # Middle of range
                high_value - 1,  # Just below maximum
                high_value,  # At maximum
                high_value + 1  # Just above maximum
            ]

            # Add edge cases
            suggestions["edge_cases"] = [
                {"value": low_value - range_diff * 0.5, "reason": "Significantly below range"},
                {"value": high_value + range_diff * 0.5, "reason": "Significantly above range"}
            ]

        elif low_value is not None:
            # Only low boundary (e.g., ">3.0 ng/mL")
            suggestions["test_values"] = [
                low_value * 0.9,  # Just below minimum
                low_value,  # At minimum
                low_value * 1.1,  # Just above minimum
                low_value * 2  # Well above minimum
            ]

            # Add edge cases
            suggestions["edge_cases"] = [
                {"value": low_value * 0.5, "reason": "Significantly below minimum"}
            ]

        elif high_value is not None:
            # Only high boundary
            suggestions["test_values"] = [
                high_value * 0.5,  # Well below maximum
                high_value * 0.9,  # Just below maximum
                high_value,  # At maximum
                high_value * 1.1  # Just above maximum
            ]

            # Add edge cases
            suggestions["edge_cases"] = [
                {"value": high_value * 2, "reason": "Significantly above maximum"}
            ]

        elif current_value is not None:
            # No reference range, use current value
            suggestions["test_values"] = [
                current_value * 0.5,  # Half of current
                current_value * 0.9,  # Just below current
                current_value,  # Current value
                current_value * 1.1,  # Just above current
                current_value * 2  # Double current
            ]

        # Clean up values (round to 2 decimal places)
        suggestions["test_values"] = [round(v, 2) if isinstance(v, float) else v for v in suggestions["test_values"]]
        for case in suggestions["edge_cases"]:
            if isinstance(case["value"], float):
                case["value"] = round(case["value"], 2)

        return suggestions

    def _extract_observation_info(self, observation: Dict) -> Dict:
        """Extract relevant information from an observation for analysis."""
        info = {
            "display": "",
            "test_code": "",
            "current_value": None,
            "unit": "",
            "interpretation_code": "",
            "interpretation_display": "",
            "reference_range": {}
        }

        # Extract code/display
        for coding in observation.get('code', {}).get('coding', []):
            if coding.get('system') == 'http://terminology.labcorp.com/CodeSystem/lcls-test-code':
                info["test_code"] = coding.get('code', '')
                info["display"] = coding.get('display', '')

        # Get value
        if 'valueQuantity' in observation:
            info["current_value"] = observation['valueQuantity'].get('value')
            info["unit"] = observation['valueQuantity'].get('unit', '')

        # Get interpretation
        if observation.get('interpretation') and observation['interpretation'][0].get('coding'):
            coding = observation['interpretation'][0]['coding'][0]
            info["interpretation_code"] = coding.get('code', '')
            info["interpretation_display"] = coding.get('display', '')

        # Get reference range
        if observation.get('referenceRange'):
            ref_range = observation['referenceRange'][0]
            if 'low' in ref_range:
                info["reference_range"]["low"] = ref_range['low'].get('value')
            if 'high' in ref_range:
                info["reference_range"]["high"] = ref_range['high'].get('value')
            if 'text' in ref_range:
                info["reference_range"]["text"] = ref_range['text']

                # Parse text representation for ranges like ">3.0 ng/mL"
                if info["reference_range"]["text"] and not info["reference_range"].get("low") and not info[
                    "reference_range"].get("high"):
                    text = info["reference_range"]["text"]
                    if text.startswith('>'):
                        try:
                            # Extract the number after '>' and before the unit
                            min_value_str = text.split('>')[1].split(' ')[0]
                            info["reference_range"]["low"] = float(min_value_str)
                        except (IndexError, ValueError):
                            pass

        return info

    def generate_test_scenarios(self, fhir_input: str, output_dir: str) -> bool:
        """Generate test scenarios with Claude-optimized test values."""
        logger.info(f"Generating test scenarios from {fhir_input}")

        # Verify input file exists
        if not os.path.exists(fhir_input):
            logger.error(f"Input file not found: {fhir_input}")
            return False

        try:
            # Load the FHIR data
            with open(fhir_input, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in {fhir_input}")
            return False
        except Exception as e:
            logger.error(f"Error reading input file: {str(e)}")
            return False

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Create a summary for the Excel report
        summary_data = []

        # Check if there are any observations in the data
        observation_count = 0
        for entry in data.get('entry', []):
            resource = entry.get('resource', {})
            if resource.get('resourceType') == 'Observation':
                observation_count += 1

        if observation_count == 0:
            logger.error("No observations found in the input file")
            return False

        logger.info(f"Found {observation_count} observations in the input file")

        # Process each observation
        successful_tests = 0
        for entry in data.get('entry', []):
            resource = entry.get('resource', {})
            if resource.get('resourceType') == 'Observation':
                observation = resource

                # Get observation details for filename/logging
                display_name = "Unknown"
                test_code = "unknown"

                for coding in observation.get('code', {}).get('coding', []):
                    if coding.get('system') == 'http://terminology.labcorp.com/CodeSystem/lcls-test-code':
                        display_name = coding.get('display', 'Unknown')
                        test_code = coding.get('code', 'unknown')
                        break

                # Create a sanitized name for filenames
                safe_name = display_name.replace(' ', '_').replace(',', '').replace('-', '_').replace('(', '').replace(
                    ')', '')
                logger.info(f"Processing observation: {display_name} (Code: {test_code})")

                try:
                    # Get Claude analysis and suggestions
                    suggestions = self.analyze_observation(observation)

                    # Create test files for suggested values
                    num_files = self._create_test_scenarios_from_suggestions(
                        data,
                        observation,
                        suggestions,
                        output_dir,
                        safe_name,
                        test_code,
                        display_name
                    )

                    if num_files > 0:
                        successful_tests += 1

                    # Add to summary
                    summary_row = {
                        "Test Name": display_name,
                        "Test Code": test_code,
                        "Original Value": observation.get('valueQuantity', {}).get('value', 'N/A'),
                        "Unit": observation.get('valueQuantity', {}).get('unit', 'N/A'),
                        "Reference Range": observation.get('referenceRange', [{}])[0].get('text', 'N/A'),
                        "AI Suggested Values": ", ".join(str(v) for v in suggestions.get('test_values', [])),
                        "Edge Cases": ", ".join(f"{case['value']} ({case['reason']})"
                                                for case in suggestions.get('edge_cases', []))
                    }
                    summary_data.append(summary_row)
                except Exception as e:
                    logger.error(f"Error processing observation {display_name}: {str(e)}")

        # Create summary Excel if we have data
        if summary_data:
            try:
                df = pd.DataFrame(summary_data)
                df.to_excel(f"{output_dir}/test_scenarios_summary.xlsx", index=False)
                logger.info(f"Created test scenarios summary: {output_dir}/test_scenarios_summary.xlsx")
            except Exception as e:
                logger.error(f"Error creating summary Excel: {str(e)}")

        logger.info(f"Successfully generated test files for {successful_tests} out of {observation_count} observations")
        return successful_tests > 0

    def _create_test_scenarios_from_suggestions(self, original_data, observation, suggestions,
                                                output_dir, safe_name, test_code, display_name):
        """Create test scenario files based on Claude suggestions."""
        # Combine regular test values and edge cases
        all_test_values = suggestions.get('test_values', [])

        # If no test values were provided, generate some reasonable defaults
        if not all_test_values:
            logger.warning(f"No test values provided for {display_name}. Generating default values.")
            # Extract reference range to generate default values
            ref_range = None
            if observation.get('referenceRange'):
                ref_range = observation['referenceRange'][0]
                low_value = None
                high_value = None

                if 'low' in ref_range:
                    low_value = ref_range['low'].get('value')
                if 'high' in ref_range:
                    high_value = ref_range['high'].get('value')

                # Generate some reasonable default values if we have a range
                if low_value is not None and high_value is not None:
                    all_test_values = [
                        low_value - 1,  # Below min
                        low_value,  # At min
                        low_value + 1,  # Just above min
                        (low_value + high_value) / 2,  # Middle
                        high_value - 1,  # Just below max
                        high_value,  # At max
                        high_value + 1  # Above max
                    ]
                elif low_value is not None:
                    # For ranges like ">3.0 ng/mL"
                    all_test_values = [
                        low_value - 1,  # Below min
                        low_value,  # At min
                        low_value + 1,  # Just above min
                        low_value * 2  # Well above min
                    ]

            # If we still don't have test values, use the current value as a base
            if not all_test_values and 'valueQuantity' in observation:
                current_value = observation['valueQuantity'].get('value')
                if current_value is not None:
                    all_test_values = [
                        current_value * 0.5,  # Half current
                        current_value,  # Current
                        current_value * 1.5  # 1.5x current
                    ]

        # Add edge case values
        for case in suggestions.get('edge_cases', []):
            if 'value' in case and case['value'] not in all_test_values:
                all_test_values.append(case['value'])

        files_created = 0
        # Create test files for each value
        for value in all_test_values:
            try:
                variant_data = copy.deepcopy(original_data)

                # Create test file with standard interpretation
                success = self._create_test_file(
                    variant_data,
                    observation,
                    value,
                    output_dir,
                    safe_name,
                    None,
                    None
                )

                if success:
                    files_created += 1
            except Exception as e:
                logger.warning(f"Error creating test file for value {value}: {str(e)}")

        # Create test files for custom interpretations
        for interp in suggestions.get('custom_interpretations', []):
            if all(k in interp for k in ['value', 'code', 'display']):
                try:
                    variant_data = copy.deepcopy(original_data)
                    success = self._create_test_file(
                        variant_data,
                        observation,
                        interp['value'],
                        output_dir,
                        safe_name,
                        interp['code'],
                        interp['display']
                    )

                    if success:
                        files_created += 1
                except Exception as e:
                    logger.warning(f"Error creating test file for custom interpretation: {str(e)}")

        logger.info(f"Created {files_created} test files for {display_name}")
        return files_created

    def _create_test_file(self, variant_data, observation, value, output_dir,
                          safe_name, interp_code, interp_display):
        """Create a test file with modified observation value and interpretation."""
        try:
            # Find and update the observation in the variant data
            found_observation = False
            for entry in variant_data.get('entry', []):
                resource = entry.get('resource', {})
                if (resource.get('resourceType') == 'Observation' and
                        resource.get('id') == observation.get('id')):
                    found_observation = True

                    # Ensure value is appropriately formatted
                    numeric_value = value
                    if isinstance(numeric_value, float) and numeric_value.is_integer():
                        numeric_value = int(numeric_value)

                    # Update valueQuantity
                    if 'valueQuantity' in resource:
                        resource['valueQuantity']['value'] = numeric_value

                    # Update interpretation
                    if interp_code or interp_display:
                        # Create the new interpretation coding object
                        new_interpretation = {
                            "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                        }

                        # Set code and display
                        if interp_code:
                            new_interpretation["code"] = interp_code
                        if interp_display:
                            new_interpretation["display"] = interp_display

                        # Update in resource
                        if 'interpretation' in resource:
                            if resource['interpretation'] and len(resource['interpretation']) > 0:
                                if 'coding' in resource['interpretation'][0]:
                                    resource['interpretation'][0]['coding'][0] = new_interpretation
                                else:
                                    resource['interpretation'][0]['coding'] = [new_interpretation]
                            else:
                                resource['interpretation'] = [{"coding": [new_interpretation]}]
                        else:
                            resource['interpretation'] = [{"coding": [new_interpretation]}]
                    # If no custom interpretation is provided, determine it automatically
                    else:
                        # Check if we have a reference range to determine interpretation
                        if resource.get('referenceRange'):
                            ref_range = resource['referenceRange'][0]
                            low_value = None
                            high_value = None

                            if 'low' in ref_range:
                                low_value = ref_range['low'].get('value')
                            if 'high' in ref_range:
                                high_value = ref_range['high'].get('value')

                            # Parse text representation for ranges like ">3.0 ng/mL"
                            if not low_value and not high_value and 'text' in ref_range:
                                text = ref_range['text']
                                if text.startswith('>'):
                                    try:
                                        min_value_str = text.split('>')[1].split(' ')[0]
                                        low_value = float(min_value_str)
                                    except (IndexError, ValueError):
                                        pass

                            # Determine appropriate interpretation
                            interpretation_code = "N"  # Default Normal
                            interpretation_display = "Normal"

                            if low_value is not None and numeric_value < low_value:
                                interpretation_code = "L"
                                interpretation_display = "Low"
                            elif high_value is not None and numeric_value > high_value:
                                interpretation_code = "H"
                                interpretation_display = "High"

                            # Update the interpretation
                            new_interpretation = {
                                "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                                "code": interpretation_code,
                                "display": interpretation_display
                            }

                            if 'interpretation' in resource:
                                if resource['interpretation'] and len(resource['interpretation']) > 0:
                                    if 'coding' in resource['interpretation'][0]:
                                        resource['interpretation'][0]['coding'][0] = new_interpretation
                                    else:
                                        resource['interpretation'][0]['coding'] = [new_interpretation]
                                else:
                                    resource['interpretation'] = [{"coding": [new_interpretation]}]
                            else:
                                resource['interpretation'] = [{"coding": [new_interpretation]}]

                    # Get the interpretation for filename
                    interpretation_code = "auto"
                    if interp_code:
                        interpretation_code = interp_code
                    elif (resource.get('interpretation') and
                          resource['interpretation'][0].get('coding') and
                          resource['interpretation'][0]['coding'][0]):
                        interpretation_code = resource['interpretation'][0]['coding'][0].get('code', 'auto')

                    # Sanitize the value for the filename
                    value_str = str(value).replace('.', '_')

                    # Save the variant file
                    output_file = f"{output_dir}/{safe_name}_value_{value_str}_interp_{interpretation_code}.json"
                    with open(output_file, 'w') as f:
                        json.dump(variant_data, f, indent=2)

                    logger.info(f"Created test file: {os.path.basename(output_file)}")
                    return True

            if not found_observation:
                logger.warning(f"Could not find observation with ID {observation.get('id')} in the data")
                return False

        except Exception as e:
            logger.error(f"Error creating test file: {str(e)}")
            return False


def main():
    """Main function to run the Claude test generator."""
    parser = argparse.ArgumentParser(description='Test scenario generator powered by Claude API')
    parser.add_argument('--api-key', required=True, help='Claude API key')
    parser.add_argument('--input', '-i', required=True, help='Input FHIR JSON file path')
    parser.add_argument('--output', '-o', default='test_files', help='Output directory for test files')
    parser.add_argument('--model', default='claude-3-sonnet-20240229', help='Claude model to use')

    args = parser.parse_args()

    # Initialize and run generator
    generator = ClaudeTestGenerator(api_key=args.api_key, model=args.model)
    success = generator.generate_test_scenarios(args.input, args.output)

    if success:
        print(f"\nTest scenarios generated successfully in: {args.output}")
        print(f"Summary report created: {args.output}/test_scenarios_summary.xlsx")
    else:
        print("\nFailed to generate test scenarios. Check the logs for details.")


if __name__ == "__main__":
    main()
