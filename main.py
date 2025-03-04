import json
import os
import ssl

import certifi
import httpx

ssl_context = ssl.create_default_context(cafile=certifi.where())
client = httpx.Client(verify=ssl_context)
response = client.get("https://api.openai.com")
print("Status code:", response.status_code)

import httpx

client = httpx.Client(verify=False)
response = client.get("https://api.openai.com")
print("Status code with verification disabled:", response.status_code)

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# set OPENAI_API_KEY="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"
load_dotenv()

# Fetch API Key
api_key = os.getenv("OPENAI_API_KEY")

# Debugging: Print the value to verify
print(f"API Key: {api_key}")  # Remove after confirming

if not api_key:
    raise ValueError("Error: OPENAI_API_KEY is not set. Please check your .env file.")

os.environ["OPENAI_API_KEY"] = api_key

# Example query to retrieve test instructions
query = query = """
Generate a list of possible login scenarios in JSON format for testing a login page. Include a variety of valid, invalid, and edge cases. Ensure the scenarios cover:
- Valid credentials
- Invalid credentials (incorrect username, incorrect password, both incorrect)
- Missing inputs (empty username, empty password, both fields empty)
- Special characters
- Long inputs
- Case sensitivity (e.g., "Username" vs "username")
- Other potential edge cases

Output should be a JSON array with the following structure:
{
    "scenario": "<Scenario Name>",
    "username": "<username>",
    "password": "<password>",
    "expected_result": "<expected outcome>"
}
Do not include any text outside the JSON array.
"""
# Set up knowledge base
documents = [
    Document(
        page_content="Test login functionality. Generate all valid scenarios using the keys: username=tomsmith, password=SuperSecretPassword!. It should be succsefully login"),
    Document(
        page_content="Test login functionality. Generate all invalid scenarios  like invalid login and invalid password using the keys: username=invalid_user, password=SuperSecretPassword!. It shouldn't succesfully login")
    # Document(page_content="Test the search functionality. Navigate to 'https://make.com', enter 'test query' in the search bar, and click the search button. Verify results are displayed."),
]

embeddings = OpenAIEmbeddings(
    openai_api_key="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")
vectorstore = Chroma.from_documents(documents, embeddings)

# Set up LangChain retriever
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-4"), retriever=retriever)

test_instructions = qa_chain.run(query)
print("Test Instructions:", test_instructions)

try:
    scenarios = json.loads(test_instructions)
    print("Parsed Scenarios:", scenarios)
except json.JSONDecodeError as e:
    print(f"Error parsing JSON: {e}")
    scenarios = []


# Selenium: Execute UI test
# driver = webdriver.Chrome()  # Ensure you have ChromeDriver installed
# try:
#     driver.get("https://make.com/login")
#     driver.find_element(By.XPATH, "//input[@name='email']").send_keys("username")
#     driver.find_element(By.XPATH, "//input[@name='password']").send_keys("password")
#     driver.find_element(By.PARTIAL_LINK_TEXT, "Sign in").click()
# Execute all test scenarios
def execute_scenarios(scenarios):
    driver = webdriver.Chrome()
    try:
        for scenario in scenarios:
            print(f"Executing Scenario: {scenario['scenario']}")

            # Open the login page
            driver.get("http://the-internet.herokuapp.com/login")

            # Enter username
            driver.find_element(By.XPATH, "//input[@name='username']").send_keys(scenario["username"])
            driver.find_element(By.XPATH, "//input[@name='password']").send_keys(scenario["password"])
            driver.find_element(By.XPATH, "//button[@type='submit']").click()

            # Validate the expected result
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            print(f"Expected Result: {scenario['expected_result']}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.quit()


# Execute all test scenarios
if scenarios:
    execute_scenarios(scenarios)
else:
    print("No valid scenarios to execute.")
