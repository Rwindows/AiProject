import httpx

client = httpx.Client(verify=False)
response = client.get("https://api.openai.com")
print("Status code (SSL verification disabled):", response.status_code)
