import requests
import time
import json

url = "http://localhost:9999/predict"
headers = {"Content-Type": "application/json"}

with open("large-json.json", "r") as f:
    payload = json.load(f)

start_time = time.time()
response = requests.post(url, json=payload, headers=headers)
end_time = time.time()

print(f"Response Time: {end_time - start_time:.2f} seconds")
print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")
