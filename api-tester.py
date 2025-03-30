import requests
import time
import json
from concurrent.futures import ThreadPoolExecutor

url = "http://localhost:9999/predict"
headers = {"Content-Type": "application/json"}

with open("large-json.json", "r") as f:
    payload = json.load(f)

def send_request(i):
    response = requests.post(url, json=payload, headers=headers)
    return response

start_time = time.time()
with ThreadPoolExecutor() as executor:
    responses = list(executor.map(send_request, range(1000)))

end_time = time.time()

# Print out the time taken and status codes for a few of the responses
print(f"Total Time: {end_time - start_time:.2f} seconds")
print(f"Sample Status Code: {responses[0].status_code}")
print(f"Sample Response: {responses[0].json()}")
