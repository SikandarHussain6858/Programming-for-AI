import urllib.request
import urllib.error
import json

test_cases = [
    {"age": 35, "income": 65000, "credit_score": 720, "years_employed": 8},
    {"age": 22, "income": 20000, "credit_score": 500, "years_employed": 1},
    {"age": 55, "income": 120000, "credit_score": 800, "years_employed": 30},
    {"age": 40, "income": 45000, "credit_score": 600, "years_employed": 5},
]

print("Sending multiple requests to the API...\n")

for i, data in enumerate(test_cases, 1):
    req = urllib.request.Request(
        'http://localhost:8000/predict', 
        method='POST', 
        headers={'Content-Type': 'application/json'}, 
        data=json.dumps(data).encode('utf-8')
    )

    try:
        resp = urllib.request.urlopen(req)
        result = resp.read().decode('utf-8')
        print(f"Request {i}:")
        print(f"Input: {data}")
        print(f"Output: {result}\n")
    except urllib.error.HTTPError as e:
        print(f"Request {i} Failed:")
        print(f"Input: {data}")
        print(f"Error: {e.read().decode('utf-8')}\n")
