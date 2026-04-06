import requests

url = "http://127.0.0.1:8000/predict"

tests = [
    {
        "name": "valid",
        "data": {
            "name": "Sara Khan",
            "gpa": 3.7,
            "email": "sara@gmail.com",
            "semester": 4,
            "study_hours": 5.5,
            "department": "AI"
        }
    },
    {
        "name": "invalid_type",
        "data": {
            "name": "Ahmed",
            "gpa": "high",
            "email": "ahmed@gmail.com",
            "semester": 4,
            "study_hours": 3.0,
            "department": "CS"
        }
    }
]

results = {}

for test in tests:
    r = requests.post(url, json=test["data"])
    results[test["name"]] = {
        "status_code": r.status_code,
        "response": r.text
    }

import json
with open("test_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Test results saved.")
