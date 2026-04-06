import requests

# TODO 1: Set correct URL
url = "http://127.0.0.1:8000/predict"

payload = {
    "name": "Sara Khan",
    "gpa": 3.7,
    "email": "sara@gmail.com",
    "semester": 4,
    "study_hours": 5.5,
    "department": "AI"
}

# TODO 2: Send POST request
response = requests.post(url, json=payload)

# TODO 3: Print status code
print("Status Code:", response.status_code)

# TODO 4: Print JSON response
print("Response JSON:", response.json())
