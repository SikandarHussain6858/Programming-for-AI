# Lab 7 - API Stress Testing, Latency Analysis, and System Robustness

**Student:** Sikandar Hussain  
**Roll No:** 502808

## Objective
This lab evaluates a deployed AI API under realistic load conditions by measuring:
- latency
- throughput
- concurrent stress behavior
- failure and slowdown points
- robustness against invalid inputs

## Folder Structure

```
Lab7_Sikandar_Hussain_502808/
|-- main.py
|-- tester.py
|-- results/
|   `-- report.txt
`-- README.md
```

## Requirements
Install dependencies:

```bash
pip install requests
```

## API Under Test
This setup targets the Lab 6 Flask API by default:
- Base URL: `http://127.0.0.1:5000`
- Endpoint: `/predict`

Expected valid payload format:

```json
{
  "pclass": 3,
  "sex": "male",
  "age": 22,
  "sibsp": 1,
  "parch": 0,
  "fare": 7.25,
  "embarked": "S"
}
```

## How To Run
1. Start your API server in one terminal (for example, Lab 6 app).
2. Run Lab 7 tests in another terminal:

```bash
cd Lab7_Sikandar_Hussain_502808
python main.py
```

Optional arguments:

```bash
python main.py --base-url http://127.0.0.1:5000 --endpoint /predict --latency-requests 50 --throughput-requests 150 --concurrent-users 50 --requests-per-user 4
```

If your API requires an auth key:

```bash
python main.py --api-key your_key_here
```

## What Is Evaluated
The framework performs all required tasks:
1. Sends valid API requests and verifies responses
2. Measures latency for multiple requests
3. Computes average latency
4. Measures throughput (requests/second)
5. Performs stress testing with concurrent users
6. Identifies slowdown/failure points via load steps
7. Tests at least 4 edge cases (plus invalid JSON)
8. Logs all results in `results/report.txt`

## Edge Cases Included
- missing fields
- wrong data types
- extreme values
- incorrect shape / missing required field
- invalid JSON body

## Output
After execution, the script writes a full report to:

- `results/report.txt`

The report includes latency stats, throughput, stress metrics, robustness observations, and edge-case outcomes.
