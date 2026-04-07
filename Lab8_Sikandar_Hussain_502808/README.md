# Lab 8 - FastAPI Implementation, Swagger Testing, and Input Validation

**Student:** Sikandar Hussain  
**Roll No:** 502808

## Objective
This lab implements a robust AI API using FastAPI with:
- structured request/response models
- input validation using Pydantic
- runtime exception handling
- interactive testing through Swagger UI

The implementation maps to:
- **CLO-1 (C3):** API development, endpoint implementation, and functional testing
- **CLO-4 (P4):** validation handling, exception handling, and API robustness

## Folder Structure

```text
Lab8_Sikandar_Hussain_502808/
|-- main.py
|-- models.py
`-- README.md
```

## Requirements
Install dependencies:

```bash
pip install fastapi uvicorn
```

## How To Run
From the lab folder, start the FastAPI server:

```bash
cd Lab8_Sikandar_Hussain_502808
uvicorn main:app --reload
```

Access URLs:
- API base: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`

## API Endpoints

### 1) Home Endpoint
- **Method:** `GET`
- **Path:** `/`
- **Purpose:** Basic connectivity check

Example response:

```json
{
  "message": "Welcome to Lab 8 AI API"
}
```

### 2) Analyze Endpoint
- **Method:** `POST`
- **Path:** `/analyze`
- **Purpose:** Analyze text emotion and map intensity level

Request body:

```json
{
  "text": "I feel very happy today",
  "intensity": 9
}
```

Response body:

```json
{
  "emotion": "Happy",
  "intensity_level": "High",
  "original_text": "I feel very happy today"
}
```

## Validation Rules (Pydantic)
Input model: `AnalyzeRequest`
- `text`: string, required, min length = 3, max length = 200
- `text`: cannot be blank/whitespace-only (custom validator)
- `intensity`: integer, required, range = 1 to 10

These checks prevent invalid API calls and improve reliability.

## Exception Handling
The `/analyze` endpoint includes `try/except` handling:
- A simulated runtime failure is triggered when `text` contains the keyword `error`
- Runtime failures are returned as HTTP 500 with a clear error message
- Unknown/unexpected failures are also returned as HTTP 500

Example runtime error response:

```json
{
  "detail": "Model processing failed: Simulated model runtime failure for testing"
}
```

## Swagger Testing Steps
1. Open `http://127.0.0.1:8000/docs`
2. Expand `POST /analyze`
3. Click **Try it out**
4. Enter a JSON request body
5. Click **Execute**
6. Observe status code and response body

## Edge Case Testing (Handled)
At least 3 edge cases are required. The following 5 are tested:

| # | Test Case | Input | Expected Status | Behavior |
|---|-----------|-------|-----------------|----------|
| 1 | Missing fields | `{}` | `422` | Validation error for required fields |
| 2 | Wrong data types | `{ "text": 123, "intensity": "high" }` | `422` | Validation error for invalid types |
| 3 | Out-of-range intensity | `{ "text": "I feel okay", "intensity": 11 }` | `422` | Validation error for `intensity <= 10` |
| 4 | Whitespace text | `{ "text": "   ", "intensity": 5 }` | `422` | Custom validator rejects blank text |
| 5 | Runtime processing failure | `{ "text": "this has error keyword", "intensity": 6 }` | `500` | Exception handler returns safe error response |

## Learning Outcomes Achieved
- Built a FastAPI application with clear endpoint design
- Practiced request-response modeling using Pydantic
- Applied validation constraints for safe input handling
- Implemented exception handling for runtime robustness
- Performed API testing with Swagger/OpenAPI
- Handled multiple edge cases to improve production readiness

## Conclusion
This lab demonstrates a production-oriented AI API workflow:

**Model logic -> API endpoint -> Validation -> Exception handling -> Reliable JSON response**

This approach improves system stability, developer testing speed, and user trust in AI-powered services.
