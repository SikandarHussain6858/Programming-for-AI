from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr, Field
from typing import Literal
import joblib
import numpy as np
import os

# TODO 1: Create FastAPI app
app = FastAPI()

MODEL_PATH = "scholarship_model.pkl"

# TODO 2: Load model once
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None


# TODO 3: Define input schema
class StudentInput(BaseModel):
    name: str = Field(..., min_length=3, max_length=30)
    gpa: float = Field(..., ge=0.0, le=4.0)
    email: EmailStr
    semester: int = Field(..., ge=1, le=8)
    study_hours: float = Field(..., ge=0.0, le=24.0)
    department: Literal["AI", "CS", "DS", "SE"]


# TODO 4: Root endpoint
@app.get("/")
def home():
    return {
        "message": "Welcome to Student Scholarship Prediction API",
        "docs": "/docs"
    }


# TODO 5: Health check endpoint
@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {"status": "ok", "model_loaded": True}


# TODO 6: Prediction endpoint
@app.post("/predict")
def predict(data: StudentInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Prediction model unavailable")

    # Manual exception handling example
    if data.gpa < 2.0 and data.study_hours > 10:
        raise HTTPException(
            status_code=400,
            detail="Unusual input combination detected"
        )

    try:
        dept_mapping = {
            "AI": 0,
            "CS": 1,
            "DS": 2,
            "SE": 3
        }

        # TODO 7: Prepare feature vector
        features = np.array([[
            data.gpa,
            data.semester,
            data.study_hours,
            dept_mapping[data.department]
        ]])

        # TODO 8: Run prediction
        prediction = model.predict(features)[0]

        # TODO 9: Convert prediction to bool eligibility
        eligible = bool(prediction)

        if eligible:
            msg = "Student is likely eligible for scholarship."
        else:
            msg = "Student is unlikely to qualify for scholarship."

        # TODO 10: Return JSON response
        return {
            "student_name": data.name,
            "predicted_class": int(prediction),
            "scholarship_eligible": eligible,
            "message": msg
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
