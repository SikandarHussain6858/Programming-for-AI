from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

app = FastAPI(title="Default Prediction API", description="Predicts likelihood of default based on user data")

# Define the input schema with validation
class PredictionInput(BaseModel):
    age: float = Field(..., gt=0, lt=120, description="Age of the individual")
    income: float = Field(..., ge=0, description="Annual income in USD")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    years_employed: int = Field(..., ge=0, le=60, description="Years of employment")

class PredictionOutput(BaseModel):
    prediction: int
    probability: float

# Load model and scaler
MODEL_PATH = os.path.join('model', 'model.pkl')
SCALER_PATH = os.path.join('model', 'scaler.pkl')

model = None
scaler = None

@app.on_event("startup")
def load_assets():
    global model, scaler
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        print("Model or scaler not found. Ensure train.py has been run.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Default Prediction API. Send a POST request to /predict."}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    try:
        # Create wealth index (feature engineering applied in training)
        wealth_index = data.income * data.age
        
        # Prepare input data
        input_df = pd.DataFrame([{
            'age': data.age,
            'income': data.income,
            'credit_score': data.credit_score,
            'years_employed': data.years_employed,
            'wealth_index': wealth_index
        }])
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1] # Probability of class 1
        
        return PredictionOutput(
            prediction=int(pred),
            probability=float(prob)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
