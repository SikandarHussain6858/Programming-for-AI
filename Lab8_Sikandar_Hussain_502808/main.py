from fastapi import FastAPI, HTTPException
from models import AnalyzeRequest, AnalyzeResponse

app = FastAPI(
    title="Lab 8 AI Emotion API",
    description="FastAPI service with validation, exception handling, and Swagger testing",
    version="1.0.0"
)


@app.get("/")
def home():
    return {"message": "Welcome to Lab 8 AI API"}


def detect_emotion(text: str) -> str:
    t = text.lower()
    if "happy" in t or "great" in t or "excited" in t:
        return "Happy"
    if "sad" in t or "down" in t or "upset" in t:
        return "Sad"
    if "angry" in t or "mad" in t:
        return "Angry"
    return "Neutral"


def map_intensity_level(intensity: int) -> str:
    if intensity <= 3:
        return "Low"
    if intensity <= 7:
        return "Medium"
    return "High"


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(data: AnalyzeRequest):
    try:
        if "error" in data.text.lower():
            raise RuntimeError("Simulated model runtime failure for testing")

        emotion = detect_emotion(data.text)
        intensity_level = map_intensity_level(data.intensity)

        return AnalyzeResponse(
            emotion=emotion,
            intensity_level=intensity_level,
            original_text=data.text
        )

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Model processing failed: {str(e)}")
    except Exception:
        raise HTTPException(status_code=500, detail="Unexpected server error")