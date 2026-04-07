from pydantic import BaseModel, Field, field_validator


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=200, description="User text to analyze")
    intensity: int = Field(..., ge=1, le=10, description="Emotion intensity from 1 to 10")

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text cannot be empty or whitespace only")
        return value


class AnalyzeResponse(BaseModel):
    emotion: str
    intensity_level: str
    original_text: str