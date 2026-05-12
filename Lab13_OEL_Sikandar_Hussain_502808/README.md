# Default Prediction API

This is a small production-ready AI service that predicts the likelihood of a default based on user data.

## Project Structure
- `data/`: Contains the generated synthetic dataset with some missing values.
- `model/`: Contains the trained machine learning model and the scaler.
- `screenshots/`: Folder for any screenshots needed for the viva/demo.
- `main.py`: FastAPI backend that serves the model.
- `train.py`: Pipeline for data creation, preprocessing, model training, evaluation, and saving.
- `Dockerfile`: Configuration for containerizing the application.
- `requirements.txt`: Python dependencies.

## Setup Instructions

### 1. Local Setup
Create a virtual environment and install requirements:
```bash
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Training the Model
Run the training script to generate data, preprocess it, train models (Logistic Regression vs Random Forest), evaluate them, and save the best model:
```bash
python train.py
```

### 3. Running the API
Start the FastAPI server:
```bash
uvicorn main:app --reload
```
You can then access the interactive API docs at `http://127.0.0.1:8000/docs`.

### 4. Docker Deployment
Build and run the application using Docker:
```bash
# Build the image
docker build -t default-prediction-api .

# Run the container
docker run -p 8000:8000 default-prediction-api
```
Access the docs at `http://localhost:8000/docs`.

## Pipeline Workflow (Viva/Demo)
1. **Data**: We generate a synthetic dataset representing individuals with features like `age`, `income`, `credit_score`, and `years_employed`. We introduce some missing values to simulate real-world data.
2. **Preprocessing**: Missing values are imputed (median for income, mean for age). We also apply feature engineering by creating a `wealth_index` (age * income).
3. **Model**: We train a Logistic Regression model and a Random Forest model. We evaluate both on a test set, compare their accuracy, and save the better performing model.
4. **API**: A FastAPI endpoint (`/predict`) takes user input, validates it using Pydantic, applies the saved scaler, and returns the prediction and probability from the loaded model.
5. **Deployment**: The API is containerized using Docker, making it easy to deploy to any environment.
