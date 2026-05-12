import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

def create_synthetic_data(file_path):
    # Create a synthetic dataset
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'years_employed': np.random.randint(0, 40, n_samples),
        'default': np.random.randint(0, 2, n_samples)
    }
    
    # Introduce some missing values
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 50, replace=False), 'income'] = np.nan
    df.loc[np.random.choice(df.index, 30, replace=False), 'age'] = np.nan
    
    df.to_csv(file_path, index=False)
    print(f"Dataset created at {file_path}")

def train_pipeline():
    data_dir = 'data'
    model_dir = 'model'
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    data_path = os.path.join(data_dir, 'dataset.csv')
    
    if not os.path.exists(data_path):
        create_synthetic_data(data_path)
        
    print("1. Loading and preprocessing dataset")
    df = pd.read_csv(data_path)
    
    print("2. Handling missing or invalid data")
    # Fill missing 'income' with median
    df['income'] = df['income'].fillna(df['income'].median())
    # Fill missing 'age' with mean
    df['age'] = df['age'].fillna(df['age'].mean())
    
    print("3. Applying feature engineering")
    # Feature engineering: wealth index
    df['wealth_index'] = df['income'] * df['age']
    
    X = df[['age', 'income', 'credit_score', 'years_employed', 'wealth_index']]
    y = df['default']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("4. Training machine learning models")
    # Experiment 1: Logistic Regression
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_preds = lr_model.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, lr_preds)
    
    # Experiment 2: Random Forest
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(X_train_scaled, y_train)
    rf_preds = rf_model.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_preds)
    
    print("5. Evaluating model performance & 6. Comparing models")
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    
    if rf_acc >= lr_acc:
        best_model = rf_model
        best_name = "Random Forest"
    else:
        best_model = lr_model
        best_name = "Logistic Regression"
        
    print(f"Selected best model: {best_name}")
    
    # Save the model and scaler
    model_path = os.path.join(model_dir, 'model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    print("Model and scaler saved successfully.")

if __name__ == "__main__":
    train_pipeline()
