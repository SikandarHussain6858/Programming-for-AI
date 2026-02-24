import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib


def load_and_explore_data():
    print("\ntask 2: regression - california housing price prediction")
    print("-" * 50)
    
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    
    print("\ndataset shape:", df.shape)
    print("\nfirst 5 rows:")
    print(df.head())
    
    print("\ndataset info:")
    df.info()
    
    print("\nstatistical summary:")
    print(df.describe())
    
    print("\nno missing values in this dataset")
    
    return df


def prepare_data(df):
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test


def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print(f"\n{name}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R²": r2}


def train_models_manual(X_train, X_test, y_train, y_test):
    print("\ntraining regression models - manual preprocessing")
    print("-" * 50)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    
    rf = RandomForestRegressor(random_state=42, n_estimators=100)
    rf.fit(X_train, y_train)
    
    results = []
    
    print("\nmodel evaluation:")
    results.append(evaluate_model("Linear Regression", lr, X_test_scaled, y_test))
    results.append(evaluate_model("Random Forest", rf, X_test, y_test))
    
    return pd.DataFrame(results), lr, rf, scaler


def demonstrate_leakage(df):
    print("\ndemonstrating data leakage (incorrect approach)")
    print("-" * 50)
    
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    preds = lr.predict(X_test)
    leakage_r2 = r2_score(y_test, preds)
    
    print(f"\nlinear regression with data leakage:")
    print(f"  r2 score: {leakage_r2:.4f}")
    print("\nproblem: scaling was done on entire dataset before split!")
    print("test set statistics leaked into training phase")
    
    return leakage_r2


def build_sklearn_pipeline(X_train, X_test, y_train, y_test):
    print("\nsklearn pipeline approach")
    print("-" * 50)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    
    pipeline.fit(X_train, y_train)
    
    train_preds = pipeline.predict(X_train)
    test_preds = pipeline.predict(X_test)
    
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)
    mae = mean_absolute_error(y_test, test_preds)
    rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    
    print(f"\npipeline linear regression:")
    print(f"  train r2: {train_r2:.4f}")
    print(f"  test r2:  {test_r2:.4f}")
    print(f"  mae:      {mae:.4f}")
    print(f"  rmse:     {rmse:.4f}")
    
    joblib.dump(pipeline, "housing_pipeline.joblib")
    print("\npipeline saved to: housing_pipeline.joblib")
    
    loaded_pipeline = joblib.load("housing_pipeline.joblib")
    test_preds_loaded = loaded_pipeline.predict(X_test)
    print("pipeline successfully loaded and tested")
    
    return pipeline


def main():
    df = load_and_explore_data()
    
    X_train, X_test, y_train, y_test = prepare_data(df)
    print(f"\ndata split: train={len(X_train)}, test={len(X_test)}")
    
    results_df, lr_model, rf_model, scaler = train_models_manual(
        X_train, X_test, y_train, y_test
    )
    
    print("\nsummary table:")
    print(results_df.to_string(index=False))
    
    leakage_r2 = demonstrate_leakage(df)
    
    pipeline = build_sklearn_pipeline(X_train, X_test, y_train, y_test)
    
    
    print("\nfinal comparison")
    print("-" * 50)
    
    best_model = results_df.loc[results_df['R²'].idxmax()]
    print(f"\n1. manual workflow (leakage-safe):")
    print(f"   best model: {best_model['Model']}")
    print(f"   r2 score: {best_model['R²']:.4f}")
    print(f"   rmse: {best_model['RMSE']:.4f}")
    
    print(f"\n2. leakage workflow (incorrect):")
    print(f"   r2 score: {leakage_r2:.4f}")
    
    print(f"\n3. pipeline workflow (recommended):")
    print(f"   ensures no leakage through proper abstraction")
    print(f"   raw input -> scaling -> prediction preserved")
    
    print("\ntask 2 complete")


if __name__ == "__main__":
    main()
