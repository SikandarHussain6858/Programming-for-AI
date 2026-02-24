import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
import joblib


def load_and_explore_data():
    print("\ntask 1: classification - titanic survival prediction")
    
    df = sns.load_dataset("titanic")
    
    print("\ndataset shape:", df.shape)
    print("\nfirst 5 rows:")
    print(df.head())
    
    print("\ndataset info:")
    df.info()
    
    print("\nmissing values:")
    print(df.isnull().sum())
    
    print("\nnumerical columns:", df.select_dtypes(include=["int64", "float64"]).columns.tolist())
    print("categorical columns:", df.select_dtypes(include=["object", "category", "bool"]).columns.tolist())
    
    return df


def prepare_data(df):
    df = df.drop(columns=["alive", "embark_town", "class", "who"], errors='ignore')
    
    X = df.drop("survived", axis=1)
    y = df["survived"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test


def manual_preprocessing(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns
    
    num_imputer = SimpleImputer(strategy="mean")
    cat_imputer = SimpleImputer(strategy="most_frequent", keep_empty_features=True)
    
    X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
    X_test[num_cols] = num_imputer.transform(X_test[num_cols])
    
    X_train_cat_imputed = cat_imputer.fit_transform(X_train[cat_cols].astype(str))
    X_test_cat_imputed = cat_imputer.transform(X_test[cat_cols].astype(str))
    
    X_train_cat_df = pd.DataFrame(X_train_cat_imputed, columns=cat_cols, index=X_train.index)
    X_test_cat_df = pd.DataFrame(X_test_cat_imputed, columns=cat_cols, index=X_test.index)
    
    X_train[cat_cols] = X_train_cat_df
    X_test[cat_cols] = X_test_cat_df
    
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    return X_train, X_test, num_cols, cat_cols


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    print("\ntraining models - manual preprocessing")
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC()
    }
    
    results = []
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)
        precision = precision_score(y_test, test_preds)
        recall = recall_score(y_test, test_preds)
        f1 = f1_score(y_test, test_preds)
        
        results.append({
            "Model": name,
            "Train Accuracy": train_acc,
            "Test Accuracy": test_acc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })
        
        print(f"\n{name}")
        print(f"  train accuracy: {train_acc:.4f}")
        print(f"  test accuracy:  {test_acc:.4f}")
        print(f"  precision:      {precision:.4f}")
        print(f"  recall:         {recall:.4f}")
        print(f"  f1 score:       {f1:.4f}")
    
    return pd.DataFrame(results), models


def demonstrate_leakage(df):
    print("\ndemonstrating data leakage (incorrect approach)")
    
    df = df.drop(columns=["alive", "embark_town", "class", "who"], errors='ignore')
    X = df.drop("survived", axis=1)
    y = df["survived"]
    
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns
    
    num_imputer = SimpleImputer(strategy="mean")
    cat_imputer = SimpleImputer(strategy="most_frequent", keep_empty_features=True)
    
    X[num_cols] = num_imputer.fit_transform(X[num_cols])
    
    X_cat_imputed = cat_imputer.fit_transform(X[cat_cols].astype(str))
    X_cat_df = pd.DataFrame(X_cat_imputed, columns=cat_cols, index=X.index)
    X[cat_cols] = X_cat_df
    
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)
    
    test_preds = log_model.predict(X_test)
    leakage_acc = accuracy_score(y_test, test_preds)
    leakage_f1 = f1_score(y_test, test_preds)
    
    print(f"\nlogistic regression with data leakage:")
    print(f"  test accuracy: {leakage_acc:.4f}")
    print(f"  f1 score:      {leakage_f1:.4f}")
    print("\nproblem: preprocessing was done on entire dataset before split!")
    print("this allows test data statistics to leak into training")
    
    return leakage_acc, leakage_f1


def build_sklearn_pipeline(X_train, X_test, y_train, y_test, num_cols, cat_cols):
    print("\nsklearn pipeline approach")
    print("-" * 50)
    
    from sklearn.preprocessing import OrdinalEncoder
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols.tolist()),
            ('cat', categorical_transformer, cat_cols.tolist())
        ])
    
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', LogisticRegression(max_iter=1000))
    ])
    
    pipeline.fit(X_train, y_train)
    
    train_preds = pipeline.predict(X_train)
    test_preds = pipeline.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)
    
    print(f"\npipeline logistic regression:")
    print(f"  train accuracy: {train_acc:.4f}")
    print(f"  test accuracy:  {test_acc:.4f}")
    print(f"  f1 score:       {f1:.4f}")
    
    joblib.dump(pipeline, "titanic_pipeline.joblib")
    print("\npipeline saved to: titanic_pipeline.joblib")
    
    return pipeline


def main():
    df = load_and_explore_data()
    
    X_train, X_test, y_train, y_test = prepare_data(df)
    print(f"\ndata split: train={len(X_train)}, test={len(X_test)}")
    
    X_train_processed, X_test_processed, num_cols, cat_cols = manual_preprocessing(X_train, X_test)
    
    results_df, trained_models = train_and_evaluate_models(
        X_train_processed, X_test_processed, y_train, y_test
    )
    
    print("\nsummary table:")
    print(results_df.to_string(index=False))
    
    leakage_acc, leakage_f1 = demonstrate_leakage(df)
    
    X_train_original, X_test_original, y_train_original, y_test_original = prepare_data(df)
    pipeline = build_sklearn_pipeline(
        X_train_original, X_test_original, 
        y_train_original, y_test_original,
        num_cols, cat_cols
    )
    
    print("\nfinal comparison")
    print("-" * 50)
    print(f"\n1. manual workflow (leakage-safe):")
    print(f"   best model: {results_df.loc[results_df['Test Accuracy'].idxmax(), 'Model']}")
    print(f"   test accuracy: {results_df['Test Accuracy'].max():.4f}")
    
    print(f"\n2. leakage workflow :")
    print(f"   test accuracy: {leakage_acc:.4f}")
    
    print(f"\n3. pipeline workflow :")
    print("\ntask 1 complete")


if __name__ == "__main__":
    main()
