# Lab 4: Building Leakage-Safe ML Pipelines

**Student Name:** Sikandar Hussain  
**Roll Number:** 502808  
**Course:** DS201 - Programming for AI  
**Instructor:** Dr. Mehwish Fatima  
**Lab Engineer:** Mr. Hafiz Arslan Ramzan  
**Date:** 17-02-2026

---

## Overview

This lab implements complete machine learning pipelines for both classification and regression tasks, with a strong focus on preventing data leakage and ensuring reproducibility. The project demonstrates proper train/test splitting, preprocessing techniques, model training, evaluation, and serialization.

---

## Datasets

### Task 1: Classification - Titanic Dataset
- **Source:** Seaborn built-in dataset
- **Loading:** `sns.load_dataset("titanic")`
- **Target Variable:** `survived` (0 = No, 1 = Yes)
- **Features:** Passenger demographics, ticket information, cabin details
- **Size:** 891 samples

### Task 2: Regression - California Housing Dataset
- **Source:** Scikit-learn built-in dataset
- **Loading:** `fetch_california_housing(as_frame=True)`
- **Target Variable:** `MedHouseVal` (median house value in $100,000s)
- **Features:** 8 numerical features including location, demographics, and housing characteristics
- **Size:** 20,640 samples

---

## Project Structure

```
Lab4_Sikandar_Hussain_502808/
├── task1_classification.py      # Classification pipeline implementation
├── task2_regression.py          # Regression pipeline implementation
├── titanic_pipeline.joblib      # Serialized classification pipeline
├── housing_pipeline.joblib      # Serialized regression pipeline
└── README.md                    # This file
```

---

## Installation & Setup

### Prerequisites
```bash
pip install pandas seaborn scikit-learn joblib numpy
```

### Running the Scripts

**Task 1 - Classification:**
```bash
cd Lab4_Sikandar_Hussain_502808
python task1_classification.py
```

**Task 2 - Regression:**
```bash
python task2_regression.py
```

---

## Task 1: Classification Pipeline

### Train/Test Split Configuration
- **Test Size:** 20% (0.2)
- **Random State:** 42 (for reproducibility)
- **Split Method:** Stratified by default in train_test_split

### Preprocessing Steps

1. **Feature Selection**
   - Dropped irrelevant columns: `alive`, `embark_town`, `class`, `who`
   - Separated features (X) and target (y)

2. **Missing Value Imputation** (fitted on train set only)
   - Numerical features: Mean imputation
   - Categorical features: Most frequent value imputation

3. **Scaling** (fitted on train set only)
   - StandardScaler applied to numerical features
   - Formula: `z = (x - μ) / σ`

4. **Encoding**
   - One-Hot Encoding for categorical features
   - `drop_first=True` to avoid multicollinearity

### Models Trained

1. **Logistic Regression**
   - Hyperparameters: `max_iter=1000`
   - Linear classification model

2. **Random Forest Classifier**
   - Hyperparameters: `random_state=42`
   - Ensemble of decision trees

3. **Support Vector Machine (SVM)**
   - Default RBF kernel
   - Non-linear decision boundary

### Evaluation Metrics

| Model | Train Accuracy | Test Accuracy | F1 Score | Precision | Recall |
|-------|---------------|---------------|----------|-----------|--------|
| Logistic Regression | 0.8371 | 0.8156 | 0.7755 | 0.7808 | 0.7703 |
| Random Forest | 0.9846 | 0.7933 | 0.7448 | 0.7606 | 0.7297 |
| SVM | 0.8357 | 0.8156 | 0.7660 | 0.8060 | 0.7297 |

**Best Model:** Logistic Regression and SVM (tied for test accuracy)

### Leakage Experiment

**Incorrect Approach:**
- Preprocessed entire dataset (imputation + scaling)
- Then performed train/test split

**Result:**
- Test accuracy may appear slightly higher
- This is **invalid** because test set statistics leaked into training

**What Leaked:**
- Mean and standard deviation from test set influenced scaling parameters
- Most frequent categories from test set influenced imputation
- Model indirectly "saw" test data during preprocessing

### Sklearn Pipeline Implementation

```python
Pipeline([
    ('preprocess', ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ])),
    ('model', LogisticRegression(max_iter=1000))
])
```

### Model Serialization
- **File:** `titanic_pipeline.joblib`
- **Method:** `joblib.dump(pipeline, filename)`
- **Contains:** All preprocessing steps + trained model
- **Usage:** `loaded_pipeline = joblib.load(filename)`

---

## Task 2: Regression Pipeline

### Train/Test Split Configuration
- **Test Size:** 20% (0.2)
- **Random State:** 42
- **No missing values** in this dataset

### Preprocessing Steps

1. **Scaling** (fitted on train set only)
   - StandardScaler for Linear Regression
   - No scaling needed for Random Forest (tree-based)

### Models Trained

1. **Linear Regression**
   - Assumes linear relationship between features and target
   - Requires scaled features

2. **Random Forest Regressor**
   - Ensemble of regression trees
   - Hyperparameters: `n_estimators=100`, `random_state=42`
   - Does not require scaling

### Evaluation Metrics

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | 0.5333 | 0.7271 | 0.5757 |
| Random Forest | 0.3275 | 0.4983 | 0.8065 |

**Best Model:** Random Forest (significantly better R² and lower error)

**Why Random Forest Performs Better:**
- Captures non-linear relationships
- Handles feature interactions automatically
- More robust to outliers

### Leakage Experiment
- Scaled entire dataset before splitting
- Results in optimistically biased R² score
- Test set mean/std leaked into scaling parameters

### Sklearn Pipeline Implementation

```python
Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
```

### Model Serialization
- **File:** `housing_pipeline.joblib`
- **Ensures:** Raw input → Scaling → Prediction workflow preserved
- **Deployment Ready:** Can be used directly on new data

---

## Reflection Questions & Answers

### 1. Why must we split before preprocessing?
**Answer:** Preprocessing parameters (mean, std, most frequent value) must be learned only from training data. If we preprocess before splitting, test set statistics influence these parameters, causing data leakage and overly optimistic performance estimates.

### 2. What is data leakage?
**Answer:** Data leakage occurs when information from outside the training dataset is used to create the model. This includes letting test set statistics influence preprocessing or feature engineering, leading to unrealistic performance metrics that won't generalize to new data.

### 3. Why can small leakage significantly inflate performance?
**Answer:** Even small amounts of leaked information can help the model "cheat" by incorporating patterns from the test set. This makes the model appear more accurate than it actually is, leading to poor real-world performance where such information isn't available.

### 4. What is the role of fit() vs transform()?
**Answer:** 
- `fit()`: Learns parameters from the training data (e.g., mean, std for scaling)
- `transform()`: Applies learned parameters to transform data
- `fit_transform()`: Combines both steps (only use on training data)
- Test data should only use `transform()` with parameters learned from training

### 5. Why is Pipeline safer than manual steps?
**Answer:** 
- Ensures preprocessing and training steps are always applied in correct order
- Prevents accidentally fitting preprocessors on test data
- Encapsulates entire workflow for reproducibility
- Makes deployment easier by packaging everything together
- Reduces human error in multi-step workflows

### 6. Why not use accuracy in regression?
**Answer:** Accuracy measures exact matches (correct vs incorrect), which doesn't make sense for continuous values. Regression uses MAE (average error), RMSE (penalizes large errors), and R² (variance explained) instead.

### 7. Why is RMSE larger than MAE?
**Answer:** RMSE squares errors before averaging, which penalizes larger errors more heavily. After squaring, large errors dominate the metric, resulting in a higher value than MAE which treats all errors equally.

### 8. Why doesn't Random Forest require scaling?
**Answer:** Tree-based models make decisions using feature splits (e.g., age > 30), not distances. They are invariant to monotonic transformations like scaling. Only distance-based models (SVM, KNN, Linear) require scaling.

---

## Model Comparison

### Classification Models
| Approach | Safety | Reproducibility | Deployment Ready |
|----------|--------|-----------------|------------------|
| Manual Workflow | ✅ Safe (if done correctly) | ⚠️ Prone to errors | ❌ Multiple components |
| Leakage Workflow | ❌ Unsafe | ❌ Invalid results | ❌ Not production-ready |
| Pipeline Workflow | ✅ Safest | ✅ Fully reproducible | ✅ Single artifact |

**Recommendation:** Always use sklearn Pipelines for production workflows.

### Regression Models
- **Linear Regression:** Good baseline, interpretable coefficients
- **Random Forest:** Better performance, captures non-linearity
- **Choice depends on:** Need for interpretability vs. predictive accuracy

---

## Key Takeaways

1. **Always split before preprocessing** to prevent data leakage
2. **Use `fit()` only on training data**, `transform()` on both train and test
3. **Sklearn Pipelines** are the safest and most reproducible approach
4. **Serialization** enables deployment without retraining
5. **Different metrics** for classification (accuracy, F1) vs regression (MAE, RMSE, R²)
6. **Tree-based models** don't need scaling, linear models do

---

## Assumptions & Considerations

1. **Random State:** Fixed at 42 for reproducibility across runs
2. **Missing Values:** Handled with simple imputation strategies
3. **Feature Selection:** Dropped columns identified as irrelevant
4. **Model Hyperparameters:** Used mostly defaults for baseline comparison
5. **Evaluation:** Used single train/test split (could use cross-validation for more robust estimates)

---

## Future Enhancements

- Cross-validation for more robust performance estimates
- Hyperparameter tuning using GridSearchCV
- Feature engineering and selection
- Handling class imbalance (SMOTE, class weights)
- Advanced imputation strategies (KNN, iterative)
- Model interpretation (SHAP values, feature importance)

---

## Conclusion

This lab successfully demonstrates building leakage-safe ML pipelines for both classification and regression tasks. The use of sklearn Pipelines ensures reproducibility, prevents data leakage, and creates deployment-ready artifacts. The experiments clearly show the importance of proper preprocessing order and the dangers of data leakage.

---

## References

- Scikit-learn Documentation: https://scikit-learn.org/
- Seaborn Dataset Gallery: https://github.com/mwaskom/seaborn-data
- sklearn Pipeline Guide: https://scikit-learn.org/stable/modules/compose.html
