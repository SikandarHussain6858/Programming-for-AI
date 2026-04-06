import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib

# Features: [gpa, semester, study_hours, department_code]
X = np.array([
    [3.8, 3, 5.0, 0],
    [3.5, 4, 4.0, 1],
    [2.1, 2, 1.0, 2],
    [3.9, 6, 6.0, 0],
    [2.7, 5, 2.0, 3],
    [3.2, 3, 3.0, 1],
    [1.9, 2, 1.0, 2],
    [3.6, 7, 5.5, 3]
])

# Labels: 1 = eligible, 0 = not eligible
y = np.array([1, 1, 0, 1, 0, 1, 0, 1])

# TODO 1: Create DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)

# TODO 2: Train the model
model.fit(X, y)

# TODO 3: Save the trained model as scholarship_model.pkl
joblib.dump(model, "scholarship_model.pkl")

print("Model saved successfully.")
