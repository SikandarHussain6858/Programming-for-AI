import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

mlflow.start_run()

# Load dataset
X, y = load_iris(return_X_y=True)

# Model (change this later for versioning)
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Accuracy
accuracy = model.score(X, y)

# Logging
mlflow.log_param("model_type", "LogisticRegression")
mlflow.log_param("max_iter", 200)
mlflow.log_metric("accuracy", accuracy)

# Save model
mlflow.sklearn.log_model(model, "model")

mlflow.end_run()