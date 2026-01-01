import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load Wine dataset
wine = load_wine(as_frame=True)
X = wine.data # type: ignore
y = wine.target # type: ignore

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_d = 5
n_es = 5

# Set MLflow tracking URI to local mlruns folder
mlflow.set_tracking_uri("file:///E:/MLOps/mlruns")

mlflow.autolog()
mlflow.set_experiment("new_exp")
with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_d, n_estimators=n_es, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    # mlflow.log_metric("accuracy", float(accuracy))
    # mlflow.log_param("max_depth", max_d)
    # mlflow.log_param("n_estimators", n_es)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names) # type: ignore
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # Save plot in the MLflow folder
    plt.savefig("E:/MLOps/mlruns/Cf-matrix.png")

    # Log artifacts using mlflow
    # mlflow.log_artifact("E:/MLOps/mlruns/Cf-matrix.png")

    # Git logs the files but mlflow also gives this functionality
    # mlflow.log_artifact(__file__) # autolog don't log file automatically

    # tags
    mlflow.set_tags({"Author": "Roman", "project": "mlflow practice"})

    # log the model
    # mlflow.sklearn.log_model(rf, "Random Forest") # type: ignore

    print(accuracy)