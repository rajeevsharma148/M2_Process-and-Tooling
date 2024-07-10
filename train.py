import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data from CSV
data = pd.read_csv('data/iris.csv')
# Verify columns
print(data.columns)

X = data.drop('variety', axis=1)  # 'variety' is the name of the label column
y = data['variety']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set the experiment name
mlflow.set_experiment("Iris RandomForest Classifier Experiment")

# Function to log evaluation artifacts
def log_evaluation_artifacts(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cr = classification_report(y_true, y_pred, output_dict=True)

    # Save confusion matrix as an image
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    cm_image_path = "confusion_matrix.png"
    plt.savefig(cm_image_path)
    plt.close()
    mlflow.log_artifact(cm_image_path)

    # Save classification report as a CSV
    cr_df = pd.DataFrame(cr).transpose()
    cr_csv_path = "classification_report.csv"
    cr_df.to_csv(cr_csv_path)
    mlflow.log_artifact(cr_csv_path)

    # Clean up
    os.remove(cm_image_path)
    os.remove(cr_csv_path)

# Define different sets of parameters for each run
param_sets = [
    {"n_estimators": 100, "max_depth": 3},
    {"n_estimators": 200, "max_depth": 5},
    {"n_estimators": 150, "max_depth": 4}
]

# Start MLflow run and log experiments with different parameters
for i, params in enumerate(param_sets):
    with mlflow.start_run():
        # Define and train model
        model = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=42)
        model.fit(X_train, y_train)

        # Predict and evaluate
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Log parameters, metrics, and model
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("max_depth", params["max_depth"])
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

        # Log the model
        mlflow.sklearn.log_model(model, "model")

        # Save the model locally with a unique name
        model_path = f"model_v{i+1}.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
    
        # Log evaluation artifacts
        log_evaluation_artifacts(y_test, predictions, labels=model.classes_)

        print(f"Logged run {i+1} with parameters: {params} and accuracy: {accuracy}")
