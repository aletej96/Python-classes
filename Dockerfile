# Use Python 3.11.11 as the base image
FROM python:3.11.11-slim

# Upgrade pip and install the necessary dependencies
RUN pip install --upgrade pip && pip install mlflow scikit-learn pandas numpy

# Serve the model from the specific MLflow run ID
CMD ["mlflow", "models", "serve", "-m", "runs:/0bca5ad59bec4263afb153183c87dc9c/model", "-h", "0.0.0.0", "-p", "5000", "--tracking-uri", "http://host.docker.internal:5000"]



