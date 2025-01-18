# Introduction to Python for Data Analysis

This repository contains an introductory guide to Python and its most commonly used libraries for data analysis. It is structured in a step-by-step manner to help you build foundational skills in Python and data science.

## How to Activate Conda on VSCode
To activate a Conda environment in Visual Studio Code (VSCode), follow these steps:

1. **Open the Terminal in VSCode**:
   - Go to `View > Terminal` or use the shortcut `Ctrl + ``.

2. **Activate the Conda Environment**:
   - Run the following command to activate your Conda environment:
     ```bash
     conda activate nombre_del_entorno
     ```
   - Replace `nombre_del_entorno` with the name of your environment (e.g., `myenv`).

3. **Set the Python Interpreter**:
   - Click on the Python interpreter in the bottom left corner of VSCode.
   - Select the Conda environment you just activated. It should look something like:
     ```
     Python 3.x.x ('myenv': conda)
     ```

4. **Verify the Environment**:
   - To ensure the correct environment is active, run:
     ```bash
     python --version
     ```
   - This should display the Python version associated with your Conda environment.

5. **Install Dependencies (Optional)**:
   - If you need to install additional packages, use:
     ```bash
     conda install nombre_del_paquete
     ```
   - Or, if the package is not available in Conda, use `pip`:
     ```bash
     pip install nombre_del_paquete
     ```

6. **Deactivate the Environment (Optional)**:
   - To deactivate the Conda environment and return to the base environment, run:
     ```bash
     conda deactivate
     ```
7. **Install our Requierements.txt (important)**:
   - To activate our libraries such as pandas, matplotlib, scykit-learn, we run:
  ```bash
   pip install -r requirements.txt
   ```
---

## **Table of Contents**

1. [Introduction to Python](#introduction-to-python)  
2. [Introduction to NumPy](#introduction-to-numpy)  
3. [Introduction to Pandas](#introduction-to-pandas)  
4. [Introduction to Matplotlib](#introduction-to-matplotlib)  
5. [Introduction to Scikit-learn](#introduction-to-scikit-learn)  
6. [Intro to MLflow](#intro-to-mlflow)  
7. [Dockerization](#dockerization)  
8. [How to Build and Call an API](#how-to-build-and-call-an-api)  
9. [Database Connection and MLOps Pipeline](#database-connection-and-mlops-pipeline)

---   
     
## **1. Introduction to Python**  
Basic Python programming concepts, including variables, data types, loops, conditionals, functions, and more. Designed to get you familiar with Python as a programming language.

### **Topics Covered**:
- Variables and Data Types (`int`, `float`, `str`, `list`, `dict`, etc.).
- Control Flow (`if`, `elif`, `else`).
- Loops (`for`, `while`).
- Functions and Modules.
- File Handling and Error Handling.

### **Example**:
```python
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

print(factorial(5))  # Output: 120
```
## **2. Introduction to NumPy**  
Learn the fundamentals of NumPy, the library for numerical computing. This section covers arrays, operations on arrays, and key functions that enable efficient mathematical calculations.

### **Topics Covered**:
- Creating Arrays.
- Vectorized Operations.
- Broadcasting.
- Mathematical Functions.

### **Example**:
```python
import numpy as np

matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix.sum(axis=0))  # Output: [5, 7, 9]
```
## **3. Introduction to Pandas**  
Explore how to handle, clean, and manipulate datasets using Pandas. You will learn about DataFrames, series, indexing, filtering, and data aggregation.

### **Topics Covered**:
- Creating and Manipulating DataFrames.
- Filtering and Aggregation.
- Handling Missing Data.

### **Example**:
```python
import pandas as pd

data = {'Name': ['Alice', 'Bob'], 'Age': [25, 30]}
df = pd.DataFrame(data)
print(df[df['Age'] > 25])
```
## **4. Introduction to Matplotlib**  
A guide to data visualization using Matplotlib. Learn how to create various plots and visually represent your data.

### **Topics Covered**:
- Creating Basic Plots.
- Customizing Titles, Axes, and Legends.
- Creating Advanced Charts.

### **Example**:
```python
import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [4, 5, 6]
plt.plot(x, y, label='Line')
plt.legend()
plt.show()
```
## **5. Introduction to Scikit-learn**  
Dive into machine learning with Scikit-learn. This section introduces basic concepts of supervised and unsupervised learning, model training, and evaluation using popular algorithms.

### **Topics Covered**:
- Classification and Regression.
- Unsupervised Learning.
- Model Evaluation Metrics.

### **Example**:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = [[1], [2], [3]], [0, 1, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = RandomForestClassifier()
model.fit(X_train, y_train)
print(accuracy_score(y_test, model.predict(X_test)))
```
## **6. Intro to MLflow**  
This section introduces MLflow, a platform to manage the machine learning lifecycle, including experimentation, reproducibility, and deployment.

### Install MLflow
   - To install MLflow, use pip:
     ```bash
     pip install mlflow
     ```

   ### Getting Started with MLflow
   - **Activate MLflow**:
     ```bash
     mlflow server
     ```
   - **Set the MLflow Tracking URI**:
     ```python
     import mlflow
     mlflow.set_tracking_uri("http://localhost:5000")
     ```
   - **Run an Experiment**:
     ```python
     with mlflow.start_run():
         mlflow.log_metric("key_metric", value)
         mlflow.log_param("parameter", value)
     ```
   - **Start the MLflow Tracking Server**:
     ```bash
     mlflow ui
     ```
   - This will open the MLflow Tracking UI in your web browser at `http://localhost:5000`, where you can view and compare all your experiments.

   ### Install Additional Dependencies
   - To ensure all dependencies are installed, you can use the following pip command:
     ```bash
     pip install numpy pandas scikit-learn
     ```

   ### Set Up MLflow Backend (Optional)
   - If you want to use a remote server or a database as a backend for MLflow, configure the MLflow tracking URI to point to your server:
     ```python
     mlflow.set_tracking_uri("your_server_uri")
     ```

   ### Example Usage
   - Here’s a simple example of logging a parameter and a metric in MLflow:
     ```python
     with mlflow.start_run():
         mlflow.log_param("param_name", "param_value")
         mlflow.log_metric("metric_name", 0.95)
     ```


## 7. **Dockerization**  
   This project has been dockerized to facilitate its deployment and execution in any Docker-compatible environment. Below is a detailed description of how to build and run the Docker container.

   ### How to Build and Run the Docker Container
   - **Create the `Dockerfile`**:  
     The `Dockerfile` contains the instructions to build the Docker image. Here’s an example:
     ```dockerfile
     FROM python:3.11.11-slim

     # Install required dependencies
     RUN pip install --upgrade pip && pip install mlflow scikit-learn pandas numpy

     # Copy your project files (adjust paths as necessary)
     COPY ./exported_model /model

     # Serve the ML model using MLflow
     CMD ["mlflow", "models", "serve", "-m", "/model", "-h", "0.0.0.0", "-p", "5000"]
     ```

   - **Build the Docker image**:
     ```bash
     docker build -t my-ml-model .
     ```

   - **Run the Docker container**:
     ```bash
     docker run -p 5001:5000 my-ml-model
     ```

   - **Test the Model API**:  
     Once the container is running, you can send a request to the model API:
     ```bash
     curl -X POST http://localhost:5001/invocations \
          -H "Content-Type: application/json" \
          -d '{"columns":["feature1", "feature2"], "data":[[value1, value2]]}'
     ```

## 8. **How to Build and Call an API**  
   This section demonstrates how to create a simple API using Flask and interact with it.

   ### Create the API
   - Below is an example of a simple Flask API to serve predictions from a machine learning model:
     ```python
     from flask import Flask, request, jsonify
     import joblib  # Load your trained model

     app = Flask(__name__)

     # Load the model
     model = joblib.load("path/to/your/model.pkl")

     @app.route('/predict', methods=['POST'])
     def predict():
         data = request.get_json()
         predictions = model.predict(data["inputs"])
         return jsonify({"predictions": predictions.tolist()})

     if __name__ == "__main__":
         app.run(host="0.0.0.0", port=5000)
     ```

   ### Run the API
   - Save the file as `app.py` and run it:
     ```bash
     python app.py
     ```

   ### Test the API
   - Use `curl` to send a POST request to the API:
     ```bash
     curl -X POST http://localhost:5000/predict \
          -H "Content-Type: application/json" \
          -d '{"inputs": [[5.1, 3.5, 1.4, 0.2]]}'
     ```

   - Alternatively, use Python to send the request:
     ```python
     import requests

     url = "http://localhost:5000/predict"
     data = {"inputs": [[5.1, 3.5, 1.4, 0.2]]}
     response = requests.post(url, json=data)
     print(response.json())
     ```

## 9. **Database Connection and MLOps Pipeline**  
   This section demonstrates how to connect to a database, extract data, and create a machine learning pipeline. The pipeline is saved using MLOps practices, ensuring reproducibility and scalability.

   ### Steps to Build the Pipeline
   - **Step 1:** Connect to a database using SQLAlchemy or PyMySQL.
   - **Step 2:** Extract and preprocess data (e.g., cleaning, normalization, handling missing values).
   - **Step 3:** Train a machine learning model using libraries like Scikit-learn.
   - **Step 4:** Log the trained model, parameters, and metrics to MLflow.
   - **Step 5:** Deploy the pipeline using Docker or integrate it with orchestration tools like Airflow.

   ### Example Pipeline
   ```python
   import pandas as pd
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import accuracy_score
   import mlflow
   from sqlalchemy import create_engine

   # Step 1: Connect to the database and extract data
   engine = create_engine('postgresql://username:password@localhost:5432/mydb')
   query = "SELECT * FROM loan_data"
   data = pd.read_sql(query, engine)

   # Step 2: Preprocess the data
   X = data.drop(columns=['target'])
   y = data['target']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Step 3: Train a machine learning model
   model = RandomForestClassifier()
   model.fit(X_train, y_train)
   y_pred = model.predict(X_test)

   # Step 4: Log the model and metrics with MLflow
   with mlflow.start_run():
       mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
       mlflow.sklearn.log_model(model, "loan_model")

