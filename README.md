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

---

## Table of Contents
1. **Introduction to Python**  
   Basic Python programming concepts, including variables, data types, loops, conditionals, functions, and more. Designed to get you familiar with Python as a programming language.

2. **Introduction to NumPy**  
   Learn the fundamentals of NumPy, the library for numerical computing. This section covers arrays, operations on arrays, and key functions that enable efficient mathematical calculations.

3. **Introduction to Pandas**  
   Explore how to handle, clean, and manipulate datasets using Pandas. You will learn about DataFrames, series, indexing, filtering, and data aggregation.

4. **Introduction to Matplotlib**  
   A guide to data visualization using Matplotlib. You will learn how to create different types of plots like line charts, bar charts, histograms, and more to visually represent your data.

5. **Introduction to Scikit-learn**  
   Dive into machine learning with Scikit-learn. This section introduces basic concepts of supervised and unsupervised learning, model training, and evaluation using popular algorithms.

6. **Kaggle Challenge**  
   Apply your skills to a real-world dataset challenge on Kaggle. You will use the tools and techniques from the previous sections to analyze and build models for a data science competition.

7. **Dockerización**  
   This project has been dockerized to facilitate its deployment and execution in any Docker-compatible environment. Below is a brief description of how to build and run the Docker container.

   ### How to Build and Run the Docker Container
   - **Build the Docker image**:
     ```bash
     docker build -t nombre_de_tu_imagen .
     ```
   - **Run the Docker container**:
     ```bash
     docker run nombre_de_tu_imagen
     ```

   The `Dockerfile` uses a Python 3.10 slim image, installs dependencies from `requirements.txt`, and runs a simple Python script.

8. **Database Connection and MLOps Pipeline**  
   In this section, you will find a notebook that demonstrates how to connect to a database, extract data, and create a machine learning pipeline. The pipeline is saved using MLOps practices, ensuring reproducibility and scalability. This notebook covers:
   - Connecting to a database using SQLAlchemy or PyMySQL.
   - Extracting and preprocessing data.
   - Building and saving a machine learning pipeline using MLflow.
   - Deploying the pipeline for future use.

---

## Contributing
If you would like to contribute to this project, please follow these steps:
1. Fork the repository.
2. Create a new branch
