# Diabetes Prediction Project

This repository contains a Jupyter Notebook implementation of a diabetes prediction model using machine learning techniques.

## Project Overview

The goal of this project is to build a predictive model that can accurately determine whether a person is likely to have diabetes based on certain medical attributes. The dataset used for this project is the **Pima Indians Diabetes Dataset**, which contains multiple features related to medical tests.

## Project Structure

- `notebooks/`
  - `Diabetes_Prediction.ipynb`: Jupyter Notebook containing all data analysis, data preprocessing, model building, and evaluation.
- `data/`
  - `diabetes.csv`: The dataset used in the project.
- `models/`
  - Contains the saved trained models.
- `main.py`: Script to run the prediction using the saved models.
- `README.md`: Description of the project (this file).

#Usage
1.Run the Jupyter Notebook:
    jupyter notebook notebooks/Diabetes_Prediction.ipynb

2.Execute the code cells in the notebook to see the data analysis, preprocessing, and model training steps.

3.Run the Python script: To use the trained model for predictions, you can execute the main.py file:
    python main.py

**Data**:
  The dataset used in this project can be found at data/diabetes.csv. The data contains the following features:

    Pregnancies: Number of times pregnant
    Glucose: Plasma glucose concentration
    BloodPressure: Diastolic blood pressure (mm Hg)
    SkinThickness: Triceps skinfold thickness (mm)
    Insulin: 2-Hour serum insulin (mu U/ml)
    BMI: Body mass index (weight in kg/(height in m)^2)
    DiabetesPedigreeFunction: Diabetes pedigree function (a function that scores likelihood of diabetes based on family history)
    Age: Age (years)
    Outcome: 1 if the patient has diabetes, 0 otherwise

**Model**:
  The model used in this project is a machine learning classification algorithm. Various algorithms were explored
      **Support Vector Machine (SVM)**

**Results**: 
  The performance of the model was evaluated using the following metrics:
    **Accuracy: The percentage of correct predictions.**
        The Accuracy Score of the training data :79%
        The Accuracy Score of the testing data :73%

