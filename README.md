# Cardiovascular Disease Prediction

A machine learning project that predicts the presence of cardiovascular disease based on patient examination data, using classical ML classifiers.

> This project was completed as part of an Artificial Intelligence training program by **Corizo** (in association with IIT Bombay's Mood Indigo), completed August 5 – September 5, 2026.

## Overview

Cardiovascular disease is one of the leading causes of death worldwide. This project applies data preprocessing, exploratory data analysis, and multiple machine learning algorithms to a dataset of 70,000 patient records in order to predict whether a patient has cardiovascular disease.

## Dataset

- **Source:** Cardiovascular Disease dataset (`cardio_train.csv`)
- **Size:** 70,000 records, 13 features
- **Target variable:** `cardio` (0 = no disease, 1 = disease present)
- **Features include:** age, gender, height, weight, systolic/diastolic blood pressure (`ap_hi`, `ap_lo`), cholesterol level, glucose level, smoking, alcohol intake, and physical activity

## Project Workflow

1. **Data Preprocessing**
   - Removed the non-informative `id` column
   - Converted age from days to years
   - Removed physiologically invalid records (e.g. unrealistic blood pressure, height, and weight values)

2. **Exploratory Data Analysis & Visualization**
   - Distribution plots for age, weight, and blood pressure
   - Comparison plots of cholesterol, smoking, and physical activity against disease status
   - Correlation heatmap across all features

3. **Feature Correlation Analysis**
   - Identified which features are most strongly correlated with cardiovascular disease

4. **Model Training & Evaluation**
   Trained and compared five classification models:
   - Logistic Regression (LR)
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)
   - Decision Tree (DT)
   - Random Forest (RF)

   Each model was evaluated using accuracy, precision, recall, and F1-score.

5. **Model Selection**
   - Compared all models' accuracy
   - Selected the best-performing model as the final prediction model

## Tech Stack

- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

## How to Run

1. Clone this repository
2. Open `Cardiovascular_Disease_Prediction.ipynb` in Jupyter Notebook or Google Colab
3. Make sure `cardio_train.csv` is in the same directory (or uploaded to the Colab session)
4. Run all cells in order

## Results

Model accuracies are compared in the notebook, with the best-performing model selected as the final solution for cardiovascular disease detection.

## Author

Razan Saad Hatrash
