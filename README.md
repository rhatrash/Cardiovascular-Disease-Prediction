Cardiovascular Disease Prediction
> This project was developed as part of an Artificial Intelligence training course organized by **Corizo** in association with **IIT Bombay's Mood Indigo**. The training was completed from **5 August 2026 to 5 September 2026**.
Machine-learning project for cardiovascular disease prediction using the `cardio_train.csv` dataset.
Contents
`Cardiovascular_Disease_Prediction_Solution.md` — complete project report and interpretation.
`cardio_solution.py` — full preprocessing, EDA, visualization, and modeling pipeline.
`cardio_fast.py` — fast reproducible model-comparison script.
`data/cardio_train.csv` — input dataset.
`figures/` — exploratory-analysis and model-performance plots.
`outputs/` — correlation matrix, model metrics, confusion matrices, and saved Random Forest model.
Main result
Random Forest achieved the best held-out accuracy (73.96%) and ROC-AUC (80.45%) among SVM, KNN, Decision Tree, Logistic Regression, and Random Forest.
Project objective
The project applies data preprocessing, exploratory data analysis, visualization, correlation analysis, and supervised machine-learning classification to predict the presence of cardiovascular disease.
Run locally
```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
python cardio_fast.py
```
The saved model is for educational use only and is not a clinical diagnostic system.
Training
This project was completed during the following training program:
Program: Artificial Intelligence Training
Organization: Corizo, in association with IIT Bombay's Mood Indigo
Training period: 5 August 2026 – 5 September 2026
Participant: Razan Saad Hatrash
