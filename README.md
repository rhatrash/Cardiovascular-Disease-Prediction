# Cardiovascular Disease Prediction

Machine-learning project for cardiovascular disease prediction using the `cardio_train.csv` dataset.

## Contents

- `Cardiovascular_Disease_Prediction_Solution.md` — complete project report and interpretation.
- `cardio_solution.py` — full preprocessing, EDA, visualization, and modeling pipeline.
- `cardio_fast.py` — fast reproducible model-comparison script.
- `data/cardio_train.csv` — input dataset.
- `figures/` — exploratory-analysis and model-performance plots.
- `outputs/` — correlation matrix, model metrics, confusion matrices, and saved Random Forest model.

## Main result

Random Forest achieved the best held-out accuracy (**73.96%**) and ROC-AUC (**80.45%**) among SVM, KNN, Decision Tree, Logistic Regression, and Random Forest.

## Run locally

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
python cardio_fast.py
```

The saved model is for educational use only and is not a clinical diagnostic system.
