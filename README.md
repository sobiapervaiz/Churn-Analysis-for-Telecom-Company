# Churn-Analysis
Churn Analysis for Telecom Company using Python Machine Learning Models including CatBoost, XGBoost, Logistic Regression Models for Classification.



## Key Results & Business Impact
- **Model Performance:** Achieved **0.6756 PR-AUC**  using CatBoost, outperforming AdaBoost and Logistic Regression with class weights and regualrization L1 by 0.9%.
- **Top Insights:** Identified that month-to-month contracts and dependents were the strongest predictors of churn.
- **Actionable Recommendation:** Targeting the highest-risk customers with retention offers could save customers from leaving the company.

## Visual Highlights



<img width="762" height="934" alt="download" src="https://github.com/user-attachments/assets/14f9e57c-131b-49c3-8512-164a1c36158b" />



## Dataset & Methodology
- **Data Source:** [Telco customer churn: IBM dataset]https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset?resource=download — 7043 rows, 33 features (target variable is Churn Value, imabalnced data with 1869 Churn records available).
- **Data Preprocessing:** Handled missing values via median imputation, applied one-hot encoding to categorical features, and used stratificat_on to preserve imabalnce.
- **Data Pipeline:** A data pipeloine is created to automatically handle the outliers, missing values and duplicated rows followed by encoding and scaling for the dataset as per the model.
- **Models Evaluated:** Logistic Regression (Baseline), Random Forest, Support Vector Machines (SVM), XGBoost, CatBoost.


## Project Structure
```text
├── data/               # Raw dataset from Kaggle
├── notebooks/          # Exploratory data analysis (EDA) and experimentation
├── src/                # Clean, modular Python scripts for preprocessing and training
├── models/             # Saved model artifacts
└── README.md
```

## Tech Stack
- **Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, RandomForest (Ensemble), Support Vector Machines, Matplotlib/Seaborn





## Contact
- **Sobia Pervaiz** — [LinkedIn Profile]https://www.linkedin.com/in/sobia-pervaiz-071848215/ | 
