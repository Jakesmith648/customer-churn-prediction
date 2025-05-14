# customer-churn-prediction
Predicting customer churn using logistic regression and random forest models.
# Customer Churn Prediction

A full end-to-end machine learning project to predict customer churn using Python, logistic regression, and random forest. Includes data cleaning, EDA, feature engineering, class imbalance handling, model training, evaluation, and feature importance visualization.

## 🚀 Tools Used
- Python (pandas, sklearn, seaborn, matplotlib)
- Jupyter Notebook

## 📈 Key Learnings
- Importance of balancing recall vs. accuracy
- How tenure, charges, and contract types affect churn
- Tradeoffs in logistic regression vs. tree-based models

## 📂 Files Included
- `churn_analysis_project.ipynb`: Full project notebook
- `synthetic_customer_churn.csv`: Sample dataset (synthetic)

# 🧠 Customer Churn Prediction

This project predicts customer churn using a synthetic dataset and walks through a full data science workflow: data cleaning, exploration, feature engineering, modeling, evaluation, and interpretation — all with Python.

---

## 📊 Problem Statement

Customer churn is a key metric for subscription-based businesses. This project helps identify which customers are most likely to leave using logistic regression and random forest models.

---

## 📁 Dataset

- A synthetic churn dataset modeled after telecom/corporate account data
- Includes features like:
  - Contract type
  - Monthly/Total charges
  - Internet services
  - Demographics (gender, dependents, senior status)

---

## 🧭 Workflow Overview

### ✅ Step 1: Load and Inspect the Data
- Load data using `pandas`
- Preview the first few rows
- Check data types and missing values using `.info()` and `.isnull().sum()`

---

### ✅ Step 2: Clean the Data
- Convert `TotalCharges` to numeric if needed
- Drop rows with missing values
- Remove `customerID` (identifier, not predictive)

---

### ✅ Step 3: Explore the Data (EDA)
- Visualize churn distribution with `seaborn`
- Explore relationships between churn and:
  - Contract type
  - Monthly charges
  - Tenure
- Use charts like `countplot`, `histplot`, `groupby().mean()`

---

### ✅ Step 4: Prepare the Data for Modeling
- One-hot encode categorical features using `pd.get_dummies()`
- Separate:
  - `X` = features
  - `y` = target (`Churn`)
- Split into train/test sets (80/20)

---

### ✅ Step 5: Scale the Data
- Use `StandardScaler` to normalize feature values
- Improves model performance and convergence

---

### ✅ Step 6: Build a Logistic Regression Model
- Train a logistic regression model on the scaled data
- Evaluate accuracy, precision, recall, and F1-score
- Initially, the model performed well on accuracy but failed to detect churners due to class imbalance

---

### ✅ Step 7: Handle Class Imbalance
- Use `class_weight='balanced'` to tell the model to pay more attention to churners
- Accuracy may drop slightly, but **recall for churners improves** — a key business goal

---

### ✅ Step 8: Train a Stronger Model — Random Forest
- Random Forest performs better at catching churners
- Use `feature_importances_` to find out which variables matter most:
  - `MonthlyCharges`
  - `TotalCharges`
  - `tenure`

---

### ✅ Step 9: Interpret Feature Importance
- Visualize top features using a bar chart
- Identify actionable drivers of churn
  - High charges, short tenure = high risk
  - Fiber optic and paperless billing also contributed to churn

---

### ✅ Step 10: Publish the Project (This GitHub Repo!)
- Include:
  - `churn_analysis_project.ipynb` (full notebook)
  - `synthetic_customer_churn.csv` (optional demo data)
  - This `README.md` summary
- Add topics: `data-science`, `machine-learning`, `churn`, `portfolio-project`

---

## 📈 Key Insights

- Most churners are on **month-to-month contracts**
- **High monthly charges + short tenure** = higher risk of churn
- Logistic regression gives a simple model, but random forest is more effective for class imbalance

---

## 🛠 Tools Used

- Python (Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn)
- Jupyter Notebook
- Git + GitHub

---

## 🧩 Future Improvements

- Try SMOTE for oversampling churners
- Deploy as a Streamlit app or Power BI dashboard
- Use SHAP or LIME for better interpretability

---

## 👩‍💼 Author

Jake Smith | M.S. Applied Economics  
> “This project demonstrates my ability to use data to solve business problems, communicate results, and build interpretable models.”

