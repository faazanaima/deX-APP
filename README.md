# DeX-APP: Smart Targeting for Term Deposit Marketing

DeX-APP is a machine learning solution that helps banks predict which customers are most likely to subscribe to term deposits. Using a tuned Random Forest model with precision-focused thresholding, it achieves 82% precision and reduces marketing costs by up to 67%. The app is built with Streamlit for easy deployment and interactive use, enabling data-driven, cost-effective marketing campaigns.

Kindly try this out: [https://dex-term-deposit-explorer.streamlit.app/](https://dex-term-deposit-explorer.streamlit.app/)

---

# 💼 Bank Term Deposit Prediction  
*By Faaza Naima*

---

## 🧠 Business Problem  
A U.S. bank faces high marketing costs due to ineffective telemarketing campaigns. The goal is to develop a predictive model to identify customers likely to subscribe to a term deposit, minimizing false positives to reduce wasted calls and costs.

---

## 🎯 Objective  
Build a classification model to predict whether a customer will subscribe to a term deposit using historical marketing campaign data.

---

## 📊 Dataset Overview  
- **Observations:** 7,813  
- **Features:** 11 (4 numerical, 7 categorical)  
- **Target:** `deposit` (yes/no)  
- **Missing Values:** None in raw data; special values present (e.g., -1, unknown)

---

## 💰 Cost Considerations  

| Type           | Cost   |
| -------------- | ------ |
| False Positive | $0.60  |
| False Negative | $35.00 |

- **Key Metric:** Precision  
- **Goal:** Maximize ROI by targeting only likely responders to reduce wasted marketing efforts.

---

## 🛠️ Solution Summary  
Developed a classification pipeline focused on maximizing precision to support cost-effective marketing. The approach included:

- Data preprocessing and feature engineering  
- Model selection and hyperparameter tuning  
- Evaluation with cost-sensitive metrics  

---

## 🔧 Data Preprocessing  

### Handling Missing & Special Values  
- **job:** Imputed missing with mode  
- **pdays:**  
  - 74.45% have value `-1` indicating no prior contact  
  - Created binary feature `is_firstcontact` to capture this  
- **poutcome:**  
  - 70% missing values treated as a new category `'not_exist'`  
- **contact:**  
  - Missing values imputed as `'other'`

### Data Consistency  
- Column names already in snake_case; no renaming required  
- Removed 11 duplicate records

---

## 🧪 Feature Engineering  

- Constructed `is_firstcontact` from `pdays` to handle skewed distribution  
- Introduced new categories for missing groups in `poutcome` and `contact`  
- Selected significant features via statistical tests (chi-square, Kruskal-Wallis)  
- Scaled numeric features using RobustScaler (skewed) and StandardScaler (normal)  
- Encoded categorical features with OneHotEncoder and reserved OrdinalEncoder (none used)

**Preprocessing Pipeline Snippet:**
```python
nominal_cat = ['housing', 'loan', 'is_firstcontact']
multi_cat = ['job', 'month', 'poutcome']
ord_cat = []
num_robust = ['age', 'balance', 'pdays']
num_standard = ['campaign']

preprocessor = ColumnTransformer([
    ('num_robust', RobustScaler(), num_robust),
    ('num_standard', StandardScaler(), num_standard),
    ('nom_cat', OneHotEncoder(drop='first', sparse_output=False), nominal_cat),
    ('multi_cat', OneHotEncoder(drop='first', sparse_output=False), multi_cat),
    ('ord_cat', OrdinalEncoder(), ord_cat),
])
```

# 📈 Feature Importance Insights

**Point-Biserial Correlation:**  
- campaign vs deposit: weak negative correlation, but kept

**Cramér’s V:**  
- Moderate association between *poutcome* and *deposit* → kept  
- Lower association for *contact*, but retained  
- Missing category features ('not_exist', 'other') showed significant association → retained

---

# 📊 Statistical Tests

**Chi-Square (Categorical):**  
Significant for *job*, *housing*, *loan*, *contact*, *month*, *poutcome*, *is_firstcontact*

**Kruskal-Wallis (Numeric & Categorical):**  
- *age*: not significant (p=0.9592)  
- Others (e.g., *balance*, *campaign*, *pdays*): significant

---

# 📉 Data Visualization Insights

- Adults (25–40) form the largest group in both subscribers and non-subscribers  
- Seniors (60+) show higher subscription rate  
- Jobs like management, retired, and students have higher subscription rates  
- Lower subscription rates in blue-collar, technician, and services groups  

---

# 📊 Modeling

### 📌 Preprocessing Step  
Dropped column `multi_cat__poutcome_not_exist` due to multicollinearity (VIF analysis)

### 📚 Pipeline Step  

```
pipeline = Pipeline([
    ('cleaner', cleaner),
    ('preprocessor', preprocessor),
    ('drop_columns', dropper),
    ('model', ThresholdClassifier(base_model=RandomForestClassifier(random_state=42), threshold=0.53))
])
```

### 🔍 Models Explored

1. **Logistic Regression**  
   - Interpretable linear classifier for binary outcomes  
   - Uses sigmoid function to model probability  

2. **Random Forest**  
   - Ensemble of decision trees with bootstrap sampling  
   - Handles complex nonlinearities and mixed data types  

3. **LightGBM**  
   - Gradient boosting with leaf-wise growth  
   - Fast, efficient, and handles categorical features natively  

---

### 📈 Model Performance Summary

| Model                 | Status              | Accuracy | Precision (Yes) | Recall (Yes) | F1-Score (Yes) | FP  | FN  | TP  | TN  |
|-----------------------|---------------------|----------|-----------------|--------------|----------------|-----|-----|-----|-----|
| Logistic Regression   | Before Optimization  | 0.70     | 0.71            | 0.62         | 0.66           | 186 | 287 | 462 | 626 |
| Logistic Regression   | After Optimization   | 0.71     | 0.76            | 0.57         | 0.65           | 131 | 324 | 422 | 684 |
| Random Forest         | Before Optimization  | 0.70     | 0.72            | 0.63         | 0.67           | 186 | 279 | 470 | 626 |
| Random Forest         | After Optimization   | 0.73     | 0.80            | 0.60         | 0.68           | 115 | 300 | 446 | 700 |
| LightGBM              | Before Optimization  | 0.72     | 0.77            | 0.60         | 0.67           | 132 | 303 | 446 | 680 |
| LightGBM              | After Optimization   | 0.73     | 0.79            | 0.60         | 0.68           | 124 | 296 | 453 | 688 |

---

### ✅ Model Selection

Random Forest (Optimized) was selected due to:  
- Highest precision (0.80) — critical to reduce wasted calls  
- Low false positives (115)  
- Balanced recall and F1-score  
- Better interpretability than LightGBM  

---

### 🎯 Threshold Tuning to Improve Precision

| Metric         | Threshold 0.55 | Threshold 0.53 |
|----------------|----------------|----------------|
| Accuracy       | 72.01%         | 73.09% ✅      |
| Precision (Yes)| 82.25% ✅      | 82.09%         |
| Recall (Yes)   | 52.82%         | 55.89% ✅      |
| F1-score (Yes) | 63.65%         | 66.69% ✅      |
| True Positives | 394            | 417 ✅         |
| False Negatives| 352            | 329 ✅         |

---

### 📊 Final Evaluation (Threshold = 0.53)

- **Test Accuracy:** 73.09%  
- **Train Accuracy:** 75.81% (no overfitting detected)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| no    | 0.69      | 0.89   | 0.78     | 815     |
| yes   | 0.82      | 0.56   | 0.67     | 746     |
| avg   | 0.75      | 0.73   | 0.72     | 1,561   |

**Confusion Matrix**

|                | Predicted No | Predicted Yes |
|----------------|--------------|---------------|
| **Actual No**  | 724          | 91            |
| **Actual Yes** | 329          | 417           |

---

### 📉 ROC Curve  
- AUC: 0.78 → good discrimination ability

### 📉 Precision-Recall Curve  
- Average Precision: 0.78  
- Maintains high precision across recall values — effective for imbalanced classes

---

### 🎯 Cost Analysis: With vs Without Predictive Model

|                 | Predicted Yes | Predicted No | Total |
|-----------------|---------------|--------------|-------|
| **Actual Yes**  | 417 (TP)      | 329 (FN)     | 746   |
| **Actual No**   | 91 (FP)       | 724 (TN)     | 815   |
| **Total**       | 508           | 1,053        | 1,561 |

**Cost Assumptions**  
- Cost per contact = $0.20/min × 3 minutes = $0.60  
- Lost revenue per missed subscriber = $35.00  

| Scenario      | Customers Contacted | Wasted Contacts (FP) | Wasted Marketing Cost | Total Marketing Cost | Lost Revenue (FN) | Total Estimated Cost |
|---------------|---------------------|----------------------|-----------------------|---------------------|-------------------|----------------------|
| Without Model | 1,561               | 815                  | $489.00               | $936.60             | $26,050.00        | $26,986.60           |
| With Model    | 508                 | 91                   | $54.60                | $304.80             | $11,515.00        | $11,819.40           |

---

# 💡 Conclusion

Implementing the Random Forest model reduces wasted marketing costs drastically by approximately 67%., more than halving total costs from approximately $27K to $12k, while maintaining good predictive performance. The model enables targeted telemarketing campaigns with higher efficiency and ROI by targeting the right customers.



---

# 📚 References

- Bank Marketing Dataset UCI  
- Scikit-learn documentation  
- Data Science best practices in classification and cost-sensitive learning

