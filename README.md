# 📊 Customer Churn Prediction

## 📌 Project Overview

This project predicts customer churn using supervised machine learning models.

Customer churn prediction is important because losing existing customers is typically more costly than acquiring new ones. Therefore, model evaluation was aligned with business objectives rather than relying solely on accuracy.

Since the dataset is imbalanced, Precision, Recall, and F1-score were prioritized over accuracy.

---

## 📂 Dataset

- Binary Classification Problem  
- Target Variable:
  - `0` → No Churn  
  - `1` → Churn  
- Imbalanced class distribution  

### Train-Test Split
- 80% Training  
- 20% Testing  
- Stratified sampling used  

---

## ⚙️ Models Implemented

### 1️⃣ Logistic Regression (Baseline Model)

A simple and interpretable linear classifier.

### 📊 Test Performance (Class 1 – Churn)

| Metric    | Value |
|-----------|-------|
| Precision | 0.51  |
| Recall    | 0.79  |
| F1 Score  | 0.62  |
| Accuracy  | 0.76  |

### Interpretation
- Highest recall among all models  
- Best F1-score for churn class  
- Balanced precision-recall tradeoff  
- Highly interpretable  

Best suited when minimizing **False Negatives** (missing churn customers) is the priority.

---

### 2️⃣ Neural Network (Keras)

- Dense layers with ReLU activation  
- L2 regularization  
- Early stopping implemented  

**Observation:**
- Moderate performance  
- Did not outperform tree-based models on structured tabular data  

---

### 3️⃣ Random Forest

- Hyperparameters tuned (`max_depth`, `min_samples_split`, `n_estimators`)  
- High recall but very low precision  
- Over-predicted churn customers  

Not selected due to excessive False Positives.

---

### 4️⃣ XGBoost

Gradient boosting ensemble model.

### 📊 Test Performance (Class 1 – Churn)

| Metric    | Value |
|-----------|-------|
| Precision | 0.49  |
| Recall    | 0.70  |
| F1 Score  | 0.58  |
| Accuracy  | 0.81  |

### Interpretation
- Higher overall accuracy  
- Better precision control  
- More balanced predictions  
- Slightly lower recall than Logistic Regression  

Best suited when reducing **False Positives** is more important.

---

## 🎯 Evaluation Strategy

Because the dataset is imbalanced:

- Accuracy alone was not considered reliable.  
- Precision, Recall, and F1-score were prioritized.  
- Model selection was aligned with business objectives.  

---

# 🏆 Final Model Selection (Business-Driven)

## ✅ If Recall is the Priority

**Scenario:**  
Missing a churn customer is very costly, and the company wants to catch as many churn customers as possible.

**Selected Model: Logistic Regression**

**Reason:**
- Highest Recall (0.79)  
- Highest F1 Score  
- Better detection of churn customers  

---

## ✅ If Precision is the Priority

**Scenario:**  
Retention campaigns are expensive, and the company wants fewer false alarms.

**Selected Model: XGBoost**

**Reason:**
- Better precision control  
- Higher overall accuracy  
- More balanced prediction behavior  

---

## 📈 Model Comparison Summary

| Model                | Recall (Churn) | Precision (Churn) | F1 Score | Best For |
|----------------------|---------------|------------------|----------|----------|
| Logistic Regression  | 0.79          | 0.51             | 0.62     | Maximizing Recall |
| XGBoost              | 0.70          | 0.49             | 0.58     | Better Precision Control |
| Random Forest        | 0.75          | 0.28             | 0.41     | Aggressive Detection |
| Neural Network       | Moderate      | Moderate         | Moderate | Experimental Comparison |

---

## 🚀 Key Learnings

- Accuracy is misleading for imbalanced datasets.  
- Simpler models can outperform complex ones.  
- Model selection must align with business goals.  
- Tree-based boosting models work well for structured data.  
- Precision–Recall tradeoff is central in churn prediction.  

---

## 📌 Future Improvements

- Threshold tuning to adjust recall/precision tradeoff  
- Cross-validation for stronger evaluation  
- SHAP analysis for feature importance  
- Hyperparameter optimization with GridSearchCV  
- Deployment using Flask or FastAPI  

---

## 🧠 Conclusion

This project demonstrates a complete ML workflow:

- Data preprocessing  
- Multiple model comparison  
- Hyperparameter tuning  
- Imbalanced data handling  
- Business-driven model selection  

The final model choice depends on whether recall or precision is more important to the business.
