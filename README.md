#  NBA Game Prediction Project  
*A Machine Learning Approach to Predict NBA Game Outcomes (2003–2022)*  
---

## Overview
This project applies machine learning to predict NBA game outcomes (Win/Loss) using historical data from 2003–2022.  
The dataset contained over 53,000 games, including team Elo ratings, home advantage, rest days, and rolling averages of recent performance.  

The goal is to evaluate multiple models, determine which features are most predictive of winning, and interpret model behavior using SHAP (SHapley Additive exPlanations) for transparency and trust.

---

## Models Used
Four predictive models were trained and compared:

| Model | Train Accuracy | Valid Accuracy | Test Accuracy | AUC | Brier Score |
|:------|:---------------:|:---------------:|:--------------:|:----:|:-------------:|
| **Logistic Regression** | 0.627 | 0.587 | 0.594 | 0.633 | 0.238 |
| **Random Forest** | 0.732 | 0.578 | 0.596 | 0.629 | 0.239 |
| **Gradient Boosting** | 0.639 | 0.587 | 0.588 | 0.623 | 0.242 |
| **HistGradientBoosting (Tuned + Calibrated)** | 0.676 | 0.634 | 0.650 | **0.698** | 0.221 |

**Best Model:** HistGradientBoosting (Calibrated)  
**Test Accuracy:** 65% | **AUC:** 0.70 | **Brier Score:** 0.22  

---

## Methodology

### 1. Data Preparation
- Loaded data (`games.csv`) containing NBA match results and metadata  
- Removed missing values and standardized variable names  
- Engineered features such as:
  - Rolling averages of points scored and allowed (3, 5, 10 games)
  - Win streaks and rest days
  - Back-to-back game indicators (`b2b`)
  - Elo ratings (`elo_pre`, `opp_elo_pre`)

### 2. Model Training
- Split data chronologically (80% train, 10% validation, 10% test)
- Tuned hyperparameters using GridSearchCV
- Calibrated Gradient Boosting model probabilities with Platt scaling
- Evaluated using:
  - **Accuracy** – Correct predictions ratio  
  - **AUC** – Ability to rank winners correctly  
  - **Brier Score** – Probability calibration quality  

### 3. Interpretability (SHAP)

**Key insights:**
- `opp_elo_pre` and `elo_pre` were the strongest predictors of outcome.  
- Home teams consistently had a positive SHAP contribution toward winning.  
- Teams with more rest and higher recent performance metrics also had higher win probabilities.  
- Features like `month`, `b2b`, and short rolling averages had near-zero importance.  

---

##  Key Results and Visuals

### Random Forest Feature Importance

- Top Predictors: `home`, `opp_roll_pts_5`, `roll_win_10`, and `rest_days`  
- Home advantage remains one of the most consistent determinants of game outcomes.  
- Rolling averages reflect team momentum and offensive consistency.  

### Confusion Matrix (Test Set)

| Metric | Value |
|:--------|:------|
| True Wins Predicted Correctly | 1,665 |
| True Losses Predicted Correctly | 1,517 |
| False Negatives | 1,003 |
| False Positives | 1,153 |

Overall, the model achieved a good balance between **precision and recall** and demonstrated realistic win probability predictions.

---
