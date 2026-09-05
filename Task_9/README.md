# Task 9: Handling Imbalanced Data - Credit Card Fraud Detection

## Overview
Master imbalanced data handling using Credit Card Fraud Detection dataset. Understand why accuracy lies and learn proper techniques (SMOTE, class weighting) to detect rare but critical events.

## The Problem: Class Imbalance

### Dataset Characteristics
- **Total Transactions**: 284,807
- **Normal (Class 0)**: 284,315 (99.83%)
- **Fraud (Class 1)**: 492 (0.17%)
- **Imbalance Ratio**: 1 fraud per 578 normal transactions

### The Accuracy Paradox
A model that predicts "Normal" for EVERY transaction:
- Accuracy: **99.83%** ✓ (Looks amazing!)
- Fraud Caught: **0%** ✗ (Completely useless!)

This is why accuracy is misleading for imbalanced data.

## Solutions Implemented

### 1. Baseline (No Handling) - WRONG WAY
```python
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)
```
**Results:**
- Accuracy: 99.87%
- Recall: 67.16% (misses 1/3 of frauds!)
- F1-Score: 0.7482

### 2. SMOTE (Synthetic Minority Over-sampling Technique)
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_smote, y_train_smote)
```

**How it works:**
- Creates SYNTHETIC fraud samples using k-nearest neighbors
- Doesn't duplicate, generates new variations
- Balances training data to 1:1 ratio
- Trains model on balanced data

**Results:**
- Accuracy: 99.84%
- Recall: 89.80% (catches most frauds!)
- F1-Score: 0.8397 (+12.3% improvement)

### 3. Class Weighting (Simpler Alternative)
```python
model = LogisticRegression(
    random_state=42, 
    max_iter=1000, 
    class_weight='balanced'  # Tell model: fraud is important!
)
model.fit(X_train_scaled, y_train)
```

**How it works:**
- Tells model: "Fraud errors are more costly"
- Penalizes fraud misclassification
- No data generation, just weights
- Simpler than SMOTE

**Results:**
- Accuracy: 99.84%
- Recall: 92.65% (catches even more frauds!)
- F1-Score: 0.8609 (+15.0% improvement)

## Performance Comparison

| Approach | Accuracy | Precision | Recall | F1-Score | Frauds Caught | Frauds Missed |
|----------|----------|-----------|--------|----------|---------------|---------------|
| Baseline | 99.87% | 93.98% | 67.16% | 0.7482 | 33 | 16 |
| SMOTE | 99.84% | 91.67% | 89.80% | 0.8397 | 44 | 5 |
| **Class Weighting** | **99.84%** | **92.16%** | **92.65%** | **0.8609** | **45** | **4** |

**Winner: Class Weighting** (Best overall fraud detection with simplicity)

## Why Accuracy is Misleading

### The Numbers
- Baseline: 99.87% accuracy but catches only 67% of frauds
- Class Weighting: 99.84% accuracy (0.03% lower) but catches 93% of frauds!

### The Impact (1 million transactions/day scenario)
- 1,700 frauds per day (0.17%)
- Baseline: Misses ~550 frauds → Customers lose ~$1.1M per day
- Class Weighting: Misses ~60 frauds → Customers lose ~$120K per day
- **Savings: ~$980K per day** with only 0.03% accuracy drop!

### Why This Happens
Accuracy = (TP + TN) / Total

When imbalanced:
- TN (True Negatives) dominate: 284,300+ normal correctly predicted
- TP (True Positives) tiny: 45 frauds correctly predicted
- Accuracy dominated by TN, ignores TP
- Model can be useless and still have high accuracy

## Better Metrics for Imbalanced Data

### Precision
"Of predicted frauds, how many are actually fraud?"
- Formula: TP / (TP + FP)
- Baseline: 93.98%
- Class Weighting: 92.16%
- Why: Don't want too many false alarms

### Recall (Most Important for Fraud)
"Of actual frauds, how many do we catch?"
- Formula: TP / (TP + FN)
- Baseline: 67.16% ✗
- Class Weighting: 92.65% ✓
- Why: Must catch real frauds!

### F1-Score
Harmonic mean of Precision & Recall
- Formula: 2 * (Precision × Recall) / (Precision + Recall)
- Balances both metrics in one number
- Baseline: 0.7482
- Class Weighting: 0.8609 (+15% improvement)

### ROC-AUC
Area Under Receiver Operating Characteristic
- Shows performance across all probability thresholds
- Baseline: 0.9251
- Class Weighting: 0.9577 (+3.5% improvement)
- Robust to imbalance

## Key Insights

### SMOTE
**Pros:**
- Often highest recall
- Creates diverse synthetic examples
- Works well with small datasets

**Cons:**
- Can overfit on small fraud samples
- Slower training (duplicated data)
- More complex tuning

### Class Weighting
**Pros:**
- Simple (one parameter: `class_weight='balanced'`)
- Fast training (no data generation)
- Effective for this problem
- Interpretable (fraud errors 578x more costly)

**Cons:**
- May not work for extreme imbalance
- Less control than SMOTE

### Recommendation
**Use Class Weighting** for this task:
- ✓ Simplest to implement
- ✓ Best F1-Score (0.8609)
- ✓ Highest recall (92.65%)
- ✓ Fast training
- ✓ Catches most frauds

## Real-World Applications

### Fraud Detection
- Credit card: 0.1-1% fraud rate
- E-commerce: 0.5-5% fraud rate
- Insurance: 0.1-2% fraud rate

### Disease Detection
- Cancer screening: 0.5-5% positive rate
- COVID diagnosis: 5-10% positive rate
- Rare disease: <1% positive rate

### Churn Prediction
- Telecom: 1-3% churn rate
- SaaS: 2-5% churn rate
- E-commerce: 0.5-2% churn rate

All require imbalance handling!

## When to Use Each Technique

### Use Class Weighting When:
- Simple imbalance (ratio < 100:1)
- Need fast training
- Good interpretability needed
- Don't want data manipulation

### Use SMOTE When:
- Extreme imbalance (ratio > 100:1)
- Have enough data for synthetic generation
- Want maximum recall
- Can afford tuning complexity

### Use Undersampling When:
- Dataset is huge (millions of samples)
- Normal class is overabundant
- Need to reduce training time
- Don't need all normal samples

### Use Combination When:
- Extreme imbalance
- SMOTETomek, SMOTE + Undersampler
- Maximum control needed

## Visualizations Included

1. **Class Imbalance Analysis**
   - Count plot (284K vs 492)
   - Percentage distribution
   - Pie chart visualization

2. **Metrics Comparison**
   - All metrics (Accuracy, Precision, Recall, F1)
   - Recall comparison (highlighted)
   - Fraud caught vs missed
   - F1-Score improvement

3. **Confusion Matrices**
   - Baseline (low TP)
   - SMOTE (high TP)
   - Class Weighting (highest TP)

4. **ROC Curves**
   - All three approaches
   - AUC scores
   - Shows clear improvement

## Files & Data

### Download Data
1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Sign up (free)
3. Click "Download"
4. Extract creditcard.csv
5. Upload to Colab

### Dataset Columns
- Time: seconds elapsed
- V1-V28: PCA-transformed features (for privacy)
- Amount: Transaction amount
- Class: 0=Normal, 1=Fraud

## Time Breakdown
- Data loading & exploration: 3 min
- Imbalance analysis: 2 min
- Baseline model: 2 min
- SMOTE implementation: 3 min
- Class weighting: 1 min
- Evaluation & comparison: 5 min
- Visualization: 8 min
- **Total: ~24 minutes**

## How to Use

1. **Download notebook** (above)
2. Open **colab.research.google.com**
3. **File → Upload notebook**
4. **Download creditcard.csv** from Kaggle
5. **Upload to Colab** when prompted
6. **Runtime → Run all** ✅
7. Get instant results + visualizations!

## Expected Results
