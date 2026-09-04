# Task 8: Ensemble Methods - Random Forest & XGBoost

## Overview
Master ensemble learning by comparing single models against Random Forest and XGBoost on Titanic survival prediction.

## What's Included

### 1. Single Models (Baseline)
- Logistic Regression (82.12% accuracy)
- Decision Tree (82.39% accuracy)

### 2. Ensemble Models
- Random Forest (100 trees, parallel training)
- XGBoost (100 boosted trees, sequential learning)

### 3. Comprehensive Comparison
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Side-by-side performance table
- Model rankings by F1-Score
- Confusion matrices (4 models)
- Performance improvement over baseline

### 4. Feature Importance Analysis
- Top 10 features from Random Forest
- Top 10 features from XGBoost
- Side-by-side comparison visualization
- Different perspectives on feature importance

## Random Forest vs XGBoost

### Random Forest
**Method**: Parallel Bagging
- Build 100 independent decision trees
- Each tree trained on random data + random features
- Predictions via majority vote (classification)
- All trees work independently, no information sharing
- Fast training (can parallelize)

**Key Strengths**:
- Reduces overfitting through averaging
- Fast to train
- Robust to outliers
- Good baseline model
- Easy to interpret

**When to Use**:
- Medium-sized datasets
- Quick baseline needed
- Parallel processing available
- Need interpretability

### XGBoost
**Method**: Sequential Boosting
- Build 100 trees ONE AFTER ANOTHER
- Each tree learns from previous tree's mistakes
- Uses gradient descent to minimize loss
- Built-in regularization prevents overfitting
- Predictions via sequential sum

**Key Strengths**:
- Often achieves highest accuracy
- Learns from mistakes (sequential)
- Handles imbalanced data
- Automatic regularization
- Production-grade algorithm
- Wins competitions

**When to Use**:
- High accuracy critical
- Time for tuning available
- Handling imbalanced data
- Competition/production systems
- Willing to tune hyperparameters

### Quick Comparison

| Aspect | Random Forest | XGBoost |
|--------|---------------|---------|
| Training | Parallel | Sequential |
| Learning | Averaging | Boosting |
| Speed | Fast | Slower |
| Accuracy | Good | Often Better |
| Tuning | Easy | Complex |
| Regularization | Limited | Built-in |
| Competition | Common | Winning |

## Model Performance

### Comparison Table
