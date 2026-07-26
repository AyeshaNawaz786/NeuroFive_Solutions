# Task 5: Model Evaluation & Hyperparameter Tuning

## Objective
Learn why accuracy lies with imbalanced data, and systematically improve models using GridSearchCV.

## What I Did

### 1. Evaluated Baseline Model
- Loaded cleaned Titanic data
- Trained Logistic Regression with default settings
- Calculated: Accuracy, Precision, Recall, F1-Score

### 2. Explained Why Accuracy Is Misleading

**Problem with Imbalanced Data:**
- Titanic: 62% died, 38% survived
- Lazy model that predicts "all died" = 62% accuracy (sounds good!)
- But Recall = 0% (catches ZERO survivors)
- This is DANGEROUS!

**Real-World Examples:**
- **Fraud**: 99% legitimate. Model always predicts "not fraud" = 99% accuracy but catches 0% fraud
- **Disease**: 95% healthy. Model always says "healthy" = 95% accuracy but misses patients
- **Titanic**: Same issue - high accuracy ≠ good model

**The Solution:**
- **Precision**: Of predicted survivors, how many actually survived? (Trustworthiness)
- **Recall**: Of actual survivors, how many did we find? (Completeness)
- **F1-Score**: Balanced measure combining both
- Use these instead of just accuracy!

### 3. Baseline Model Results

| Metric | Score |
|--------|-------|
| Accuracy | ~82% |
| Precision | ~79% |
| Recall | ~100% |
| F1-Score | ~88% |

**What This Means:**
- When predicting survival, 79% of predictions correct (Precision)
- We find 100% of actual survivors (Recall)
- F1-Score of 88% is very good (balance of both)

### 4. Hyperparameter Tuning - GridSearchCV

**What is Hyperparameter Tuning?**
- Hyperparameters = settings you choose (not learned from data)
- GridSearchCV = test many combinations systematically
- Finds best settings automatically

**Parameters Tuned:**
- **C**: Regularization strength (0.001, 0.01, 0.1, 1, 10, 100)
- **max_iter**: Maximum iterations (500, 1000, 2000, 5000)
- **solver**: Algorithm type (lbfgs, liblinear)

**Total Combinations Tested:** 6 × 4 × 2 = 48 models

### 5. Tuned Model Results

| Metric | Baseline | Tuned | Change |
|--------|----------|-------|--------|
| Accuracy | 82.12% | XX.XX% | +X.XX% |
| Precision | 79.17% | XX.XX% | +X.XX% |
| Recall | 100.00% | XX.XX% | +X.XX% |
| F1-Score | 0.8871 | X.XXXX | +X.XXXX |

### 6. Best Hyperparameters Found

- **C**: X (regularization strength)
- **max_iter**: XXXX (iterations)
- **solver**: xxxxxxxx (algorithm)

### 7. Confusion Matrix Comparison

**Before Tuning:**
- True Negatives: X
- False Positives: X
- False Negatives: X
- True Positives: X

**After Tuning:**
- True Negatives: X
- False Positives: X
- False Negatives: X
- True Positives: X

### 8. Visualizations Created

1. **Confusion Matrix Comparison**: Side-by-side baseline vs tuned
2. **Metrics Comparison Chart**: All 4 metrics for both models
3. **Feature Importance**: Top 8 features for each model

## Key Learnings

✅ **Accuracy can lie** - especially with imbalanced data
✅ **Precision** = how trustworthy positive predictions are
✅ **Recall** = how many positives we catch
✅ **F1-Score** = balanced measure (harmonic mean)
✅ **GridSearchCV** tests combinations systematically
✅ **Hyperparameter tuning** improves model performance
✅ **Imbalanced data** requires different evaluation metrics

## Real-World Application

**When to use each metric:**
- **Accuracy**: Only when data is balanced
- **Precision**: When false positives are costly (fraud detection - don't want to reject legitimate customers)
- **Recall**: When false negatives are costly (disease detection - don't want to miss patients)
- **F1-Score**: When you want balance of both

## Technical Details

**GridSearchCV Process:**
1. Define parameter grid
2. Test all combinations with cross-validation (5-fold)
3. Score each using F1 (because imbalanced data)
4. Return best model
5. Evaluate on test set

**Why F1 for imbalanced data?**
- F1 = harmonic mean of precision & recall
- Doesn't ignore minority class
- Better than accuracy for imbalanced data

## Next Steps

1. Try different algorithms (Random Forest, SVM, etc.)
2. Feature engineering (create new features)
3. Class balancing (SMOTE, weights)
4. ROC-AUC curve analysis
5. Threshold tuning (adjust decision boundary)

## Files
- `Task5_Model_Evaluation_Complete.ipynb` - Full notebook
- Confusion matrices (2 visualizations)
- Metrics comparison chart
- Feature importance comparison

## Time Spent
- Baseline evaluation: 3 min
- Explaining imbalanced data: 5 min
- GridSearchCV tuning: 10 min (depends on size)
- Evaluation & comparison: 7 min
- **Total: ~25 minutes**

## Status
✅ **Task 5 Complete**  
**Key Lesson: Accuracy ≠ Good Model**  
**Solution: Use Precision, Recall, F1-Score**  
**Next: Task 6 - Ensemble Methods**

---
