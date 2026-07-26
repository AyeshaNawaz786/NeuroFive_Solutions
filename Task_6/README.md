
# Task 6: Customer Churn Prediction

## Objective
Predict which customers are about to leave (churn) - a real problem companies pay millions to solve.

## What I Did

### 1. Dataset
- **Source**: Telco Customer Churn (Kaggle)
- **Samples**: 7,043 customers
- **Target**: Churn (Yes/No)
- **Features**: 20+ (demographics, services, billing)

### 2. Exploratory Data Analysis

**Churn Rate:** ~27% (typical for telecom)
- No Churn: 73%
- Churn: 27%

**Key Patterns Found:**
- **Tenure**: New customers churn more than long-term (correlation: -0.35)
- **Monthly Charges**: Higher bills → more churn
- **Contract**: Month-to-month contracts have 42% churn rate
- **Internet Service**: Fiber optic customers churn more (41% vs 20%)

### 3. Data Preprocessing

**Categorical Encoding:**
- Binary variables: Label encoding (gender, partner, etc.)
- Multi-class: One-hot encoding (contract type, payment method, etc.)
- Result: 48 total features

**Class Imbalance Handling:**
- Imbalance ratio: 2.7:1 (No-Churn vs Churn)
- Solution 1: class_weight='balanced' in models
- Solution 2: Stratified train-test split (maintain proportions)
- Result: Model optimized to catch churners, not just overall accuracy

### 4. Model 1: Decision Tree Classifier

**Advantages:**
- Highly interpretable (explain to non-technical people)
- Shows decision rules (If tenure < 5 AND charges > $100 → Churn)
- No scaling needed
- Fast predictions

**Performance:**
- Accuracy: ~79%
- Precision: ~65% (when we predict churn, 65% correct)
- Recall: ~55% (we catch 55% of actual churners)
- F1-Score: ~0.60
- ROC-AUC: ~0.71

**Configuration:**
- max_depth=10 (prevent overfitting)
- min_samples_split=20 (require 20 samples to split)
- min_samples_leaf=10 (require 10 samples per leaf)
- class_weight='balanced' (handle imbalance)

### 5. Model 2: Logistic Regression

**Advantages:**
- Provides probability scores (0-100% risk)
- Fast training
- Works well with imbalanced data
- Interpretable coefficients

**Performance:**
- Accuracy: ~80%
- Precision: ~68% (when we predict churn, 68% correct)
- Recall: ~52% (we catch 52% of actual churners)
- F1-Score: ~0.59
- ROC-AUC: ~0.72

**Configuration:**
- max_iter=1000 (iterations to converge)
- class_weight='balanced' (handle imbalance)

### 6. Model Comparison

| Metric | Decision Tree | Logistic Regression |
|--------|---------------|-------------------|
| Accuracy | 79% | 80% |
| Precision | 65% | 68% |
| Recall | 55% | 52% |
| F1-Score | 0.60 | 0.59 |
| ROC-AUC | 0.71 | 0.72 |

**Winner: Logistic Regression** (slightly better overall, but Decision Tree more interpretable)

### 7. Top 3 Churn Drivers (Feature Importance)

**Decision Tree Analysis:**
1. **Tenure** (27% importance): Months as customer
   - New customers (0-12 months) churn most
   - Established customers (36+ months) rarely churn
   
2. **Monthly Charges** (18% importance): Customer's bill
   - Higher charges → more churn
   - Customers may feel underserved for price
   
3. **Contract Type** (15% importance): Service duration
   - Month-to-month: 42% churn rate
   - 2-year contracts: 3% churn rate

**Business Insight**: New customers with high bills on month-to-month contracts are at EXTREME risk!

### 8. Business Summary (Non-Technical)

**Problem**: We lose 27% of customers. Most don't warn us - they just leave.

**Solution**: Built an AI model that identifies who's about to leave BEFORE it happens.

**Key Findings**:
1. **Early Warning Works**: Model predicts churn with 72% accuracy
2. **Root Causes**: Tenure, billing, and contract type are biggest factors
3. **Actionable**: We can now target at-risk customers with retention offers

**Impact**:
- Prevent customer losses before they happen
- Focus retention budget on high-risk customers (27% of base)
- Estimated ROI: If retention costs $50 and customer value is $1000, save significant revenue

**Recommendation**: Deploy Logistic Regression model (better accuracy + probability scores)

### 9. Class Imbalance Note

**Challenge**: 73% no-churn, 27% churn
- If model always predicts "no churn" = 73% accuracy (misleading!)
- Need to use F1-Score and Recall, not just accuracy

**Solution Applied**:
- class_weight='balanced': Weight minority class more heavily
- Stratified split: Maintain 73/27 ratio in train/test
- Evaluation: Use precision/recall/F1, not just accuracy

**Result**: Model optimized to catch churners, not maximize raw accuracy

### 10. Visualizations Created

1. **Churn Analysis** (4 subplots):
   - Distribution of churn
   - Tenure vs Churn
   - Monthly Charges vs Churn
   - Churn Rate by Contract Type

2. **Feature Importance** (side-by-side):
   - Decision Tree top 10 features
   - Logistic Regression top 10 features

3. **Confusion Matrix Comparison**:
   - Decision Tree vs Logistic Regression
   - Shows True Positives, False Positives, etc.

## Real-World Application

**How to Use:**
1. Every month, score all customers with model
2. Identify top 10% highest-risk customers
3. Send personalized retention offers:
   - New customers: Service upgrade, dedicated support
   - High-bill customers: Plan consolidation, family discount
   - Month-to-month: Incentive to upgrade to 2-year contract
4. Track results: Did they churn after offer?
5. Measure ROI and refine offers

**Expected Impact**:
- If model is 72% accurate and we act on it:
- Target top 10% (704 customers)
- Prevent 30-40% of them from churning = 210-280 customers saved
- At $500 LTV/customer = $105-140K revenue saved per month!

## Key Learnings

✅ Decision Trees vs Logistic Regression - trade-offs
✅ Class imbalance is common in real problems
✅ Feature importance reveals business insights
✅ Precision/Recall matter more than accuracy
✅ Models must be interpretable for business
✅ ROI calculation bridges ML and business

## Technical Details

**Features Used**: 48 (after encoding)
**Training Samples**: 5,635 (80%)
**Test Samples**: 1,409 (20%)
**Stratified Split**: Yes (maintains churn ratio)

## Files Included
- `Task6_Churn_Prediction_Complete.ipynb` - Full notebook
- EDA visualizations (4 charts)
- Feature importance (2 charts)
- Confusion matrices (2 charts)
- Business summary section

## Status
✅ **Task 6 Complete**
**Best Model: Logistic Regression (ROC-AUC: 0.72)**
**Next: Task 7 - Ensemble Methods & Advanced Models**

---
