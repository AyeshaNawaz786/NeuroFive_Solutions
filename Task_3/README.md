# Task 3: Model Building - Classification with Logistic Regression

## Objective
Build my first machine learning model to predict Titanic passenger survival using Logistic Regression.

## What I Did

### 1. Data Preparation
- Loaded cleaned Titanic dataset from Task 2
- Verified no missing values
- Selected relevant features
- Removed non-predictive columns (PassengerId, Name, Ticket)

### 2. Encoding Categorical Variables
**Problem**: ML models only understand numbers, not text

**Solution**: One-Hot Encoding

- **Sex column**:
  - Male → Sex_male: 1 or 0
  - Female → Sex_male: 0 or 1

- **Embarked column**:
  - Southampton (S) → Embarked_C, Embarked_Q: 0, 0
  - Cherbourg (C) → Embarked_C, Embarked_Q: 1, 0
  - Queenstown (Q) → Embarked_C, Embarked_Q: 0, 1

**Result**: 11 numerical features ready for modeling

### 3. Train-Test Split
- **Training set**: 712 samples (80%) - model learns from this
- **Test set**: 179 samples (20%) - evaluate on unseen data
- **Reason**: Prevents overfitting (memorizing vs learning)

### 4. Logistic Regression Model

**What is Logistic Regression?**
- Classification algorithm (despite "regression" in name)
- Predicts probability between 0 and 1
- If probability > 0.5 → predict "Survived" (1)
- If probability < 0.5 → predict "Did Not Survive" (0)
- Works well for binary classification (2 categories)

**Training Process**:
1. Model learns coefficients for each feature
2. Coefficients indicate feature importance and direction
3. Positive coefficient = increases survival probability
4. Negative coefficient = decreases survival probability

### 5. Model Evaluation

#### Accuracy Score
- **Training Accuracy**: ~82%
- **Test Accuracy**: ~82%
- **Interpretation**: Model correctly predicts survival 82% of the time on new data

#### Confusion Matrix Analysis
