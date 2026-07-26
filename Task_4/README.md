# Task 4: Regression - Predicting House Prices

## Objective
Learn regression - predicting continuous values (prices) instead of categories.

## What I Did

### 1. Dataset
- **Source**: California Housing Dataset (sklearn)
- **Samples**: 20,640 houses
- **Target**: Median house price (in $100,000s)
- **Range**: $14,999 - $500,001

### 2. Feature Selection (5 Features)

| Feature | Reason |
|---------|--------|
| **MedInc** | Median income - strongest price indicator |
| **AveRooms** | More rooms = larger house = higher price |
| **HouseAge** | Newer homes typically cost more |
| **Latitude** | Geographic location affects value |
| **Longitude** | East-west position impacts desirability |

### 3. Model Training
- **Algorithm**: Linear Regression
- **Training Samples**: 16,512 (80%)
- **Test Samples**: 4,128 (20%)
- **Approach**: Find best-fit line through data

### 4. Model Evaluation

**Performance Metrics:**

| Metric | Training | Testing |
|--------|----------|---------|
| R² Score | 0.5757 (57.57%) | 0.5769 (57.69%) |
| RMSE | $72,400 | $73,650 |
| MAE | $53,280 | $54,090 |

### 5. Key Results

**R² Score Explanation (57.69%):**
- Model explains 57.69% of house price differences
- Remaining 42.31% due to unmeasured factors (renovations, exact location quality, school ratings)
- **In plain English**: When guessing why prices differ, our model gets about 58% of the story right
- **Practical impact**: Average prediction error is $73,650 on homes ranging up to $500K

**Feature Importance (Coefficients):**

| Feature | Price Impact |
|---------|--------------|
| MedInc | +$45,058 per unit increase |
| Latitude | +$9,620 per unit increase |
| AveRooms | -$3,785 per unit (negative? see notes) |
| HouseAge | +$977 per year |
| Longitude | -$4,285 per unit |

### 6. Visualizations Created

1. **Predicted vs Actual Scatter Plot**
   - Shows model accuracy visually
   - Points close to diagonal line = good predictions
   - Test set R² = 0.577

2. **Residual Analysis**
   - Errors scattered around zero = unbiased
   - Random pattern = good model
   - Histogram of errors shows mostly centered

3. **Feature Importance Chart**
   - Median income has strongest effect
   - Each $1,000 income increase = $45,058 price increase (approx)

4. **Error Distribution**
   - Most predictions within $75K of actual
   - Normal distribution = realistic model

### 7. Performance Assessment

**Good:**
✅ R² = 0.577 is solid for simple model
✅ Training ≈ Testing (no overfitting)
✅ Errors centered around zero
✅ Simple model, easy to understand

**Could Improve:**
⚠️ Only explains 57% of variation
⚠️ May have non-linear patterns
⚠️ Missing important features
⚠️ Some systematic errors visible

### 8. Real-World Application

If building **house price estimator**:
- Use this model for **rough estimates** (±$75K)
- Better than random guessing (explains 57% of variation)
- Need more features for **precise valuations**
- Add: school quality, crime rate, renovations, exact coordinates

## Key Learnings

✅ **Regression predicts continuous values** (not categories)
✅ **R² shows explanatory power** (0-1 scale)
✅ **RMSE measures prediction error** (in original units)
✅ **Train = Test means good generalization**
✅ **Simple models interpretable** but may underfit
✅ **Feature selection matters** - right inputs = better output

## What R² Really Means

**For non-technical person:**
"Imagine you're explaining why house prices vary. R² = 0.58 means: 'Your explanation (location, size, income) covers 58% of the reasons. The other 42% is stuff you didn't mention (renovations, street view, market timing, etc.)'"

## Next Steps

1. **Polynomial Regression** - Capture curved relationships
2. **Random Forest** - Handle non-linear patterns
3. **Feature Engineering** - Create new features (price per room, etc.)
4. **More Data** - Add crime rate, school quality
5. **Hyperparameter Tuning** - Optimize model

## Files Included
- `Task4_Regression_Professional.ipynb` - Complete notebook
- All visualizations (4 charts)
- Predictions & evaluations

## Time Spent
- Data exploration: 5 min
- Model training: 2 min  
- Evaluation: 10 min
- **Total: ~17 minutes**

## Status
✅ **Task 4 Complete**  
**Accuracy: R² = 57.69%**  
**Next: Task 5 - Advanced Models & Comparison**

---
