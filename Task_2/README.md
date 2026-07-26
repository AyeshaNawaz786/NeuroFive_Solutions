# Task 2: Data Cleaning & Visualization

## Objective
Learn how to handle missing values, detect outliers, and create meaningful visualizations 
that reveal patterns before building machine learning models.

## What I Did

### 1. Missing Value Analysis & Handling

**Dataset had missing values in 3 columns:**

- **Age**: 177 missing (19.87%)
  - Decision: Fill with MEDIAN
  - Reason: Age is numerical. Median is robust against outliers
  - Keeps all passenger data without losing information

- **Cabin**: 687 missing (77.10%)
  - Decision: DROP entire column
  - Reason: Too sparse (only 214 values out of 891)
  - Not reliable for prediction

- **Embarked**: 2 missing (0.22%)
  - Decision: Fill with MODE (most common value)
  - Reason: Only 2 values missing. Southampton (S) is most frequent
  - Safe assumption for these 2 records

**Result**: Zero missing values remaining, dataset clean and ready for analysis

### 2. Outlier Detection

Used boxplots to identify outliers using IQR method:
- Lower Bound = Q1 - 1.5 × IQR
- Upper Bound = Q3 + 1.5 × IQR

**Fare column findings:**
- Found 108 outliers (12.12% of data)
- All outliers are HIGH fares (1st class expensive tickets)
- Decision: KEEP these outliers (valid data, not errors)
- Wealthy passengers paid 7-15x more for 1st class tickets

### 3. Four Required Visualizations

#### Visualization 1: Histogram
- Shows Age distribution of all passengers
- Overlaid: Age distribution by survival status
- **Key Insight**: Children (0-15) heavily concentrated in "Survived" group
- Most passengers aged 20-40
- Clear pattern: Younger = Higher survival chance

#### Visualization 2: Boxplot
- **Left**: Fare by Passenger Class
  - 1st Class median: £55
  - 2nd Class median: £15
  - 3rd Class median: £8
  - Rich passengers paid much more

- **Right**: Age by Survival Status
  - Survived group: Younger median age
  - Did NOT Survive: Older median age
  - Age distribution differs significantly

#### Visualization 3: Bar Charts (4 charts)
- **Chart 1 - Survival by Gender**:
  - Female: 74% survival
  - Male: 19% survival
  - Difference: 55 percentage points (HUGE!)

- **Chart 2 - Survival by Passenger Class**:
  - 1st Class: 63% survival
  - 2nd Class: 47% survival
  - 3rd Class: 24% survival
  - Wealth clearly mattered

- **Chart 3 - Survival by Embarked Port**:
  - Cherbourg: 55% survival
  - Queenstown: 39% survival
  - Southampton: 38% survival
  - Minor effect (2-7% variation)

- **Chart 4 - Passenger Count by Class**:
  - 1st Class: 216 passengers (24%)
  - 2nd Class: 184 passengers (21%)
  - 3rd Class: 491 passengers (55%)
  - Most passengers in lower class

#### Visualization 4: Correlation Heatmap
Shows correlation between all numerical features and survival:

**Top Correlations with Survival:**
1. **Pclass**: -0.338 (negative = lower class number = higher survival)
2. **Fare**: +0.257 (positive = higher fare = higher survival)
3. **Age**: -0.077 (negative = younger = higher survival)
4. **SibSp**: -0.035 (weak - family size minimal impact)
5. **Parch**: +0.082 (weak - parents/children minimal impact)

## Key Finding: Which Feature Most Affects Survival?

### Answer: GENDER (Sex) is the most important feature

**Evidence:**

1. **Largest Difference**: 
   - Female: 74% survival
   - Male: 19% survival
   - Gap: 55 percentage points (largest of any feature)

2. **Consistent Across All Categories**:
   - Females survived at high rates in ALL passenger classes
   - Males had low survival in ALL passenger classes
   - Gender effect overrides other factors

3. **Historical Context**:
   - Titanic followed "Women and Children First" protocol
   - Crew prioritized female passenger evacuation
   - Explains the dramatic survival difference

4. **Visual Confirmation**:
   - Bar chart shows the biggest separation
   - Heatmap confirms strong correlation pattern
   - Histogram shows clear age-stratified survival

### Secondary Important Features (in order):

**2nd Place: Passenger Class**
- 1st Class: 63% survival
- 3rd Class: 24% survival
- Difference: 39 percentage points
- Wealthier = better access to lifeboats

**3rd Place: Age**
- Children (0-15): Very high survival
- Adults (20-60): Lower survival
- Elderly (60+): Lower survival
- Children were prioritized in lifeboats

**4th Place: Fare**
- Higher fare correlates with survival
- But highly linked to Pclass (not independent)
- Rich passengers = 1st class = high fare

### Features with Minimal Impact:
- **SibSp** (Siblings/Spouse): Weak correlation
- **Parch** (Parents/Children): Weak correlation  
- **Embarked**: Only 2-7% variation in survival
- Family size had little influence

### Conclusion:
If I had to predict survival with ONE feature, I would choose **GENDER** 
because it shows the clearest, most dramatic, most consistent relationship with survival 
across the entire dataset. Gender alone is a better predictor than any other single feature.

## Libraries Used
- **pandas**: Data manipulation (fillna, dropna, groupby)
- **numpy**: Numerical operations
- **matplotlib**: Basic visualizations (histograms, bar charts)
- **seaborn**: Advanced visualizations (heatmaps, boxplots)

## Technical Skills Demonstrated
- ✓ Handling missing values with domain reasoning
- ✓ Choosing appropriate strategies (median vs mode vs drop)
- ✓ Outlier detection using IQR method
- ✓ Creating multiple visualization types
- ✓ Interpreting correlation values
- ✓ Data-driven storytelling
- ✓ Statistical analysis

## Data Quality Improvements
- Before: 891 rows × 12 columns (with missing data)
- After: 891 rows × 11 columns (Cabin dropped, no missing values)
- Completeness: 100% (all cells filled or removed)
- No duplicate records found

## Insights for Next Steps (Feature Engineering)

1. **Gender is critical feature** - will be most predictive in model
2. **Class matters** - keep as separate feature
3. **Age is important** - may need binning (child/adult/elderly)
4. **Fare has outliers** - may need scaling
5. **Family size weak** - consider removing or combining
6. **Embarked has little impact** - consider removing

## Time Spent
- Analysis: 30 minutes
- Visualization: 20 minutes
- Interpretation: 15 minutes
- Total: ~1 hour

## Next Tasks
- **Task 3**: Feature Engineering (create new features, handle categorical data)
- **Task 4**: Model Building (train Logistic Regression, Random Forest)
- **Task 5**: Model Evaluation (accuracy, precision, recall, F1 score)

## Lessons Learned

1. **Data Cleaning is Critical**: 77% missing in Cabin showed importance of cleaning strategy
2. **Visualizations > Raw Numbers**: Bar charts immediately showed gender's importance
3. **Domain Knowledge Matters**: Understanding Titanic history helped interpret data
4. **Not All Outliers are Bad**: High fares were valid data, not errors
5. **One Feature ≠ Full Story**: Gender is important but class/age also matter
6. **Different Data Types = Different Strategies**: 
   - Numerical (Age) → Median
   - Categorical (Embarked) → Mode
   - Sparse (Cabin) → Drop

## GitHub Commit
```bash
git commit -m "Task 2: Data Cleaning & Visualization - Handle missing values, detect outliers, create 4 visualizations, identify gender as key survival factor"
```

---

**Status**: ✅ Task 2 Complete  
**Date Completed**: July 2026  
**Next**: Task 3 - Feature Engineering
