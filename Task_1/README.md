# Titanic EDA - Exploratory Data Analysis

## Overview
This notebook contains my first exploratory data analysis (EDA) on the Titanic dataset. 
I analyzed the data to understand patterns, missing values, and survival factors.

## Dataset Information
- **Total Rows**: 891 passengers
- **Total Columns**: 12 features
- **Target Variable**: Survived (38% survival rate)

## What I Found

### Missing Data
- **Age**: 177 missing values (19.87%)
- **Cabin**: 687 missing values (77.10%)
- **Embarked**: 2 missing values (0.22%)

### Column Types
- **Numerical Columns**: PassengerId, Pclass, Age, SibSp, Parch, Fare
- **Categorical Columns**: Name, Sex, Ticket, Cabin, Embarked

### Key Survival Patterns
1. **By Gender**:
   - Females: 74% survival rate
   - Males: 19% survival rate

2. **By Passenger Class**:
   - 1st Class: 63% survival rate
   - 2nd Class: 47% survival rate
   - 3rd Class: 24% survival rate

3. **By Age**: Younger passengers had better survival chances

## Data Quality
- No duplicate rows found
- Data is ready for cleaning and feature engineering

## Libraries Used
- pandas (data manipulation)
- numpy (numerical operations)
- matplotlib (visualization)

## Next Steps
1. Handle missing values
2. Feature engineering
3. Build machine learning models
4. Model evaluation

## Files in This Folder
- `Titanic_EDA_Complete.ipynb` - Main notebook with analysis
- `titanic.csv` - Dataset file
- `README.md` - This file
