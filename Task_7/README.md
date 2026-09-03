
# Task 7: Production ML Pipelines

## Objective
Master industry-standard ML Pipelines  build clean, reusable, deployable code instead of messy notebooks.

## What I Did

### 1. Dataset & Feature Engineering
- Used Titanic dataset (891 passengers)
- Created 3 new features:
  - **FamilySize**: SibSp + Parch + 1 (family size indicator)
  - **Title**: Extracted from Name (Mr, Mrs, Miss, Master, Other)
  - **IsAlone**: 1 if traveling alone, 0 otherwise

### 2. Manual Approach (Old Way - Messy)
```python
# Step 1: Scale numerical
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 2: Encode categorical
encoder = OneHotEncoder()
X_train_encoded = encoder.fit_transform(X_train_cat)
X_test_encoded = encoder.transform(X_test_cat)

# Step 3: Combine (messy!)
X_train_final = np.concatenate([X_train_scaled, X_train_encoded])

# Step 4: Train
model = LogisticRegression()
model.fit(X_train_final, y_train)
```

**Problems:**
- Easy to forget steps
- Different order = different results
- Risk of data leakage
- Hard to reproduce
- Can't deploy easily

### 3. Pipeline Approach (Professional - Clean)
```python
# Define preprocessing for numerical
num_transformer = Pipeline([('scaler', StandardScaler())])

# Define preprocessing for categorical
cat_transformer = Pipeline([('onehot', OneHotEncoder())])

# Combine preprocessors
preprocessor = ColumnTransformer([
    ('num', num_transformer, numerical_cols),
    ('cat', cat_transformer, categorical_cols)
])

# Create complete pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression())
])

# Fit (preprocessing happens automatically!)
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

**Advantages:**
✅ No data leakage (fit only on training data)
✅ Consistent order (always same)
✅ Reproducible (same pipeline = same results)
✅ Deployable (save with joblib)
✅ Scalable (easy to add steps)

### 4. Pipeline Performance
- Accuracy: 82.12%
- Precision: 79.17%
- Recall: 100%
- F1-Score: 0.8871
- 5-Fold CV: Mean ~0.88

**Comparison with Manual Approach:**
- Accuracy: Identical
- Data Leakage Risk: Eliminated
- Reproducibility: Guaranteed
- Production Readiness: Yes!

### 5. Feature Engineering Impact
- FamilySize: Correlation with survival
- Title: Indicates gender/status
- IsAlone: Travel situation affects survival

Total features: 10 (vs 7 without engineering)

### 6. Saved Pipeline
- Saved with joblib: `titanic_pipeline_model.pkl`
- Backup with pickle: `titanic_pipeline_backup.pkl`
- Load anywhere: `joblib.load("titanic_pipeline_model.pkl")`
- Use on new data: `pipeline.predict(new_data)`

## Why Pipelines Matter

### 1. No Data Leakage
```python
# WRONG - Fits on ALL data then splits
scaler.fit(X_all)  # Sees test data!

# RIGHT - Fits only on training
pipeline.fit(X_train)  # Only training data
```

### 2. Consistency
Without pipeline: Team member applies steps differently
With pipeline: Same pipeline object = same results

### 3. Reproducibility
Without pipeline: Different notebooks = different results
With pipeline: Load pipeline = same results always

### 4. Deployability
Without pipeline: How to deploy 10 manual steps?
With pipeline: Deploy single .pkl file

### 5. Scalability
Without pipeline: Add feature → redo everything
With pipeline: Pipeline handles automatically

## Pipeline Architecture
