# 🏏 .pkl Files - Complete Visual Guide

## 📊 What Are .pkl Files?

`.pkl` = Pickle format (Python serialization)
- Binary files that save Python objects
- Models, scalers, encoders saved as .pkl
- Load them with `pickle.load()`

---

## 🎯 The 5 Critical .pkl Files

```
models/
├── cricket_win_predictor.pkl ⭐ (74 MB) - THE MODEL
├── feature_scaler.pkl          (1.4 KB) - Feature normalization
├── label_encoders.pkl          (1.2 KB) - Team/City encoding
├── feature_columns.pkl         (351 B)  - Feature order
└── model_metadata.pkl          (438 B)  - Performance metrics
```

---

## 🔄 How They Work Together

### **Flow 1: Training (Generate .pkl)**

```
dataset.csv (94k rows)
    ↓
cricket_win_predictor.py (training script)
    ↓
    ├─→ Feature Engineering (22 features)
    ├─→ Encode categorical variables
    ├─→ Scale features
    ├─→ Train 3 models
    ├─→ Select best model
    ↓
Save as .pkl files:
    ├─→ cricket_win_predictor.pkl
    ├─→ feature_scaler.pkl
    ├─→ label_encoders.pkl
    ├─→ feature_columns.pkl
    └─→ model_metadata.pkl
```

### **Flow 2: Prediction (Load .pkl)**

```
Streamlit App (app_streamlit.py)
    ↓
Load all 5 .pkl files
    ├─→ model, scaler, encoders, features, metadata
    ↓
User input (runs_left, balls_left, etc.)
    ↓
Engineer features
    ↓
Encode teams/city
    ↓
Scale features (using scaler.pkl)
    ↓
Predict (using model.pkl)
    ↓
Display probability in app
```

---

## 📋 File-by-File Breakdown

### **1. cricket_win_predictor.pkl** (74 MB)

**What it is:**
- Trained Random Forest Classifier
- 200 decision trees
- Pre-fitted on 94k+ match data

**What it does:**
```python
model = pickle.load(open('models/cricket_win_predictor.pkl', 'rb'))
prediction = model.predict(X_scaled)              # 0 = Loss, 1 = Win
probability = model.predict_proba(X_scaled)[0][1] # 0.0-1.0
```

**Performance:**
- Accuracy: 98.83%
- F1-Score: 98.63%
- ROC-AUC: 0.9993

**Size:** 74 MB (because it has 200 trees + all parameters)

---

### **2. feature_scaler.pkl** (1.4 KB)

**What it is:**
- StandardScaler object (pre-fitted)
- Learned mean and std of all 22 features

**What it does:**
```python
scaler = pickle.load(open('models/feature_scaler.pkl', 'rb'))

# Scale new features
X_scaled = scaler.transform(X_raw)

# Before scaling:   X_raw = [42, 36, 6, 7.0, 7.0, ...]  (raw values)
# After scaling:    X_scaled = [-0.23, 0.15, -0.89, ...]  (normalized)
```

**Why needed:**
- Model was trained on scaled data
- Must scale test/prediction data the same way
- Otherwise predictions are wrong!

**Scaling formula:**
```
X_scaled = (X_raw - mean) / std
```

---

### **3. label_encoders.pkl** (1.2 KB)

**What it is:**
- Dictionary of 3 LabelEncoder objects
- Converts categorical text → numbers

```python
{
    'batting_team': LabelEncoder (12 teams → 0-11),
    'bowling_team': LabelEncoder (12 teams → 0-11),
    'city': LabelEncoder (12 cities → 0-11)
}
```

**What it does:**
```python
encoders = pickle.load(open('models/label_encoders.pkl', 'rb'))

# Example
batting_encoded = encoders['batting_team'].transform(['Mumbai Indians'])
# 'Mumbai Indians' → [4]

bowling_encoded = encoders['bowling_team'].transform(['Chennai Super Kings'])
# 'Chennai Super Kings' → [1]

city_encoded = encoders['city'].transform(['Mumbai'])
# 'Mumbai' → [9]
```

**Why needed:**
- Model only understands numbers
- Team names must be converted to IDs
- Ensures same mapping (Mumbai = 4, always)

---

### **4. feature_columns.pkl** (351 B)

**What it is:**
- Python list of 22 feature names
- Exact order the model expects

```python
[
    'runs_left', 'balls_left', 'wickets_remaining', 'total_run_x',
    'crr', 'rrr', 'batting_team', 'bowling_team', 'city',
    'runs_per_ball', 'wickets_per_ball', 'run_rate_gap',
    'crr_to_rrr_ratio', 'balls_played', 'progress_percentage',
    'wickets_lost', 'wicket_loss_rate', 'high_scoring_phase',
    'death_phase', 'critical_moment', 'rrr_normalized', 'crr_normalized'
]
```

**What it does:**
```python
feature_cols = pickle.load(open('models/feature_columns.pkl', 'rb'))

# Arrange features in correct order
X = df[feature_cols]
# Ensures column order = training order
```

**Why CRITICAL:**
- If order is wrong, model gets confused
- Features must be in exact same order as training
- Example wrong: [runs_left, crr, balls_left] ❌
- Example right: [runs_left, balls_left, wickets_remaining] ✅

---

### **5. model_metadata.pkl** (438 B)

**What it is:**
- Dictionary with model performance metrics
- Version info and feature list

```python
{
    'model_name': 'Random Forest',
    'num_features': 22,
    'features': [...list of 22...],
    'best_accuracy': 0.9883,
    'best_f1': 0.9863,
    'best_roc_auc': 0.9993,
    'training_samples': 75368,
    'test_samples': 18692,
    'confusion_matrix': [[18427, 267], [342, 17658]],
    'class_distribution': {0: 35000, 1: 58368}
}
```

**What it does:**
```python
metadata = pickle.load(open('models/model_metadata.pkl', 'rb'))

print(f"Model: {metadata['model_name']}")
print(f"Accuracy: {metadata['best_accuracy']*100:.2f}%")
print(f"ROC-AUC: {metadata['best_roc_auc']:.4f}")
```

**Used for:**
- Display in Streamlit app (model stats)
- Verify model version
- Performance documentation

---

## 🔗 How They Connect

### **During Training:**

```
cricket_win_predictor.py
    ↓
    Training data: dataset.csv
    ↓
    1. Fit StandardScaler → save as feature_scaler.pkl
    2. Fit LabelEncoders → save as label_encoders.pkl
    3. Engineer 22 features → save list as feature_columns.pkl
    4. Train Random Forest → save as cricket_win_predictor.pkl
    5. Calculate metrics → save as model_metadata.pkl
```

### **During Prediction (Streamlit App):**

```
app_streamlit.py
    ↓
    Load all 5 .pkl files
    ↓
    User inputs: "Mumbai Indians vs Chennai Super Kings, Mumbai, 42 runs left, 36 balls"
    ↓
    1. Use label_encoders to convert team names → numbers
    2. Engineer 22 features from user input
    3. Arrange features using feature_columns order
    4. Scale features using feature_scaler
    5. Predict using cricket_win_predictor model
    6. Show probability: "78% Win Probability"
```

---

## 📊 Code Examples

### **Example 1: Load All 5 Files**

```python
import pickle

# Load all artifacts
with open('models/cricket_win_predictor.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('models/feature_columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

with open('models/model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

print(f"Model loaded: {metadata['model_name']}")
print(f"Accuracy: {metadata['best_f1']*100:.2f}%")
```

### **Example 2: Make Prediction**

```python
import pandas as pd

# User input
match_data = {
    'batting_team': 'Mumbai Indians',
    'bowling_team': 'Chennai Super Kings',
    'city': 'Mumbai',
    'runs_left': 42,
    'balls_left': 36,
    'wickets_remaining': 6,
    'crr': 7.0,
    'rrr': 7.0,
    'total_run_x': 160
}

# Encode categorical variables
encoded_batting = label_encoders['batting_team'].transform([match_data['batting_team']])[0]
encoded_bowling = label_encoders['bowling_team'].transform([match_data['bowling_team']])[0]
encoded_city = label_encoders['city'].transform([match_data['city']])[0]

# Create feature vector
features = {
    'runs_left': match_data['runs_left'],
    'balls_left': match_data['balls_left'],
    'wickets_remaining': match_data['wickets_remaining'],
    'crr': match_data['crr'],
    'rrr': match_data['rrr'],
    'batting_team': encoded_batting,
    'bowling_team': encoded_bowling,
    'city': encoded_city,
    # ... engineer other 13 features
}

# Create dataframe in correct order
X = pd.DataFrame([features])[feature_columns]

# Scale
X_scaled = scaler.transform(X)

# Predict
probability = model.predict_proba(X_scaled)[0][1]

print(f"Win Probability: {probability*100:.1f}%")
```

---

## ✅ Verification Checklist

After training (`python cricket_win_predictor.py`):

- [ ] `models/` folder created
- [ ] All 5 .pkl files exist:
  ```bash
  ls -lh models/
  # Output:
  # cricket_win_predictor.pkl  74M
  # feature_scaler.pkl          1.4K
  # label_encoders.pkl          1.2K
  # feature_columns.pkl         351B
  # model_metadata.pkl          438B
  ```

- [ ] Files are not zero-sized
- [ ] Model file > 50 MB (has trained weights)
- [ ] Can load without errors:
  ```python
  import pickle
  model = pickle.load(open('models/cricket_win_predictor.pkl', 'rb'))
  print("✅ Model loaded successfully!")
  ```

---

## 🚀 Integration Into App

**In `app_streamlit.py` (lines 99-114):**

```python
@st.cache_resource
def load_model_artifacts():
    """Load trained model and preprocessing artifacts"""
    try:
        # Load all 5 .pkl files
        model = pickle.load(open('models/cricket_win_predictor.pkl', 'rb'))
        scaler = pickle.load(open('models/feature_scaler.pkl', 'rb'))
        label_encoders = pickle.load(open('models/label_encoders.pkl', 'rb'))
        feature_cols = pickle.load(open('models/feature_columns.pkl', 'rb'))
        metadata = pickle.load(open('models/model_metadata.pkl', 'rb'))
        
        return model, scaler, label_encoders, feature_cols, metadata
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None
```

---

## 💾 Why Pickle?

| Format | Pros | Cons |
|--------|------|------|
| **Pickle (.pkl)** | ✅ Fast load, ✅ Preserves Python objects | ❌ Python-only |
| **JSON** | ✅ Human readable, ✅ Any language | ❌ Can't save models |
| **HDF5** | ✅ Good for large data | ❌ Complex setup |
| **ONNX** | ✅ Universal format, ✅ Any language | ❌ Extra conversion needed |

**We use pickle because:**
- Scikit-learn models save perfectly as .pkl
- Fast to load (< 1 second)
- No format conversion needed
- Standard practice for ML projects

---

## 🎓 Summary

| File | Size | Purpose | Critical? |
|------|------|---------|-----------|
| cricket_win_predictor.pkl | 74 MB | Trained model | ✅ YES |
| feature_scaler.pkl | 1.4 KB | Normalize features | ✅ YES |
| label_encoders.pkl | 1.2 KB | Encode teams/cities | ✅ YES |
| feature_columns.pkl | 351 B | Feature order | ✅ YES |
| model_metadata.pkl | 438 B | Performance metrics | ⚠️ Optional |

**All 5 needed for production deployment! 🚀**

---

## 🔍 Debugging

**If app says "Failed to load model":**

1. Check file exists:
   ```bash
   ls models/cricket_win_predictor.pkl
   ```

2. Check file size (should be > 50 MB):
   ```bash
   du -h models/cricket_win_predictor.pkl
   ```

3. Test load:
   ```python
   import pickle
   model = pickle.load(open('models/cricket_win_predictor.pkl', 'rb'))
   ```

4. Fix paths in app (use relative paths):
   ```python
   import os
   current_dir = os.path.dirname(os.path.abspath(__file__))
   model_path = os.path.join(current_dir, 'models', 'cricket_win_predictor.pkl')
   ```

---

**Ready to train and generate .pkl files? 🚀**

```bash
python create_sample_dataset.py
python cricket_win_predictor.py
```

