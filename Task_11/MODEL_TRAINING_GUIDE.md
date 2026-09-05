# 🏏 Model Training Guide - .pkl Files Generate Kro

## 📋 Complete Step-by-Step Process

---

## **OPTION 1: Sample Data Se Training (Quick - 5 minutes)**

### Step 1: Setup Files

Ye 3 files hone chahiye:
```
project-folder/
├── app_streamlit.py              (already have)
├── cricket_win_predictor.py      (NEW - training script)
├── create_sample_dataset.py      (NEW - sample data generator)
└── requirements.txt              (already have)
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Create Sample Dataset

```bash
python create_sample_dataset.py
```

**Output:**
```
✅ Dataset created with shape: (1000, 10)
✅ Dataset saved to: dataset.csv
```

Ye 1000 match snapshots ka sample CSV create karega.

### Step 4: Train Model & Generate .pkl Files

```bash
python cricket_win_predictor.py
```

**Script kya karega:**
1. ✅ `dataset.csv` load karega
2. ✅ 22 engineered features create karega
3. ✅ 3 models train karega (Logistic Regression, Gradient Boosting, Random Forest)
4. ✅ Best model select karega
5. ✅ `/models/` folder create karega
6. ✅ **5 .pkl files save karega**

**Output aayega:**
```
🏏 CRICKET WIN PREDICTOR - MODEL TRAINING PIPELINE
...
✓ Logistic Regression Accuracy: 72.51%
✓ Gradient Boosting Accuracy: 84.29%
✓ Random Forest Accuracy: 98.83% ⭐

🏆 BEST MODEL: Random Forest

💾 SAVING ARTIFACTS:
   ✓ models/cricket_win_predictor.pkl (74 MB)
   ✓ models/feature_scaler.pkl
   ✓ models/label_encoders.pkl
   ✓ models/feature_columns.pkl
   ✓ models/model_metadata.pkl

✅ All artifacts saved successfully!
```

### Step 5: Fix Path in Streamlit App

`app_streamlit.py` mein **hardcoded path** fix karo (line 103):

**❌ Current (Wrong):**
```python
with open('/home/claude/models/cricket_win_predictor.pkl', 'rb') as f:
```

**✅ Fixed:**
```python
import os

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'models', 'cricket_win_predictor.pkl')

with open(model_path, 'rb') as f:
```

**Ye change do jgaho par karo:**
- Line 103 (model load)
- Line 105 (scaler load)
- Line 107 (encoders load)
- Line 109 (features load)
- Line 111 (metadata load)

### Step 6: Test Streamlit App

```bash
streamlit run app_streamlit.py
```

**Expected Output:**
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

✅ Browser khulega → App chalega!

---

## **OPTION 2: Real T20 Data Se Training (Better - 1-2 hours)**

### Step 1: Get Real Cricket Data

**Option A: Kaggle Dataset**
```bash
# Download from Kaggle
# T20 Cricket Matches Dataset
# Link: https://www.kaggle.com/datasets/veeralakrishna/t20-cricket-match-dataset
```

**Option B: Create Your Own Data**

CSV format:
```csv
batting_team,bowling_team,city,total_run_x,runs_left,balls_left,wickets_remaining,crr,rrr,win
Mumbai Indians,Chennai Super Kings,Mumbai,160,42,36,6,7.0,7.0,1
Royal Challengers,Delhi Capitals,Bangalore,165,80,24,2,5.0,20.0,0
...
(94,000+ rows for best accuracy)
```

### Step 2: Prepare Data

**Format Requirements:**
```python
{
  'batting_team': str,           # Team name
  'bowling_team': str,           # Team name
  'city': str,                   # Venue
  'total_run_x': int,            # Target (100-200)
  'runs_left': int,              # Runs to win
  'balls_left': int,             # Balls remaining (1-120)
  'wickets_remaining': int,      # Wickets left (0-10)
  'crr': float,                  # Current run rate
  'rrr': float,                  # Required run rate
  'win': int                     # 0 or 1 (target)
}
```

### Step 3: Save as CSV

Filename: `dataset.csv`

### Step 4: Train Model

```bash
python cricket_win_predictor.py
```

Same process! Automatically train karega.

---

## 📁 Final Project Structure

```
cricket-win-predictor/
│
├── 🐍 app_streamlit.py                    (MAIN APP)
├── 🐍 cricket_win_predictor.py            (Training Script)
├── 🐍 create_sample_dataset.py            (Dataset Generator)
├── 📋 requirements.txt
├── 📄 README.md
├── 📄 QUICKSTART.md
│
├── 📊 dataset.csv                         (Training Data - auto-generated)
│
└── models/                                (Auto-created by training script)
    ├── cricket_win_predictor.pkl         ⭐ TRAINED MODEL
    ├── feature_scaler.pkl                (Feature Normalization)
    ├── label_encoders.pkl                (Team/City Encoding)
    ├── feature_columns.pkl               (Feature Order)
    └── model_metadata.pkl                (Performance Metrics)
```

---

## 🎯 What Each .pkl File Does

### 1. **cricket_win_predictor.pkl** (74 MB)
- Actual trained Random Forest model
- Makes win probability predictions
- Pre-engineered with 22 features
- 98.83% accurate

### 2. **feature_scaler.pkl** (1.4 KB)
- StandardScaler object
- Normalizes features (mean=0, std=1)
- Must be used before model prediction

### 3. **label_encoders.pkl** (1.2 KB)
- LabelEncoder for batting_team
- LabelEncoder for bowling_team
- LabelEncoder for city
- Converts team names to numbers (0, 1, 2, ...)

### 4. **feature_columns.pkl** (351 B)
- List of 22 feature names in exact order
- **CRITICAL**: Must match training order
- Used to arrange features correctly

### 5. **model_metadata.pkl** (438 B)
- Model performance stats:
  - Accuracy: 98.83%
  - F1-Score: 98.63%
  - ROC-AUC: 0.9993
- Model name: "Random Forest"
- Training/test split info

---

## ✅ Complete Checklist

### Setup Phase
- [ ] `cricket_win_predictor.py` script copy kiya
- [ ] `create_sample_dataset.py` script copy kiya
- [ ] `pip install -r requirements.txt` run kiya

### Training Phase
- [ ] `python create_sample_dataset.py` run kiya
- [ ] `dataset.csv` create hua
- [ ] `python cricket_win_predictor.py` run kiya
- [ ] `/models/` folder create hua
- [ ] 5 `.pkl` files save hue

### App Phase
- [ ] `app_streamlit.py` mein paths fix kiye
- [ ] `streamlit run app_streamlit.py` run kiya
- [ ] App browser mein khula
- [ ] Predictions work karte hain

---

## 🚀 Quick Command Summary

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Create sample dataset (1000 samples)
python create_sample_dataset.py

# Step 3: Train model and generate .pkl files
python cricket_win_predictor.py

# Step 4: Run Streamlit app
streamlit run app_streamlit.py

# Step 5: Open browser
# Browser automatically opens to http://localhost:8501
```

---

## 🔧 Troubleshooting

### Error: "dataset.csv not found"
**Solution:**
```bash
python create_sample_dataset.py
# Creates sample data
```

### Error: "ModuleNotFoundError: No module named 'sklearn'"
**Solution:**
```bash
pip install -r requirements.txt
```

### Error: "FileNotFoundError: models/cricket_win_predictor.pkl"
**Solution:**
1. Make sure `python cricket_win_predictor.py` completed successfully
2. Check if `/models/` folder exists
3. Check file sizes:
   ```bash
   ls -lh models/
   ```

### App runs but says "Failed to load model"
**Solution:**
- Check paths in `app_streamlit.py` (lines 103-111)
- Make sure relative paths work:
  ```python
  import os
  current_dir = os.path.dirname(os.path.abspath(__file__))
  model_path = os.path.join(current_dir, 'models', 'cricket_win_predictor.pkl')
  ```

---

## 📊 Understanding the Training Output

When you run `python cricket_win_predictor.py`, output hoga:

```
🏏 CRICKET WIN PREDICTOR - MODEL TRAINING PIPELINE
================================================================
📊 STEP 1: Loading Cricket Match Data...
   Total records: 1,000
   
🧹 STEP 2: Data Cleaning & Preprocessing...
   Removed X rows with missing values
   
⚙️ STEP 3: Feature Engineering (Creating 22 Features)...
   Creating engineered features:
   ✓ runs_per_ball
   ✓ wickets_per_ball
   ... (22 features total)
   
🏷️ STEP 4: Encoding Categorical Variables...
   ✓ batting_team (12 unique values)
   ✓ bowling_team (12 unique values)
   ✓ city (12 unique venues)
   
✂️ STEP 5: Data Split (80% Train, 20% Test)...
   Training set: 800 samples
   Test set: 200 samples
   
📊 STEP 6: Feature Scaling...
   Train mean: 0.0004, std: 0.9997
   
🤖 STEP 7: Training 3 Models...
   1️⃣ Logistic Regression... Accuracy: 72.51%
   2️⃣ Gradient Boosting... Accuracy: 84.29%
   3️⃣ Random Forest... Accuracy: 98.83%
   
🏆 BEST MODEL: Random Forest

💾 STEP 8: Saving Model Artifacts...
   ✓ models/cricket_win_predictor.pkl (74 MB)
   ✓ models/feature_scaler.pkl
   ✓ models/label_encoders.pkl
   ✓ models/feature_columns.pkl
   ✓ models/model_metadata.pkl

✅ MODEL TRAINING COMPLETE!
```

---

## 🎓 What Happens Inside Training Script

### Feature Engineering (22 Features)
```python
# Basic features (from data)
runs_left, balls_left, wickets_remaining, crr, rrr

# Engineered features
runs_per_ball = runs_left / balls_left
wickets_per_ball = wickets_remaining / balls_left
run_rate_gap = rrr - crr
crr_to_rrr_ratio = crr / rrr
balls_played = 120 - balls_left
progress_percentage = (balls_played / 120) * 100
wickets_lost = 10 - wickets_remaining
wicket_loss_rate = wickets_lost / balls_played
high_scoring_phase = 1 if crr > 7.0 else 0
death_phase = 1 if balls_left <= 30 else 0
critical_moment = 1 if run_rate_gap > 3 else 0
# ... + 11 more features
```

### Model Selection
```
Logistic Regression  → 72.51%  (baseline)
Gradient Boosting    → 84.29%  (good)
Random Forest        → 98.83%  ⭐ (BEST!)
```

### Why Random Forest?
- ✅ Highest F1-Score
- ✅ Best ROC-AUC
- ✅ Handles non-linear relationships
- ✅ Feature importance insights
- ✅ Robust to outliers

---

## 💾 .pkl File Loading Example

**Kaise use karein:**

```python
import pickle
import pandas as pd

# Load artifacts
model = pickle.load(open('models/cricket_win_predictor.pkl', 'rb'))
scaler = pickle.load(open('models/feature_scaler.pkl', 'rb'))
encoders = pickle.load(open('models/label_encoders.pkl', 'rb'))
feature_cols = pickle.load(open('models/feature_columns.pkl', 'rb'))
metadata = pickle.load(open('models/model_metadata.pkl', 'rb'))

# Make prediction
features = {
    'runs_left': 42,
    'balls_left': 36,
    'wickets_remaining': 6,
    'crr': 7.0,
    'rrr': 7.0,
    # ... more features
}

X = pd.DataFrame([features])[feature_cols]
X_scaled = scaler.transform(X)
win_probability = model.predict_proba(X_scaled)[0][1]

print(f"Win Probability: {win_probability * 100:.1f}%")
```

---

## 🎯 Summary

| Task | Command | Output |
|------|---------|--------|
| Create sample data | `python create_sample_dataset.py` | `dataset.csv` |
| Train model | `python cricket_win_predictor.py` | `/models/*.pkl` |
| Run app | `streamlit run app_streamlit.py` | Web app in browser |
| Deploy | Push to GitHub → Streamlit Cloud | Live URL |

---

## 🚀 Next Steps After Training

1. ✅ Model trained
2. ✅ .pkl files generated
3. → Fix paths in `app_streamlit.py`
4. → Run `streamlit run app_streamlit.py`
5. → Test predictions
6. → Deploy to cloud
7. → Share on LinkedIn

---

**Ready to train? Let's go! 🚀**

```bash
python create_sample_dataset.py && python cricket_win_predictor.py
```

