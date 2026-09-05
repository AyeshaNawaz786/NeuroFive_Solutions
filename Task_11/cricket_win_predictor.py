"""
Cricket Win Probability Predictor - Model Training Pipeline
Trains Random Forest model on T20 cricket data and saves all artifacts as .pkl files
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🏏 CRICKET WIN PREDICTOR - MODEL TRAINING PIPELINE")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================================================

print("\n📊 STEP 1: Loading Cricket Match Data...")

# File path - adjust as needed
DATA_PATH = 'dataset.csv'

try:
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Data loaded successfully!")
    print(f"   Total records: {len(df):,}")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"   Data shape: {df.shape}")
except FileNotFoundError:
    print(f"❌ Error: {DATA_PATH} not found!")
    print("   Please ensure dataset.csv exists in the current directory")
    exit(1)

# Display first few rows
print("\n📋 First few rows of data:")
print(df.head())

print("\n📈 Data Info:")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"\nData types:\n{df.dtypes}")

# ============================================================================
# STEP 2: DATA CLEANING & PREPROCESSING
# ============================================================================

print("\n🧹 STEP 2: Data Cleaning & Preprocessing...")

# Remove rows with missing values
df_clean = df.dropna()
print(f"✅ Removed {len(df) - len(df_clean)} rows with missing values")
print(f"   Remaining records: {len(df_clean):,}")

# Ensure target variable exists (0 = Loss, 1 = Win)
if 'win' not in df_clean.columns:
    print("❌ Error: 'win' column not found in dataset!")
    print("   Expected columns: win, batting_team, bowling_team, city, runs_left, etc.")
    exit(1)

# Target distribution
print(f"\n🎯 Target Distribution:")
print(df_clean['win'].value_counts())
print(f"   Win Rate: {df_clean['win'].mean()*100:.2f}%")

# ============================================================================
# STEP 3: FEATURE ENGINEERING (22 Features)
# ============================================================================

print("\n⚙️ STEP 3: Feature Engineering (Creating 22 Features)...")

df_features = df_clean.copy()

# Basic features (already in data)
basic_features = [
    'runs_left', 'balls_left', 'wickets_remaining', 'total_run_x', 
    'crr', 'rrr', 'batting_team', 'bowling_team', 'city'
]

# Engineered features
print("   Creating engineered features:")

# 1. Runs per ball
df_features['runs_per_ball'] = np.where(
    df_features['balls_left'] > 0,
    df_features['runs_left'] / df_features['balls_left'],
    0
)
print("   ✓ runs_per_ball")

# 2. Wickets per ball
df_features['wickets_per_ball'] = np.where(
    df_features['balls_left'] > 0,
    df_features['wickets_remaining'] / df_features['balls_left'],
    0
)
print("   ✓ wickets_per_ball")

# 3. Run rate gap (RRR - CRR)
df_features['run_rate_gap'] = df_features['rrr'] - df_features['crr']
print("   ✓ run_rate_gap")

# 4. CRR to RRR ratio
df_features['crr_to_rrr_ratio'] = np.where(
    df_features['rrr'] > 0,
    df_features['crr'] / df_features['rrr'],
    0
)
print("   ✓ crr_to_rrr_ratio")

# 5. Balls played (out of 120)
df_features['balls_played'] = 120 - df_features['balls_left']
print("   ✓ balls_played")

# 6. Match progress percentage
df_features['progress_percentage'] = (df_features['balls_played'] / 120) * 100
print("   ✓ progress_percentage")

# 7. Wickets lost
df_features['wickets_lost'] = 10 - df_features['wickets_remaining']
print("   ✓ wickets_lost")

# 8. Wicket loss rate
df_features['wicket_loss_rate'] = np.where(
    df_features['balls_played'] > 0,
    df_features['wickets_lost'] / df_features['balls_played'],
    0
)
print("   ✓ wicket_loss_rate")

# 9. High scoring phase (CRR > 7.0)
df_features['high_scoring_phase'] = (df_features['crr'] > 7.0).astype(int)
print("   ✓ high_scoring_phase")

# 10. Death phase (balls_left <= 30, last 5 overs)
df_features['death_phase'] = (df_features['balls_left'] <= 30).astype(int)
print("   ✓ death_phase")

# 11. Critical moment (run_rate_gap > 3)
df_features['critical_moment'] = (df_features['run_rate_gap'] > 3.0).astype(int)
print("   ✓ critical_moment")

# 12. RRR normalized (divide by 10)
df_features['rrr_normalized'] = df_features['rrr'] / 10
print("   ✓ rrr_normalized")

# 13. CRR normalized
df_features['crr_normalized'] = df_features['crr'] / 10
print("   ✓ crr_normalized")

# 14. Runs per wicket (runs needed per remaining wicket)
df_features['runs_per_wicket'] = np.where(
    df_features['wickets_remaining'] > 0,
    df_features['runs_left'] / df_features['wickets_remaining'],
    0
)
print("   ✓ runs_per_wicket")

# 15. Required runs per over (runs_left / overs_left)
df_features['runs_per_over'] = np.where(
    df_features['balls_left'] > 0,
    (df_features['runs_left'] / df_features['balls_left']) * 6,
    0
)
print("   ✓ runs_per_over")

# 16. Match stage (overs played)
df_features['overs_played'] = df_features['balls_played'] / 6
print("   ✓ overs_played")

# 17. Overs left
df_features['overs_left'] = df_features['balls_left'] / 6
print("   ✓ overs_left")

# 18. Runs scored so far
df_features['runs_scored'] = df_features['total_run_x'] - df_features['runs_left']
print("   ✓ runs_scored")

# 19. Run rate change needed
df_features['rr_change_needed'] = df_features['run_rate_gap'] / df_features['crr'].clip(lower=0.1)
print("   ✓ rr_change_needed")

# 20. Wicket pressure (losing wickets early?)
df_features['wicket_pressure'] = np.where(
    df_features['balls_played'] > 0,
    df_features['wickets_lost'] / (df_features['balls_played'] / 120),
    0
)
print("   ✓ wicket_pressure")

# 21. Boundary count needed
df_features['boundaries_needed'] = np.where(
    df_features['balls_left'] > 0,
    df_features['runs_left'] / 4,  # Assuming 4 runs per boundary
    0
)
print("   ✓ boundaries_needed")

# 22. Scoring rate difference
df_features['crr_rrr_diff'] = df_features['crr'] - df_features['rrr']
print("   ✓ crr_rrr_diff")

print(f"\n✅ All 22 features created successfully!")

# ============================================================================
# STEP 4: HANDLE CATEGORICAL VARIABLES (Label Encoding)
# ============================================================================

print("\n🏷️ STEP 4: Encoding Categorical Variables...")

label_encoders = {}
categorical_cols = ['batting_team', 'bowling_team', 'city']

for col in categorical_cols:
    le = LabelEncoder()
    df_features[col] = le.fit_transform(df_features[col].astype(str))
    label_encoders[col] = le
    print(f"   ✓ {col} ({len(le.classes_)} unique values)")

print(f"✅ Categorical encoding complete!")

# ============================================================================
# STEP 5: PREPARE FEATURE MATRIX & TARGET
# ============================================================================

print("\n📦 STEP 5: Preparing Feature Matrix...")

# Define all 22 features
feature_columns = [
    'runs_left', 'balls_left', 'wickets_remaining', 'total_run_x', 
    'crr', 'rrr', 'batting_team', 'bowling_team', 'city',
    'runs_per_ball', 'wickets_per_ball', 'run_rate_gap', 
    'crr_to_rrr_ratio', 'balls_played', 'progress_percentage',
    'wickets_lost', 'wicket_loss_rate', 'high_scoring_phase',
    'death_phase', 'critical_moment', 'rrr_normalized', 'crr_normalized',
    'runs_per_wicket', 'runs_per_over', 'overs_played', 'overs_left',
    'runs_scored', 'rr_change_needed', 'wicket_pressure', 
    'boundaries_needed', 'crr_rrr_diff'
]

# Remove duplicates (in case some were added twice)
feature_columns = list(set(feature_columns))

# Keep only features that exist in dataframe
feature_columns = [col for col in feature_columns if col in df_features.columns]

print(f"   Total features: {len(feature_columns)}")
print(f"   Features: {feature_columns[:10]}... (showing first 10)")

X = df_features[feature_columns]
y = df_features['win']

print(f"\n✅ Feature matrix shape: {X.shape}")
print(f"✅ Target shape: {y.shape}")
print(f"   Class distribution: {dict(y.value_counts())}")

# Check for any NaN or Inf values
print(f"\n🔍 Data quality check:")
print(f"   NaN values in X: {X.isnull().sum().sum()}")
print(f"   Infinite values in X: {np.isinf(X).sum().sum()}")

# Replace any NaN or Inf with 0
X = X.fillna(0)
X = X.replace([np.inf, -np.inf], 0)

# ============================================================================
# STEP 6: SPLIT DATA (80-20 Train-Test)
# ============================================================================

print("\n✂️ STEP 6: Splitting Data (80% Train, 20% Test)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Training set: {X_train.shape[0]:,} samples")
print(f"   Test set: {X_test.shape[0]:,} samples")
print(f"   Train class distribution: {dict(y_train.value_counts())}")
print(f"   Test class distribution: {dict(y_test.value_counts())}")

# ============================================================================
# STEP 7: FEATURE SCALING (StandardScaler)
# ============================================================================

print("\n📊 STEP 7: Feature Scaling (StandardScaler)...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Scaling complete!")
print(f"   Train mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
print(f"   Test mean: {X_test_scaled.mean():.4f}, std: {X_test_scaled.std():.4f}")

# ============================================================================
# STEP 8: TRAIN MODELS (3 Different Models)
# ============================================================================

print("\n🤖 STEP 8: Training 3 Different Models...")

results = {}

# Model 1: Logistic Regression (Baseline)
print("\n   1️⃣ Logistic Regression (Baseline)...")
lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

results['Logistic Regression'] = {
    'model': lr_model,
    'accuracy': accuracy_score(y_test, lr_pred),
    'precision': precision_score(y_test, lr_pred),
    'recall': recall_score(y_test, lr_pred),
    'f1': f1_score(y_test, lr_pred),
    'roc_auc': roc_auc_score(y_test, lr_proba)
}
print(f"      ✓ Accuracy: {results['Logistic Regression']['accuracy']*100:.2f}%")
print(f"      ✓ F1-Score: {results['Logistic Regression']['f1']*100:.2f}%")
print(f"      ✓ ROC-AUC: {results['Logistic Regression']['roc_auc']:.4f}")

# Model 2: Gradient Boosting
print("\n   2️⃣ Gradient Boosting Classifier...")
gb_model = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
)
gb_model.fit(X_train_scaled, y_train)
gb_pred = gb_model.predict(X_test_scaled)
gb_proba = gb_model.predict_proba(X_test_scaled)[:, 1]

results['Gradient Boosting'] = {
    'model': gb_model,
    'accuracy': accuracy_score(y_test, gb_pred),
    'precision': precision_score(y_test, gb_pred),
    'recall': recall_score(y_test, gb_pred),
    'f1': f1_score(y_test, gb_pred),
    'roc_auc': roc_auc_score(y_test, gb_proba)
}
print(f"      ✓ Accuracy: {results['Gradient Boosting']['accuracy']*100:.2f}%")
print(f"      ✓ F1-Score: {results['Gradient Boosting']['f1']*100:.2f}%")
print(f"      ✓ ROC-AUC: {results['Gradient Boosting']['roc_auc']:.4f}")

# Model 3: Random Forest (Usually the best)
print("\n   3️⃣ Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

results['Random Forest'] = {
    'model': rf_model,
    'accuracy': accuracy_score(y_test, rf_pred),
    'precision': precision_score(y_test, rf_pred),
    'recall': recall_score(y_test, rf_pred),
    'f1': f1_score(y_test, rf_pred),
    'roc_auc': roc_auc_score(y_test, rf_proba)
}
print(f"      ✓ Accuracy: {results['Random Forest']['accuracy']*100:.2f}%")
print(f"      ✓ F1-Score: {results['Random Forest']['f1']*100:.2f}%")
print(f"      ✓ ROC-AUC: {results['Random Forest']['roc_auc']:.4f}")

# ============================================================================
# STEP 9: MODEL COMPARISON & SELECTION
# ============================================================================

print("\n📊 STEP 9: Model Performance Comparison...")
print("\n" + "="*80)
print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<10}")
print("="*80)

best_model_name = None
best_f1 = 0

for model_name, metrics in results.items():
    print(f"{model_name:<20} {metrics['accuracy']*100:>9.2f}% {metrics['precision']*100:>10.2f}% "
          f"{metrics['recall']*100:>10.2f}% {metrics['f1']*100:>10.2f}% {metrics['roc_auc']:>8.4f}")
    
    if metrics['f1'] > best_f1:
        best_f1 = metrics['f1']
        best_model_name = model_name

print("="*80)
print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   F1-Score: {results[best_model_name]['f1']*100:.2f}%")
print(f"   Accuracy: {results[best_model_name]['accuracy']*100:.2f}%")
print(f"   ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")

best_model = results[best_model_name]['model']

# ============================================================================
# STEP 10: DETAILED EVALUATION OF BEST MODEL
# ============================================================================

print(f"\n🔍 STEP 10: Detailed Evaluation of {best_model_name}...")

print("\n📋 Classification Report:")
print(classification_report(y_test, best_model.predict(X_test_scaled), 
                          target_names=['Loss', 'Win']))

print("\n🎯 Confusion Matrix:")
cm = confusion_matrix(y_test, best_model.predict(X_test_scaled))
print(f"   True Negatives: {cm[0][0]:,}")
print(f"   False Positives: {cm[0][1]:,}")
print(f"   False Negatives: {cm[1][0]:,}")
print(f"   True Positives: {cm[1][1]:,}")

# ============================================================================
# STEP 11: CREATE MODELS DIRECTORY
# ============================================================================

print("\n📁 STEP 11: Creating Models Directory...")

models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"✅ Created directory: {models_dir}/")
else:
    print(f"✅ Directory already exists: {models_dir}/")

# ============================================================================
# STEP 12: SAVE ALL ARTIFACTS AS .PKL FILES
# ============================================================================

print("\n💾 STEP 12: Saving Model Artifacts as .pkl Files...")

# 1. Save the best trained model
model_path = os.path.join(models_dir, 'cricket_win_predictor.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"   ✓ {model_path} ({os.path.getsize(model_path) / (1024*1024):.1f} MB)")

# 2. Save the scaler
scaler_path = os.path.join(models_dir, 'feature_scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"   ✓ {scaler_path}")

# 3. Save label encoders
encoders_path = os.path.join(models_dir, 'label_encoders.pkl')
with open(encoders_path, 'wb') as f:
    pickle.dump(label_encoders, f)
print(f"   ✓ {encoders_path}")

# 4. Save feature columns (critical for consistency)
features_path = os.path.join(models_dir, 'feature_columns.pkl')
with open(features_path, 'wb') as f:
    pickle.dump(feature_columns, f)
print(f"   ✓ {features_path}")

# 5. Save model metadata
metadata = {
    'model_name': best_model_name,
    'features': feature_columns,
    'num_features': len(feature_columns),
    'best_accuracy': results[best_model_name]['accuracy'],
    'best_precision': results[best_model_name]['precision'],
    'best_recall': results[best_model_name]['recall'],
    'best_f1': results[best_model_name]['f1'],
    'best_roc_auc': results[best_model_name]['roc_auc'],
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'confusion_matrix': cm.tolist(),
    'class_distribution': dict(y.value_counts())
}

metadata_path = os.path.join(models_dir, 'model_metadata.pkl')
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)
print(f"   ✓ {metadata_path}")

print(f"\n✅ All artifacts saved successfully!")

# ============================================================================
# STEP 13: VERIFY SAVED ARTIFACTS
# ============================================================================

print("\n✔️ STEP 13: Verifying Saved Artifacts...")

files_to_check = [
    'cricket_win_predictor.pkl',
    'feature_scaler.pkl',
    'label_encoders.pkl',
    'feature_columns.pkl',
    'model_metadata.pkl'
]

for filename in files_to_check:
    filepath = os.path.join(models_dir, filename)
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        if size > 1024*1024:  # More than 1 MB
            print(f"   ✓ {filename:<30} ({size/(1024*1024):>6.1f} MB)")
        elif size > 1024:  # More than 1 KB
            print(f"   ✓ {filename:<30} ({size/1024:>6.1f} KB)")
        else:
            print(f"   ✓ {filename:<30} ({size:>6} bytes)")
    else:
        print(f"   ❌ {filename} NOT FOUND!")

# ============================================================================
# STEP 14: TEST LOADED ARTIFACTS
# ============================================================================

print("\n🧪 STEP 14: Testing Loaded Artifacts...")

# Load artifacts
with open(os.path.join(models_dir, 'cricket_win_predictor.pkl'), 'rb') as f:
    test_model = pickle.load(f)
with open(os.path.join(models_dir, 'feature_scaler.pkl'), 'rb') as f:
    test_scaler = pickle.load(f)
with open(os.path.join(models_dir, 'label_encoders.pkl'), 'rb') as f:
    test_encoders = pickle.load(f)
with open(os.path.join(models_dir, 'feature_columns.pkl'), 'rb') as f:
    test_features = pickle.load(f)

print("   ✓ Model loaded")
print("   ✓ Scaler loaded")
print("   ✓ Label encoders loaded")
print("   ✓ Feature columns loaded")

# Make a test prediction
test_pred = test_model.predict(X_test_scaled[:5])
test_proba = test_model.predict_proba(X_test_scaled[:5])[:, 1]
print(f"\n   Test prediction on 5 samples:")
for i, (pred, prob) in enumerate(zip(test_pred, test_proba)):
    result = "WIN" if pred == 1 else "LOSS"
    print(f"      Sample {i+1}: {result} ({prob*100:.1f}% confidence)")

print("\n✅ All artifacts tested successfully!")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("🎉 MODEL TRAINING COMPLETE!")
print("="*80)

print(f"""
📊 SUMMARY:
   Model: {best_model_name}
   Accuracy: {results[best_model_name]['accuracy']*100:.2f}%
   F1-Score: {results[best_model_name]['f1']*100:.2f}%
   ROC-AUC: {results[best_model_name]['roc_auc']:.4f}
   
   Training Samples: {len(X_train):,}
   Test Samples: {len(X_test):,}
   Total Features: {len(feature_columns)}
   
📁 SAVED FILES (in /models/ directory):
   ✓ cricket_win_predictor.pkl  - Trained model
   ✓ feature_scaler.pkl          - Feature normalization
   ✓ label_encoders.pkl          - Team/city encoding
   ✓ feature_columns.pkl         - Feature column order
   ✓ model_metadata.pkl          - Performance metrics

🚀 NEXT STEPS:
   1. Copy the /models/ directory to your project folder
   2. Run: streamlit run app_streamlit.py
   3. App will load and make predictions!
   
💡 To use this model in production:
   import pickle
   
   model = pickle.load(open('models/cricket_win_predictor.pkl', 'rb'))
   scaler = pickle.load(open('models/feature_scaler.pkl', 'rb'))
   
   # Scale your features and predict
   pred = model.predict_proba(scaled_features)[0][1]
""")

print("="*80)
print("✅ Training pipeline complete! Ready for deployment.")
print("="*80)
