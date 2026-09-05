"""
Create Sample Cricket Dataset for Model Training
This generates sample T20 match data in the correct format
"""

import pandas as pd
import numpy as np
import os

print("=" * 80)
print("🏏 CREATING SAMPLE CRICKET DATASET")
print("=" * 80)

# Set random seed for reproducibility
np.random.seed(42)

# Cricket teams
teams = [
    'Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore',
    'Delhi Capitals', 'Kolkata Knight Riders', 'Rajasthan Royals',
    'Sunrisers Hyderabad', 'Punjab Kings', 'Gujarat Titans',
    'Lucknow Super Giants', 'Rising Pune Supergiant', 'Deccan Chargers'
]

# Venues
cities = [
    'Mumbai', 'Chennai', 'Bangalore', 'Delhi', 'Kolkata', 'Jaipur',
    'Hyderabad', 'Chandigarh', 'Ahmedabad', 'Lucknow', 'Pune', 'Rajkot'
]

# Generate sample data
n_samples = 1000  # 1000 match snapshots (you'll need 94k for real training)

print(f"\n📊 Generating {n_samples:,} sample match snapshots...")

data = {
    'batting_team': np.random.choice(teams, n_samples),
    'bowling_team': np.random.choice(teams, n_samples),
    'city': np.random.choice(cities, n_samples),
    'total_run_x': np.random.randint(100, 200, n_samples),  # Target score
    'runs_left': np.random.randint(0, 150, n_samples),       # Runs remaining
    'balls_left': np.random.randint(1, 121, n_samples),      # Balls remaining (max 120)
    'wickets_remaining': np.random.randint(0, 11, n_samples), # Wickets left
    'crr': np.random.uniform(3, 12, n_samples),              # Current run rate
    'rrr': np.random.uniform(2, 15, n_samples),              # Required run rate
}

df = pd.DataFrame(data)

# Generate target (win/loss) based on cricket logic
# If team is maintaining or exceeding RRR, they're more likely to win
df['win'] = np.where(df['crr'] >= df['rrr'], 1, 0)

# Add some randomness (not all wins are guaranteed)
random_noise = np.random.random(len(df)) < 0.1
df.loc[random_noise, 'win'] = 1 - df.loc[random_noise, 'win']

print(f"✅ Dataset created with shape: {df.shape}")
print(f"✅ Win rate: {df['win'].mean()*100:.1f}%")

# Display sample
print("\n📋 Sample Data (First 5 rows):")
print(df.head())

print("\n📊 Data Types:")
print(df.dtypes)

print("\n📈 Data Summary:")
print(df.describe())

# Save to CSV
output_file = 'dataset.csv'
df.to_csv(output_file, index=False)
print(f"\n✅ Dataset saved to: {output_file}")
print(f"   File size: {os.path.getsize(output_file) / 1024:.1f} KB")

# Show column information
print("\n📋 DATASET COLUMNS:")
print("""
Required Columns:
  1. batting_team      (str)    - Team batting (e.g., 'Mumbai Indians')
  2. bowling_team      (str)    - Team bowling (e.g., 'Chennai Super Kings')
  3. city              (str)    - Match venue (e.g., 'Mumbai')
  4. total_run_x       (int)    - Target score set for chase
  5. runs_left         (int)    - Runs needed to win
  6. balls_left        (int)    - Balls remaining out of 120
  7. wickets_remaining (int)    - Wickets left (0-10)
  8. crr               (float)  - Current run rate (e.g., 6.5)
  9. rrr               (float)  - Required run rate (e.g., 7.2)
  10. win              (int)    - TARGET: 1=Win, 0=Loss

CONSTRAINTS:
  - balls_left: 1 to 120 (T20 = 120 balls = 20 overs)
  - wickets_remaining: 0 to 10
  - crr & rrr: Positive numbers (usually 0-20)
  - win: 0 or 1 (binary classification)
""")

print("\n💡 HOW TO CREATE REAL DATASET:")
print("""
This sample has 1000 rows. For better model training, you need:
  - 50,000+ match snapshots for good accuracy
  - Real T20 cricket data (IPL, BBL, CPL, etc.)
  
Sources:
  1. ESPN Cricinfo API
  2. Kaggle Cricket Datasets
  3. Official IPL/T20 statistics
  4. Cricket APIs (cricket-api.com, etc.)
  
Once you have more data:
  1. Export as CSV with same columns
  2. Run: python cricket_win_predictor.py
  3. It will train and create .pkl files
""")

print("\n" + "="*80)
print("✅ SAMPLE DATASET READY!")
print("="*80)
print(f"""
You can now run the training:
  $ python cricket_win_predictor.py
  
This will:
  1. Load dataset.csv
  2. Create 22 engineered features
  3. Train 3 models
  4. Save .pkl artifacts to /models/
  5. Display performance metrics
  
Then run the app:
  $ streamlit run app_streamlit.py
""")
