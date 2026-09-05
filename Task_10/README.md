# 🚢 Titanic Survival Prediction - AI Web App

## 🌐 Live App: **[Click Here to Predict!](https://titanic-survival-ai.streamlit.app)**

---

## 📸 Features & Screenshots

### ✨ Beautiful UI Components:
- 🎨 **Animated Gradients** - Eye-catching purple & blue theme
- ⚡ **Smooth Animations** - Fade-in, slide-up effects
- 📊 **Interactive Gauges** - Real-time probability visualization
- 💫 **Glassmorphism Cards** - Modern frosted glass design
- 🎯 **Responsive Layout** - Works on desktop, tablet, mobile

---

## 🎯 What This App Does

**Predicts whether a Titanic passenger would have SURVIVED!**

Enter passenger details:
- 👤 Age
- 💰 Ticket Fare
- 🎫 Passenger Class
- 👨‍👩 Gender
- 🌊 Port of Embarkation
- 👫 Family Details

Get instant AI prediction with:
- ✅/❌ Survival Result
- 📊 Probability (0-100%)
- 💎 Confidence Level
- 📈 Feature Breakdown
- 🎓 Model Stats

---

## 🏆 Model Performance

| Metric | Score |
|--------|-------|
| 🎖️ **Accuracy** | **83.74%** |
| 🔍 **Precision** | **81.82%** |
| 🎯 **Recall** | **96.49%** |
| ⚖️ **F1-Score** | **0.8889** |
| 📊 **ROC-AUC** | **0.9085** |

**Training Data**: 891 Titanic passengers
**Algorithm**: XGBoost Classifier (Gradient Boosting)

---

## 🚀 Quick Start

### Option 1: Run Locally
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/titanic-ml-app.git
cd titanic-ml-app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

# Open http://localhost:8501
```

### Option 2: Access Live App
**[🌐 Click here for live app](https://titanic-survival-ai.streamlit.app)**

---

## 📁 Project Structure

```
titanic-ml-app/
├── 📄 app.py                      # Main Streamlit application
├── 📦 titanic_model.pkl           # Trained XGBoost model
├── 📦 scaler.pkl                  # Feature scaling
├── 📦 le_sex.pkl                  # Gender encoder
├── 📦 le_embarked.pkl             # Port encoder
├── 📦 le_title.pkl                # Title encoder
├── 📄 requirements.txt             # Dependencies
└── 📄 README.md                    # This file
```

---

## 🎨 UI/UX Highlights

### Beautiful Animations
- ✨ Fade-in animations on page load
- 🎯 Slide-up card animations
- 💫 Pulsing prediction boxes
- ⚡ Smooth hover effects on buttons
- 📊 Real-time gauge updates

### Color Scheme
- 🟣 Purple Gradient: `#667eea` → `#764ba2`
- 🔵 Blue Accents: `#00d4ff` → `#0099ff`
- 🟢 Success Green: `#11998e` → `#38ef7d`
- 🔴 Danger Red: `#eb3349` → `#f45c43`
- ⚪ Glass Effect: Frosted glass cards with blur

### Interactive Elements
- 🔢 Number sliders with smooth input
- 📋 Dropdown selectors with emojis
- 🔘 Large prediction button with hover effects
- 📊 Plotly gauges and charts
- 📈 Data tables with styling

---

## 🔑 Key Features

### 1. **Real-Time Predictions**
   - Submit passenger info
   - Get instant survival prediction
   - See probability percentage

### 2. **Visual Analytics**
   - Probability gauge (0-100%)
   - Feature importance breakdown
   - Confidence level indicator
   - Statistical insights

### 3. **Model Information**
   - Model accuracy stats
   - Feature explanations
   - Training data details
   - Algorithm description

### 4. **User Experience**
   - Beautiful gradient backgrounds
   - Smooth animations
   - Responsive design
   - Error handling
   - Loading indicators

---

## 📊 How Features Impact Survival

### 1. **Gender** 👫 (Most Important!)
   - Female passengers: ✅ Much higher survival
   - "Women and children first" policy visible in data

### 2. **Ticket Fare** 💰
   - Higher fare = Better accommodations = Better survival
   - 1st class passengers paid more

### 3. **Age** 👶
   - Younger passengers had better chances
   - Children prioritized for lifeboats

### 4. **Passenger Class** 🎫
   - 1st Class: ~63% survival rate
   - 2nd Class: ~47% survival rate
   - 3rd Class: ~24% survival rate

### 5. **Family Size** 👨‍👧‍👦
   - Traveling alone: Worse chances
   - Small family: Better chances
   - Very large family: Worse chances

---

## 🛠️ Technologies Used

### Backend & ML
- **Python 3.9+**
- **XGBoost** - Gradient Boosting Classifier
- **scikit-learn** - Data preprocessing
- **joblib** - Model serialization
- **pandas** - Data manipulation
- **numpy** - Numerical computing

### Frontend
- **Streamlit** - Web framework
- **Plotly** - Interactive visualizations
- **HTML/CSS** - Custom styling & animations

### Deployment
- **Streamlit Community Cloud** - Free hosting
- **GitHub** - Code repository

---

## 🚀 Deployment Instructions

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Add Titanic ML web app"
git push origin main
```

### Step 2: Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "Deploy an app"
3. Select your GitHub repository
4. Set main file to `app.py`
5. Click "Deploy!"

**Your app will be live in 2-3 minutes!**

### Step 3: Share Your Live Link
```
https://[your-username]-titanic-ml-app.streamlit.app
```

---

## 🎓 What I Learned

✅ End-to-end ML project (EDA → Deployment)
✅ Feature engineering & preprocessing
✅ Model training & evaluation
✅ Streamlit app development
✅ Cloud deployment
✅ Professional UI/UX design
✅ Animation & styling with CSS
✅ Production-ready code

---

## 📈 Model Development

### Task 1-3: Data & Baseline Models
- EDA on Titanic dataset
- Data cleaning & feature engineering
- Logistic Regression classifier

### Task 4-6: Advanced Models
- Linear regression
- Model evaluation
- Customer churn prediction

### Task 7-9: Production Skills
- ML Pipelines
- Ensemble methods (XGBoost won!)
- Imbalanced data handling

### Task 10: Deployment (This Project!)
- Streamlit web app
- Beautiful UI/animations
- Cloud deployment
- Live prediction system

---

## 💻 Example Input & Output

### Input:
```
Age: 25 years
Fare: $150
Class: 1st Class
Gender: Female
Port: Southampton
Siblings: 1
Parents: 0
```

### Output:
```
✅ SURVIVED
89% Probability
Confidence: HIGH
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **App won't load** | Check `requirements.txt` syntax |
| **Model not found** | Ensure all `.pkl` files in same directory |
| **Slow performance** | Clear browser cache, refresh page |
| **Wrong predictions** | Check input ranges (realistic values) |

---

## 📱 Live App Features

### Desktop Version
- Full width layout
- Side-by-side columns
- Large visualizations
- Optimal for testing

### Mobile Version
- Responsive single column
- Stacked cards
- Optimized for small screens
- Full functionality

---

## 🎯 How to Use

1. **Visit** the live app link
2. **Fill in** passenger information
3. **Click** "PREDICT SURVIVAL" button
4. **View** results and analysis
5. **Explore** feature importance
6. **Try** different scenarios

---

## 📊 Performance Comparison

| Model | Accuracy | F1-Score | Used |
|-------|----------|----------|------|
| Logistic Regression | 82.12% | 0.887 | ❌ |
| Decision Tree | 82.39% | 0.883 | ❌ |
| Random Forest | 82.92% | 0.879 | ❌ |
| **XGBoost** | **83.74%** | **0.889** | ✅ |

**Winner**: XGBoost (Best accuracy & F1-score)

---

## 🌟 Highlights

### Why This Project Stands Out

✨ **Beautiful UI** - Professional animations & gradients
⚡ **Fast Performance** - Cached model loading
🎯 **Accurate Model** - 83.74% accuracy
🚀 **Live Deployment** - Real working app
📱 **Responsive Design** - Works on all devices
🎓 **Educational** - Learn full ML pipeline

---

## 🎬 Demo Video

Record a video showing:
1. App loading with animations
2. Filling in passenger data
3. Getting prediction
4. Viewing results
5. Showing model accuracy

---

## 👨‍💼 Connect & Share

- **Live App**: [titanic-ml-app.streamlit.app](https://titanic-ml-app.streamlit.app)
- **GitHub**: [Your Repository](https://github.com/YOUR_USERNAME/titanic-ml-app)
- **LinkedIn**: [Your Profile](https://linkedin.com/in/YOUR_PROFILE)

---

## 📝 License

MIT License - Free to use and modify

---

## 🎉 Summary

This is a **production-ready ML web app** that shows:
- 🎓 Full ML pipeline knowledge
- 💼 Professional development skills
- 🎨 UI/UX design abilities
- 🚀 Cloud deployment expertise

From notebook to deployed product! 🚀

---

**Built with ❤️ using Streamlit | Model: XGBoost | Data: Titanic**

**@NeurofiveSolutions ML Learning Platform**

---

**[🌐 Try the Live App Now!](https://titanic-survival-ai.streamlit.app)**

