import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="🚢 Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS & ANIMATIONS
# ============================================
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Title styling */
    .title-main {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00d4ff, #0099ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 0.5rem;
        animation: fadeInDown 0.8s ease-out;
    }
    
    .subtitle-main {
        font-size: 1.3rem;
        color: #fff;
        font-weight: 500;
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* Card styling */
    .card-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin: 20px 0;
        animation: slideInUp 0.6s ease-out;
    }
    
    .card-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Input styling */
    .stNumberInput, .stSelectbox {
        border-radius: 12px;
        border: 2px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .stNumberInput:hover, .stSelectbox:hover {
        border-color: #764ba2;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.3);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 15px 40px;
        border: none;
        border-radius: 50px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Prediction boxes */
    .survived-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(17, 153, 142, 0.3);
        animation: pulse-green 2s infinite;
    }
    
    .not-survived-box {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(235, 51, 73, 0.3);
        animation: pulse-red 2s infinite;
    }
    
    .prediction-text {
        font-size: 2.5rem;
        font-weight: 900;
        margin: 10px 0;
    }
    
    .probability-text {
        font-size: 1.5rem;
        margin-top: 15px;
        font-weight: 600;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(40px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse-green {
        0%, 100% {
            box-shadow: 0 8px 32px rgba(17, 153, 142, 0.3);
        }
        50% {
            box-shadow: 0 8px 32px rgba(17, 153, 142, 0.6);
        }
    }
    
    @keyframes pulse-red {
        0%, 100% {
            box-shadow: 0 8px 32px rgba(235, 51, 73, 0.3);
        }
        50% {
            box-shadow: 0 8px 32px rgba(235, 51, 73, 0.6);
        }
    }
    
    /* Info box */
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border-left: 5px solid #667eea;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        color: #333;
    }
    
    /* Feature importance */
    .feature-row {
        display: flex;
        justify-content: space-between;
        padding: 15px;
        background: #f8f9ff;
        margin: 10px 0;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .feature-row:hover {
        background: #f0f2ff;
        transform: translateX(5px);
    }
    
    /* Metrics */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 900;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Section divider */
    hr {
        border: 0;
        height: 2px;
        background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
        margin: 30px 0;
    }
    
    /* Text styling */
    h1, h2, h3 {
        color: #333;
    }
    
    p {
        color: #555;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# LOAD MODELS (CACHED)
# ============================================
@st.cache_resource
def load_models():
    try:
        model = joblib.load('titanic_model.pkl')
        scaler = joblib.load('scaler.pkl')
        le_sex = joblib.load('le_sex.pkl')
        le_embarked = joblib.load('le_embarked.pkl')
        le_title = joblib.load('le_title.pkl')
        return model, scaler, le_sex, le_embarked, le_title
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None, None, None, None

model, scaler, le_sex, le_embarked, le_title = load_models()

# ============================================
# HEADER SECTION
# ============================================
st.markdown('<p class="title-main">🚢 Titanic Survival Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-main">🤖 Predict your survival using Machine Learning</p>', unsafe_allow_html=True)

# Add loading animation
with st.spinner('⏳ Loading models...'):
    time.sleep(0.5)

st.markdown("""
<div class="info-box">
💡 Enter your passenger information below and let our AI predict your survival chances on the Titanic!
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================
# MAIN CONTENT - TWO COLUMNS
# ============================================
col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">📋 Passenger Information</div>', unsafe_allow_html=True)
    
    # Create input fields with better layout
    col_age, col_fare = st.columns(2)
    with col_age:
        age = st.number_input('👤 Age (years)', min_value=0, max_value=120, value=30, step=1)
    with col_fare:
        fare = st.number_input('💰 Ticket Fare ($)', min_value=0.0, max_value=500.0, value=100.0, step=10.0)
    
    col_class, col_sex = st.columns(2)
    with col_class:
        pclass = st.selectbox('🎫 Passenger Class', [1, 2, 3], format_func=lambda x: f"{'1st Class - Upper Deck' if x==1 else '2nd Class - Middle' if x==2 else '3rd Class - Lower Deck'}")
    with col_sex:
        sex = st.selectbox('👨‍👩 Gender', ['Male', 'Female'])
    
    col_port, col_empty = st.columns(2)
    with col_port:
        embarked = st.selectbox('🌊 Port of Embarkation', 
                               ['Southampton', 'Cherbourg', 'Queenstown'],
                               format_func=lambda x: f"{'Southampton 🇬🇧' if x=='Southampton' else 'Cherbourg 🇫🇷' if x=='Cherbourg' else 'Queenstown 🇮🇪'}")
    
    col_sibsp, col_parch = st.columns(2)
    with col_sibsp:
        sibsp = st.number_input('👫 Siblings/Spouses', min_value=0, max_value=8, value=0, step=1)
    with col_parch:
        parch = st.number_input('👨‍👧 Parents/Children', min_value=0, max_value=6, value=0, step=1)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">ℹ️ About This Model</div>', unsafe_allow_html=True)
    
    with st.expander('🎯 Model Performance', expanded=True):
        st.markdown("""
        **Algorithm**: XGBoost Classifier
        
        **Accuracy**: 🎖️ 83.74%
        
        **Training Data**: 891 Titanic passengers
        
        **Key Metrics**:
        - Precision: 81.82%
        - Recall: 96.49%
        - F1-Score: 0.8889
        """)
    
    with st.expander('🔑 Important Features'):
        st.markdown("""
        1. **Gender** 👫 - Most important!
           - Women had much better survival rates
        
        2. **Fare** 💰 - Ticket price matters
           - Higher fare = better accommodation
        
        3. **Age** 👶 - Younger = Better
           - Children prioritized for lifeboats
        
        4. **Class** 🎫 - Social class effect
           - 1st: 63% vs 3rd: 24% survival
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ============================================
# PREDICTION SECTION
# ============================================
st.markdown('<div class="card-container">', unsafe_allow_html=True)

col1_pred, col2_pred = st.columns([1.5, 1], gap="large")

with col1_pred:
    st.markdown('<div class="card-header">🔮 Make Prediction</div>', unsafe_allow_html=True)
    predict_button = st.button('⚡ PREDICT SURVIVAL', key='predict_btn', use_container_width=True)

with col2_pred:
    st.markdown('<div style="text-align: center; padding: 20px;">', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 0.9rem; color: #666;">Click the button to get your prediction</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# PREDICTION LOGIC
# ============================================
if predict_button:
    if model is None:
        st.error("❌ Models not loaded. Please try again.")
    else:
        with st.spinner('🤔 Analyzing passenger data...'):
            time.sleep(1)
            
            try:
                # Feature engineering
                family_size = sibsp + parch + 1
                is_alone = 1 if family_size == 1 else 0
                
                # Encode categorical
                sex_encoded = le_sex.transform([sex])[0]
                embarked_encoded = le_embarked.transform([embarked])[0]
                title_encoded = le_title.transform(['Mr'])[0]
                
                # Create feature array
                features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, 
                                     embarked_encoded, family_size, title_encoded, is_alone]])
                
                # Scale
                features_scaled = scaler.transform(features)
                
                # Predict
                prediction = model.predict(features_scaled)[0]
                probability = model.predict_proba(features_scaled)[0]
                
                st.markdown("---")
                
                # ============================================
                # PREDICTION RESULT
                # ============================================
                col_result1, col_result2 = st.columns([1, 1], gap="large")
                
                with col_result1:
                    if prediction == 1:
                        st.markdown("""
                        <div class="survived-box">
                            <div class="prediction-text">✅ SURVIVED</div>
                            <div class="probability-text">%s%% Probability</div>
                        </div>
                        """ % int(probability[1]*100), unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="not-survived-box">
                            <div class="prediction-text">❌ DID NOT SURVIVE</div>
                            <div class="probability-text">%s%% Probability</div>
                        </div>
                        """ % int(probability[0]*100), unsafe_allow_html=True)
                
                with col_result2:
                    # Probability gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=probability[1]*100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Survival Probability", 'font': {'size': 20}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "rgba(102, 126, 234, 0.8)", 'thickness': 0.7},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 25], 'color': "rgba(255, 107, 107, 0.2)"},
                                {'range': [25, 50], 'color': "rgba(255, 193, 7, 0.2)"},
                                {'range': [50, 75], 'color': "rgba(76, 175, 80, 0.2)"},
                                {'range': [75, 100], 'color': "rgba(76, 175, 80, 0.4)"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        },
                        number={'suffix': "%", 'font': {'size': 30, 'color': '#667eea'}}
                    ))
                    
                    fig_gauge.update_layout(
                        height=350,
                        font=dict(family="Arial", size=12),
                        margin=dict(l=10, r=10, t=50, b=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
                
                st.markdown("---")
                
                # ============================================
                # PREDICTION BREAKDOWN
                # ============================================
                st.markdown('<div class="card-header">📊 Prediction Breakdown</div>', unsafe_allow_html=True)
                
                breakdown_data = pd.DataFrame({
                    'Feature': ['Age', 'Ticket Fare', 'Class', 'Gender', 'Embarkation Port', 'Family Size', 'Traveling Alone'],
                    'Your Value': [f'{age} years', f'${fare:.2f}', f'{pclass}', sex, embarked, family_size, '✅ Yes' if is_alone else '❌ No'],
                    'Impact': ['Younger ↑', 'Higher ↑', 'Lower ↑', 'Female ↑', 'Varies', 'Medium', 'Worse ↓']
                })
                
                st.dataframe(breakdown_data, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                # ============================================
                # CONFIDENCE METRICS
                # ============================================
                st.markdown('<div class="card-header">💎 Confidence Analysis</div>', unsafe_allow_html=True)
                
                col_m1, col_m2, col_m3 = st.columns(3)
                
                with col_m1:
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-label">Model Accuracy</div>
                        <div class="metric-value">83.74%</div>
                        <div class="metric-label">↑ High Precision</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m2:
                    confidence = "🟢 HIGH" if max(probability) > 0.75 else "🟡 MEDIUM" if max(probability) > 0.6 else "🔴 LOW"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Prediction Confidence</div>
                        <div class="metric-value">{int(max(probability)*100)}%</div>
                        <div class="metric-label">{confidence}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m3:
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-label">Survival Rate</div>
                        <div class="metric-value">%s%%</div>
                        <div class="metric-label">Your Chances</div>
                    </div>
                    """ % int(probability[1]*100), unsafe_allow_html=True)
                
                st.markdown("---")
                
                # ============================================
                # SIMILAR PASSENGER STATISTICS
                # ============================================
                st.markdown('<div class="card-header">📈 Statistical Insights</div>', unsafe_allow_html=True)
                
                col_insight1, col_insight2 = st.columns(2)
                
                with col_insight1:
                    st.markdown(f"""
                    <div class="info-box">
                    **Your Passenger Profile**:
                    - Age Group: {age} years
                    - Ticket Class: {pclass}
                    - Gender: {sex}
                    - Family Aboard: {family_size} people
                    - Ticket Cost: ${fare:.2f}
                    </div>
                    """)
                
                with col_insight2:
                    st.markdown(f"""
                    <div class="info-box">
                    **Key Factors for You**:
                    - Gender Impact: {'✅ Strong Advantage' if sex == 'Female' else '⚠️ Disadvantage'}
                    - Class Impact: {'✅ Good Position' if pclass == 1 else '⚠️ Lower Priority' if pclass == 3 else '➖ Average'}
                    - Age Factor: {'✅ Favorable' if age < 18 else '➖ Average' if age < 50 else '⚠️ Less Favorable'}
                    </div>
                    """)
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                st.info("Please check your inputs and try again")

st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================
st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns([1, 1, 1])

with footer_col1:
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
    <h4>🎓 ML Model</h4>
    <p style="font-size: 0.9rem;">XGBoost Classifier<br>Trained on 891 passengers</p>
    </div>
    """, unsafe_allow_html=True)

with footer_col2:
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
    <h4>📊 Performance</h4>
    <p style="font-size: 0.9rem;">Accuracy: 83.74%<br>ROC-AUC: 0.9085</p>
    </div>
    """, unsafe_allow_html=True)

with footer_col3:
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
    <h4>🚀 Platform</h4>
    <p style="font-size: 0.9rem;">Built with Streamlit<br>@NeurofiveSolutions</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; color: #999; margin-top: 30px; padding: 20px; border-top: 2px solid #ddd;">
    <p><small>💖 Made with love | ML Deployment Demo | Data Science Portfolio</small></p>
    <p><small>© 2024 Titanic Survival Prediction | All Rights Reserved</small></p>
</div>
""", unsafe_allow_html=True)

