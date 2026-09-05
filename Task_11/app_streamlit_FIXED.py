"""
Cricket Win Probability Predictor - Streamlit App
Beautiful UI with animations, real-time predictions, and visualizations
FIXED VERSION: Uses relative paths instead of hardcoded paths
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import os

# ============================================================================
# PAGE CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(
    page_title="🏏 Cricket Win Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for animations and styling
st.markdown("""
<style>
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.7;
        }
    }
    
    .main-header {
        animation: slideIn 0.6s ease-out;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    .metric-card {
        animation: slideIn 0.8s ease-out;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-box {
        animation: pulse 2s infinite;
        background: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    
    .stat-label {
        font-size: 12px;
        color: #666;
        text-transform: uppercase;
        font-weight: bold;
        letter-spacing: 1px;
    }
    
    .stat-value {
        font-size: 24px;
        font-weight: bold;
        color: #333;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD TRAINED MODEL & ARTIFACTS (FIXED PATHS)
# ============================================================================

@st.cache_resource
def load_model_artifacts():
    """Load trained model and preprocessing artifacts with relative paths"""
    try:
        # Get current directory (where app_streamlit.py is located)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Build paths to model files
        model_path = os.path.join(current_dir, 'models', 'cricket_win_predictor.pkl')
        scaler_path = os.path.join(current_dir, 'models', 'feature_scaler.pkl')
        encoders_path = os.path.join(current_dir, 'models', 'label_encoders.pkl')
        features_path = os.path.join(current_dir, 'models', 'feature_columns.pkl')
        metadata_path = os.path.join(current_dir, 'models', 'model_metadata.pkl')
        
        # Check if files exist
        required_files = {
            'cricket_win_predictor.pkl': model_path,
            'feature_scaler.pkl': scaler_path,
            'label_encoders.pkl': encoders_path,
            'feature_columns.pkl': features_path,
            'model_metadata.pkl': metadata_path
        }
        
        missing_files = [name for name, path in required_files.items() if not os.path.exists(path)]
        
        if missing_files:
            st.error(f"""
            ❌ Missing model files: {', '.join(missing_files)}
            
            Please run: python cricket_win_predictor.py
            to generate .pkl files in the /models/ directory
            """)
            return None, None, None, None, None
        
        # Load files
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(encoders_path, 'rb') as f:
            label_encoders = pickle.load(f)
        with open(features_path, 'rb') as f:
            feature_cols = pickle.load(f)
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        return model, scaler, label_encoders, feature_cols, metadata
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}\n\nDebug info:\n{str(e)}")
        return None, None, None, None, None

# ============================================================================
# FEATURE ENGINEERING FUNCTION
# ============================================================================

def engineer_prediction_features(runs_left, balls_left, wickets_remaining, crr, rrr, total_runs):
    """Create features for a single prediction"""
    
    features = {
        'runs_left': runs_left,
        'balls_left': balls_left,
        'wickets_remaining': wickets_remaining,
        'total_run_x': total_runs,
        'crr': crr,
        'rrr': rrr,
    }
    
    # Engineered features
    features['runs_per_ball'] = runs_left / balls_left if balls_left > 0 else 0
    features['wickets_per_ball'] = wickets_remaining / balls_left if balls_left > 0 else 0
    features['run_rate_gap'] = rrr - crr
    features['crr_to_rrr_ratio'] = crr / rrr if rrr > 0 else 0
    
    balls_played = 120 - balls_left
    features['balls_played'] = balls_played
    features['progress_percentage'] = (balls_played / 120) * 100
    features['wickets_lost'] = 10 - wickets_remaining
    features['wicket_loss_rate'] = features['wickets_lost'] / balls_played if balls_played > 0 else 0
    
    features['high_scoring_phase'] = 1 if crr > 7.0 else 0
    features['death_phase'] = 1 if balls_left <= 30 else 0
    features['critical_moment'] = 1 if features['run_rate_gap'] > 3.0 else 0
    
    features['rrr_normalized'] = rrr / 10
    features['crr_normalized'] = crr / 10
    
    # Additional features
    features['runs_per_wicket'] = runs_left / wickets_remaining if wickets_remaining > 0 else 0
    features['runs_per_over'] = (runs_left / balls_left * 6) if balls_left > 0 else 0
    features['overs_played'] = balls_played / 6
    features['overs_left'] = balls_left / 6
    features['runs_scored'] = total_runs - runs_left
    features['rr_change_needed'] = features['run_rate_gap'] / max(crr, 0.1)
    features['wicket_pressure'] = features['wickets_lost'] / max(balls_played / 120, 0.1) if balls_played > 0 else 0
    features['boundaries_needed'] = runs_left / 4 if runs_left > 0 else 0
    features['crr_rrr_diff'] = crr - rrr
    
    return features

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def make_prediction(model, scaler, label_encoders, feature_cols, batting_team, bowling_team, city,
                   runs_left, balls_left, wickets_remaining, crr, rrr, total_runs):
    """Make win probability prediction"""
    
    # Create feature vector
    features = engineer_prediction_features(runs_left, balls_left, wickets_remaining, crr, rrr, total_runs)
    
    # Encode categorical variables
    try:
        encoded_batting = label_encoders['batting_team'].transform([batting_team])[0]
        encoded_bowling = label_encoders['bowling_team'].transform([bowling_team])[0]
        encoded_city = label_encoders['city'].transform([city])[0]
    except:
        encoded_batting, encoded_bowling, encoded_city = 0, 0, 0
    
    features['batting_team'] = encoded_batting
    features['bowling_team'] = encoded_bowling
    features['city'] = encoded_city
    
    # Create dataframe in correct order
    feature_df = pd.DataFrame([features])[feature_cols]
    
    # Scale features
    feature_scaled = scaler.transform(feature_df)
    
    # Make prediction
    win_probability = model.predict_proba(feature_scaled)[0][1]
    prediction = model.predict(feature_scaled)[0]
    
    return win_probability, prediction

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_probability_gauge(probability):
    """Create animated probability gauge"""
    fig = go.Figure(data=[go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Win Probability %"},
        delta={'reference': 50, 'suffix': "%"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "#ff6b6b"},
                {'range': [25, 50], 'color': "#ffd93d"},
                {'range': [50, 75], 'color': "#a8e6cf"},
                {'range': [75, 100], 'color': "#51cf66"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    )])
    
    fig.update_layout(
        height=400,
        font={'size': 14},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_performance_metrics(runs_left, balls_left, wickets_remaining, crr, rrr):
    """Create performance metrics visualization"""
    
    metrics = {
        'Metric': ['Runs Needed', 'Balls Left', 'Run Rate Gap', 'Wickets Left'],
        'Value': [runs_left, balls_left, rrr - crr, wickets_remaining],
        'Color': ['#667eea', '#764ba2', '#f5576c', '#f093fb']
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=metrics['Metric'],
            y=metrics['Value'],
            marker=dict(color=metrics['Color']),
            text=metrics['Value'],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Match Performance Metrics',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(240,240,240,0.5)',
        showlegend=False
    )
    
    return fig

def create_match_situation_chart(balls_played, progress, runs_left, total_runs, crr, rrr):
    """Create match situation visualization"""
    
    fig = go.Figure()
    
    # Progress bar
    fig.add_trace(go.Bar(
        x=[progress],
        y=['Match Progress'],
        orientation='h',
        marker=dict(color='#667eea'),
        text=f'{progress:.0f}%',
        textposition='auto',
    ))
    
    fig.update_layout(
        title='Match Progress & Run Rate',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(240,240,240,0.5)',
        showlegend=False
    )
    
    return fig

def create_scenario_analysis(model, scaler, label_encoders, feature_cols, base_params):
    """Create scenario analysis - probability vs CRR"""
    
    scenarios = []
    crr_values = np.arange(base_params['crr'] - 2, base_params['crr'] + 3, 0.2)
    
    for crr in crr_values:
        prob, _ = make_prediction(
            model, scaler, label_encoders, feature_cols,
            base_params['batting_team'],
            base_params['bowling_team'],
            base_params['city'],
            base_params['runs_left'],
            base_params['balls_left'],
            base_params['wickets_remaining'],
            crr,
            base_params['rrr'],
            base_params['total_runs']
        )
        scenarios.append({'Current Run Rate': f"{crr:.1f}", 'Win Probability': prob * 100})
    
    df_scenarios = pd.DataFrame(scenarios)
    
    fig = px.line(
        df_scenarios,
        x='Current Run Rate',
        y='Win Probability',
        markers=True,
        title='Win Probability vs Current Run Rate',
        labels={'Win Probability': 'Win Probability (%)'},
        line_shape='spline'
    )
    
    fig.update_traces(line=dict(color='#667eea', width=3))
    fig.update_layout(
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(240,240,240,0.5)',
        hovermode='x unified'
    )
    
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Load model
    model, scaler, label_encoders, feature_cols, metadata = load_model_artifacts()
    
    if model is None:
        st.error("Failed to load model. Please check the model artifacts.")
        st.info("""
        📋 Steps to fix:
        1. Run: python cricket_win_predictor.py
        2. Check that /models/ folder exists with .pkl files
        3. Refresh this page (F5)
        """)
        return
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏏 Cricket Win Probability Predictor</h1>
        <p>Real-time match situation analysis powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Info Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model Accuracy", f"{metadata['best_f1']*100:.1f}%", "F1-Score")
    with col2:
        st.metric("ROC-AUC Score", f"{metadata['best_roc_auc']:.4f}", "Classification")
    with col3:
        st.metric("Matches Analyzed", "93,466", "Training Data")
    with col4:
        st.metric("Features Used", len(feature_cols), "Engineered")
    
    st.divider()
    
    # Sidebar for Input
    st.sidebar.header("📋 Match Input")
    
    # Teams and Venue
    teams = list(label_encoders['batting_team'].classes_)
    cities = list(label_encoders['city'].classes_)
    
    batting_team = st.sidebar.selectbox("Batting Team", teams)
    bowling_team = st.sidebar.selectbox("Bowling Team", [t for t in teams if t != batting_team])
    city = st.sidebar.selectbox("Venue", cities)
    
    st.sidebar.divider()
    
    # Match Situation
    st.sidebar.subheader("Match Situation")
    
    col_left, col_right = st.sidebar.columns(2)
    with col_left:
        runs_left = st.number_input("Runs Left", min_value=0, max_value=300, value=50, step=1)
        balls_left = st.number_input("Balls Left", min_value=1, max_value=120, value=60, step=1)
        wickets_remaining = st.number_input("Wickets", min_value=0, max_value=10, value=8, step=1)
    
    with col_right:
        crr = st.number_input("Current Run Rate", min_value=0.0, max_value=20.0, value=6.5, step=0.1)
        rrr = st.number_input("Required Run Rate", min_value=0.0, max_value=30.0, value=7.0, step=0.1)
        total_runs = st.number_input("Total Target", min_value=50, max_value=300, value=160, step=1)
    
    st.sidebar.divider()
    
    # Make Prediction
    if st.sidebar.button("🎯 Predict Win Probability", key="predict_btn", use_container_width=True):
        with st.spinner("Analyzing match situation..."):
            time.sleep(0.5)  # Simulate processing
            
            win_prob, prediction = make_prediction(
                model, scaler, label_encoders, feature_cols,
                batting_team, bowling_team, city,
                runs_left, balls_left, wickets_remaining, crr, rrr, total_runs
            )
            
            st.session_state.win_prob = win_prob
            st.session_state.prediction = prediction
            st.session_state.base_params = {
                'batting_team': batting_team,
                'bowling_team': bowling_team,
                'city': city,
                'runs_left': runs_left,
                'balls_left': balls_left,
                'wickets_remaining': wickets_remaining,
                'crr': crr,
                'rrr': rrr,
                'total_runs': total_runs
            }
    
    # Display Results
    if 'win_prob' in st.session_state:
        st.divider()
        
        # Prediction Result
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            prediction_result = st.session_state.prediction
            win_prob = st.session_state.win_prob
            
            if prediction_result == 1:
                result_text = "✅ LIKELY WIN"
                result_color = "#51cf66"
            else:
                result_text = "⚠️ LIKELY LOSS"
                result_color = "#ff6b6b"
            
            st.markdown(f"""
            <div style="background-color: {result_color}; padding: 20px; border-radius: 10px; 
                        text-align: center; color: white;">
                <h2>{result_text}</h2>
                <h3>{win_prob*100:.1f}% Win Probability</h3>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Visualizations
        st.subheader("📊 Match Analysis")
        
        # Probability Gauge
        col1, col2 = st.columns(2)
        
        with col1:
            fig_gauge = create_probability_gauge(win_prob)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            fig_rates = create_performance_metrics(
                st.session_state.base_params['runs_left'],
                st.session_state.base_params['balls_left'],
                st.session_state.base_params['wickets_remaining'],
                st.session_state.base_params['crr'],
                st.session_state.base_params['rrr']
            )
            st.plotly_chart(fig_rates, use_container_width=True)
        
        # Additional Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            fig_runs = create_match_situation_chart(
                120 - st.session_state.base_params['balls_left'],
                (120 - st.session_state.base_params['balls_left']) / 120 * 100,
                st.session_state.base_params['runs_left'],
                st.session_state.base_params['total_runs'],
                st.session_state.base_params['crr'],
                st.session_state.base_params['rrr']
            )
            st.plotly_chart(fig_runs, use_container_width=True)
        
        with col2:
            fig_scenario = create_scenario_analysis(
                model, scaler, label_encoders, feature_cols,
                st.session_state.base_params
            )
            st.plotly_chart(fig_scenario, use_container_width=True)
        
        # Match Details
        st.subheader("📈 Detailed Match Information")
        
        detail_col1, detail_col2, detail_col3 = st.columns(3)
        
        with detail_col1:
            st.info(f"""
            **Match Details**
            - Batting: {st.session_state.base_params['batting_team']}
            - Bowling: {st.session_state.base_params['bowling_team']}
            - Venue: {st.session_state.base_params['city']}
            """)
        
        with detail_col2:
            balls_played = 120 - st.session_state.base_params['balls_left']
            overs_played = balls_played / 6
            overs_left = st.session_state.base_params['balls_left'] / 6
            
            st.info(f"""
            **Over Progress**
            - Overs Played: {overs_played:.1f}
            - Overs Left: {overs_left:.1f}
            - Runs Scored: {st.session_state.base_params['total_runs'] - st.session_state.base_params['runs_left']}
            """)
        
        with detail_col3:
            st.info(f"""
            **Rate Analysis**
            - Current Rate: {st.session_state.base_params['crr']:.2f} runs/over
            - Required Rate: {st.session_state.base_params['rrr']:.2f} runs/over
            - Gap: {st.session_state.base_params['rrr'] - st.session_state.base_params['crr']:.2f}
            """)

if __name__ == "__main__":
    main()
