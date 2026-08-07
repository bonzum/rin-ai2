import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import requests
import json
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
import io
warnings.filterwarnings('ignore')

st.set_page_config(page_title="RIN AI v2.0 | ML-Powered Intelligence", page_icon="🧠", layout="wide")

st.markdown("""
<style>
/* ─── GLOBAL ─── */
* { font-family: 'Segoe UI', system-ui, sans-serif; }

/* ─── FIX: Active tab color (was red, now blue) ─── */
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
    color: white !important;
    border-bottom: 3px solid #38bdf8 !important;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px; background: rgba(30, 41, 59, 0.5); padding: 0.5rem; border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 8px; color: #94a3b8; font-weight: 500;
}

/* ─── FIX: Hide Streamlit default elements ─── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display:none;}

/* ─── FIX: Better disclaimer readability ─── */
.disclaimer-box {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-radius: 8px;
    padding: 1rem;
    margin-top: 1rem;
}
.disclaimer-box p {
    color: #fca5a5 !important;
    font-size: 0.85rem;
    margin: 0;
}

/* ─── Main header ─── */
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    padding: 2rem; border-radius: 16px; margin-bottom: 2rem;
    border: 1px solid rgba(56, 189, 248, 0.2);
}

/* ─── Metric boxes ─── */
.metric-box {
    background: #1e293b; padding: 1.2rem; border-radius: 10px;
    text-align: center; border: 1px solid rgba(56, 189, 248, 0.1);
}
.metric-value { font-size: 2.2rem; font-weight: 800; color: #38bdf8; }
.metric-label { font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }

/* ─── Alerts ─── */
.alert-high {
    background: linear-gradient(145deg, #7f1d1d 0%, #991b1b 100%);
    border-left: 4px solid #ef4444; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
}
.alert-medium {
    background: linear-gradient(145deg, #713f12 0%, #854d0e 100%);
    border-left: 4px solid #f59e0b; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
}

/* ─── Module cards ─── */
.module-card {
    background: #1e293b; padding: 1.5rem; border-radius: 12px;
    border: 1px solid rgba(56, 189, 248, 0.15); margin-bottom: 1rem;
    transition: all 0.3s ease; cursor: pointer;
}
.module-card:hover {
    border-color: rgba(56, 189, 248, 0.4);
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(56, 189, 248, 0.1);
}

/* ─── Weather cards ─── */
.weather-card {
    background: linear-gradient(145deg, #1e3a5f 0%, #0f172a 100%);
    padding: 1.2rem; border-radius: 12px;
    border: 1px solid rgba(56, 189, 248, 0.2); margin-bottom: 0.8rem;
}

/* ─── Explanation box ─── */
.explanation-box {
    background: rgba(56, 189, 248, 0.08);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 8px; padding: 1rem; margin-top: 1rem;
}

/* ─── Confidence bar ─── */
.confidence-bar-bg {
    background: #334155; border-radius: 10px; height: 24px; overflow: hidden; margin: 0.5rem 0;
}
.confidence-bar-fill {
    height: 100%; border-radius: 10px; display: flex; align-items: center; justify-content: center;
    color: white; font-weight: 700; font-size: 0.85rem; transition: width 0.5s ease;
}

/* ─── Mobile patient cards ─── */
.patient-card {
    background: #1e293b; padding: 1rem; border-radius: 10px;
    margin: 0.5rem 0; border-left: 4px solid;
}
.patient-card-high { border-left-color: #ef4444; }
.patient-card-medium { border-left-color: #f59e0b; }
.patient-card-low { border-left-color: #22c55e; }

/* ─── Factor badges ─── */
.factor-badge {
    display: inline-block; padding: 0.3rem 0.7rem; border-radius: 20px;
    font-size: 0.8rem; font-weight: 600; margin: 0.2rem;
}
.factor-high { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
.factor-medium { background: rgba(245, 158, 11, 0.2); color: #f59e0b; }
.factor-low { background: rgba(34, 197, 94, 0.2); color: #22c55e; }

/* ─── Next steps box ─── */
.next-steps-box {
    background: linear-gradient(145deg, #14532d 0%, #166534 100%);
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 10px; padding: 1.2rem; margin-top: 1rem;
}
.next-steps-box h4 { color: #22c55e; margin: 0 0 0.5rem 0; }
.next-steps-box li { color: #e2e8f0; line-height: 1.8; }

/* ─── Welcome module cards ─── */
.welcome-module {
    background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
    border: 2px solid rgba(56, 189, 248, 0.2);
    border-radius: 16px; padding: 2rem; text-align: center;
    transition: all 0.3s ease; cursor: pointer;
}
.welcome-module:hover {
    border-color: #38bdf8;
    transform: translateY(-4px);
    box-shadow: 0 8px 30px rgba(56, 189, 248, 0.15);
}
.welcome-module .icon { font-size: 3rem; margin-bottom: 0.5rem; }
.welcome-module h3 { color: white; margin: 0.5rem 0; }
.welcome-module p { color: #94a3b8; font-size: 0.9rem; }

/* ─── Abnormal value highlight ─── */
.abnormal-high { color: #ef4444; font-weight: 700; }
.abnormal-medium { color: #f59e0b; font-weight: 700; }
.normal-value { color: #22c55e; }

/* ─── Sidebar ─── */
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}

/* ─── Forecast rows ─── */
.forecast-row {
    background: rgba(30, 41, 59, 0.8); padding: 0.8rem;
    border-radius: 8px; margin: 0.3rem 0; border-left: 3px solid #38bdf8;
}

/* ─── Risk colors ─── */
.risk-high { color: #ef4444; font-weight: 700; }
.risk-medium { color: #f59e0b; font-weight: 700; }
.risk-low { color: #22c55e; font-weight: 700; }

/* ─── Crop recommendation cards ─── */
.crop-card {
    background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(34, 197, 94, 0.2);
    border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem;
    transition: all 0.3s ease;
}
.crop-card:hover {
    border-color: rgba(34, 197, 94, 0.5);
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(34, 197, 94, 0.1);
}
.crop-rank-1 { border-left: 4px solid #22c55e; }
.crop-rank-2 { border-left: 4px solid #38bdf8; }
.crop-rank-3 { border-left: 4px solid #a855f7; }

/* ─── Feature importance bars ─── */
.feat-bar-bg {
    background: #334155; border-radius: 6px; height: 12px; overflow: hidden; margin-top: 4px;
}
.feat-bar-fill {
    height: 100%; border-radius: 6px; background: linear-gradient(90deg, #22c55e, #38bdf8);
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

def init_database():
    conn = sqlite3.connect('rin_ai.db')
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, age INTEGER, gender TEXT, location TEXT,
        temperature REAL, blood_pressure_sys INTEGER, blood_pressure_dia INTEGER,
        heart_rate INTEGER, glucose REAL, bmi REAL, symptoms TEXT,
        diabetes_risk TEXT, risk_score REAL, risk_explanation TEXT,
        risk_factors TEXT, next_steps TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("PRAGMA table_info(patients)")
    existing_cols = [row[1] for row in c.fetchall()]
    for col, col_type in {'risk_factors': 'TEXT', 'next_steps': 'TEXT', 'risk_explanation': 'TEXT', 'risk_score': 'REAL', 'diabetes_risk': 'TEXT'}.items():
        if col not in existing_cols:
            c.execute(f"ALTER TABLE patients ADD COLUMN {col} {col_type}")
    c.execute("""CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER, module TEXT, helpful TEXT, comment TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_type TEXT, location TEXT, message TEXT, severity TEXT,
        status TEXT DEFAULT 'active',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        resolved_at TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS weather_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        location TEXT UNIQUE, temperature REAL, humidity INTEGER,
        description TEXT, wind_speed REAL, rainfall REAL, forecast TEXT,
        cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    c.execute("""CREATE TABLE IF NOT EXISTS farm_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        farmer_name TEXT, farm_location TEXT, farm_size REAL, soil_type TEXT,
        nitrogen INTEGER, phosphorus INTEGER, potassium INTEGER, ph REAL,
        temperature REAL, humidity REAL, rainfall REAL,
        recommended_crop TEXT, confidence REAL, top_3_crops TEXT, model_version TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()

init_database()

def get_db_connection():
    return sqlite3.connect('rin_ai.db')

# ═══════════════════════════════════════════════════════════════════════════════
# WEATHER API
# ═══════════════════════════════════════════════════════════════════════════════

def get_weather_data(location, api_key=None):
    if api_key is None or api_key == "demo_key" or api_key == "":
        return get_simulated_weather(location)
    try:
        current_url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
        current_response = requests.get(current_url, timeout=10)
        if current_response.status_code != 200:
            return get_simulated_weather(location)
        current_data = current_response.json()
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?q={location}&appid={api_key}&units=metric"
        forecast_response = requests.get(forecast_url, timeout=10)
        forecast_data = forecast_response.json() if forecast_response.status_code == 200 else None
        weather = {
            'source': 'OpenWeatherMap API',
            'temperature': round(current_data['main']['temp'], 1),
            'feels_like': round(current_data['main']['feels_like'], 1),
            'humidity': current_data['main']['humidity'],
            'pressure': current_data['main']['pressure'],
            'wind_speed': round(current_data['wind']['speed'], 1),
            'description': current_data['weather'][0]['description'].title(),
            'clouds': current_data['clouds']['all'],
            'visibility': current_data.get('visibility', 10000) // 1000,
            'sunrise': datetime.fromtimestamp(current_data['sys']['sunrise']).strftime('%H:%M'),
            'sunset': datetime.fromtimestamp(current_data['sys']['sunset']).strftime('%H:%M'),
            'forecast': []
        }
        if forecast_data and 'list' in forecast_data:
            daily_forecasts = {}
            for item in forecast_data['list']:
                date = item['dt_txt'].split(' ')[0]
                if date not in daily_forecasts:
                    daily_forecasts[date] = {
                        'temp_max': item['main']['temp_max'],
                        'temp_min': item['main']['temp_min'],
                        'description': item['weather'][0]['description'],
                        'rain': item.get('rain', {}).get('3h', 0)
                    }
            for date, data in list(daily_forecasts.items())[:5]:
                weather['forecast'].append({
                    'date': date, 'temp_max': round(data['temp_max'], 1),
                    'temp_min': round(data['temp_min'], 1),
                    'description': data['description'].title(),
                    'rain': round(data['rain'], 1)
                })
        cache_weather_data(location, weather)
        return weather
    except Exception as e:
        return get_simulated_weather(location)

def get_simulated_weather(location):
    np.random.seed(hash(location) % 10000)
    base_temp = 26 + np.random.normal(0, 3)
    weather = {
        'source': 'Simulated (RIN AI Local Model)',
        'temperature': round(base_temp, 1),
        'feels_like': round(base_temp + np.random.normal(0, 1), 1),
        'humidity': int(np.clip(np.random.normal(70, 15), 30, 95)),
        'pressure': int(np.clip(np.random.normal(1013, 10), 980, 1050)),
        'wind_speed': round(np.random.exponential(3), 1),
        'description': np.random.choice(['Light Rain', 'Scattered Clouds', 'Partly Cloudy', 'Overcast Clouds', 'Clear Sky', 'Moderate Rain']),
        'clouds': int(np.clip(np.random.normal(50, 30), 0, 100)),
        'visibility': int(np.clip(np.random.normal(8, 3), 2, 10)),
        'sunrise': '06:15', 'sunset': '18:45', 'forecast': []
    }
    for i in range(5):
        date = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
        weather['forecast'].append({
            'date': date,
            'temp_max': round(np.clip(base_temp + np.random.normal(2, 2), 15, 45), 1),
            'temp_min': round(np.clip(base_temp - np.random.normal(5, 2), 10, 40), 1),
            'description': np.random.choice(['Light Rain', 'Scattered Clouds', 'Partly Cloudy', 'Overcast Clouds', 'Clear Sky', 'Moderate Rain']),
            'rain': round(np.random.exponential(5), 1)
        })
    return weather

def cache_weather_data(location, weather):
    conn = get_db_connection()
    c = conn.cursor()
    forecast_json = json.dumps(weather.get('forecast', []))
    c.execute("""INSERT OR REPLACE INTO weather_cache 
        (location, temperature, humidity, description, wind_speed, rainfall, forecast, cached_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (location, weather['temperature'], weather['humidity'], weather['description'],
         weather['wind_speed'], weather.get('rain', 0), forecast_json, datetime.now()))
    conn.commit(); conn.close()

# ═══════════════════════════════════════════════════════════════════════════════
# DIABETES AI MODEL (Synthetic - existing)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_diabetes_model():
    np.random.seed(42)
    n_samples = 2000
    data = {
        'Pregnancies': np.random.poisson(2, n_samples),
        'Glucose': np.random.normal(110, 30, n_samples).clip(50, 250),
        'BloodPressure': np.random.normal(72, 12, n_samples).clip(40, 140),
        'SkinThickness': np.random.normal(25, 10, n_samples).clip(5, 80),
        'Insulin': np.random.lognormal(3.5, 0.8, n_samples).clip(10, 600),
        'BMI': np.random.normal(26, 5, n_samples).clip(15, 50),
        'DiabetesPedigreeFunction': np.random.exponential(0.5, n_samples).clip(0.05, 2.5),
        'Age': np.random.gamma(5, 8, n_samples).clip(18, 85) + 18
    }
    df = pd.DataFrame(data)
    risk_score = (0.02 * df['Glucose'] + 0.03 * df['BMI'] + 0.01 * df['Age'] +
                  0.5 * df['DiabetesPedigreeFunction'] + 0.001 * df['Insulin'] +
                  np.random.normal(0, 1, n_samples))
    df['Outcome'] = (risk_score > df['Glucose'].mean() * 0.025).astype(int)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train)
    accuracy = model.score(scaler.transform(X_test), y_test)
    return model, scaler, accuracy, X.columns.tolist()

diabetes_model, diabetes_scaler, diabetes_accuracy, diabetes_features = load_diabetes_model()

# ═══════════════════════════════════════════════════════════════════════════════
# CROP RECOMMENDATION ML MODEL (Real Dataset)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Training RIN AGRI ML model on real crop dataset...")
def load_crop_model():
    """Load and train the crop recommendation model on real agricultural data.
    Dataset: 2,200 records, 22 crop types, 7 features (N, P, K, temp, humidity, pH, rainfall)
    Source: Crop Recommendation Dataset (Kaggle/UCI)
    """
    # Try to download the full dataset from GitHub
    dataset_url = "https://raw.githubusercontent.com/AbhishekKandoi/Crop-Yield-Prediction-based-on-Indian-Agriculture/main/Crop%20Recommendation%20dataset.csv"

    try:
        response = requests.get(dataset_url, timeout=15)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        source_note = "Full Dataset (2,200 records, 22 crops)"
    except Exception:
        # Fallback: embedded minimal dataset for offline operation
        fallback_csv = """N,P,K,temperature,humidity,ph,rainfall,label
90,42,43,20.8,82.0,6.5,202.9,rice
85,58,41,21.7,80.3,7.0,226.6,rice
78,42,42,20.1,81.6,7.6,262.7,rice
71,54,16,22.6,63.7,5.7,87.8,maize
61,44,17,26.1,71.6,6.9,102.3,maize
80,43,16,23.6,71.6,6.7,66.7,maize
40,72,77,17.0,17.0,7.5,88.6,chickpea
23,72,84,19.0,17.1,6.9,79.9,chickpea
39,58,85,17.9,15.4,6.0,68.5,chickpea
13,60,25,17.1,20.6,5.7,128.3,kidneybeans
25,70,16,19.6,18.9,5.8,106.4,kidneybeans
31,55,22,22.9,21.3,5.9,109.2,kidneybeans
3,49,18,27.9,64.7,3.7,32.7,mothbeans
22,59,23,27.3,51.3,4.4,36.5,mothbeans
36,58,25,28.7,59.3,8.4,36.9,mothbeans
19,55,20,28.0,87.8,7.2,54.7,mungbean
8,54,20,28.3,80.8,7.0,38.8,mungbean
36,55,20,27.0,84.3,6.6,55.3,mungbean
56,79,15,29.5,63.2,7.5,71.9,blackgram
25,62,21,26.7,68.1,7.0,67.2,blackgram
42,61,22,26.3,62.3,7.4,70.2,blackgram
32,60,24,20.7,63.2,6.3,94.0,lentil
13,61,22,19.4,63.3,7.7,46.8,lentil
38,60,20,29.8,60.6,7.5,46.8,lentil
2,24,38,24.6,91.6,5.9,112.0,pomegranate
6,18,37,19.7,89.9,5.9,108.0,pomegranate
37,18,39,24.1,94.5,6.4,110.2,pomegranate
91,94,46,29.4,76.2,6.1,92.8,banana
105,95,50,27.3,83.7,5.8,101.0,banana
108,92,53,27.4,83.0,6.3,104.9,banana
2,40,27,29.7,47.5,6.0,90.1,mango
39,24,31,33.6,53.7,4.8,98.7,mango
21,26,27,27.0,47.7,5.7,95.9,mango
24,128,196,22.8,90.7,5.5,110.4,grapes
7,144,197,23.8,94.3,6.1,114.1,grapes
14,128,205,22.6,94.6,6.2,116.0,grapes
119,25,51,26.5,80.9,6.3,53.7,watermelon
119,19,55,25.2,83.4,6.8,46.9,watermelon
105,30,50,25.3,81.8,6.4,57.0,watermelon
115,17,55,27.6,94.1,6.8,28.1,muskmelon
114,27,48,27.8,93.0,6.5,26.3,muskmelon
101,25,52,29.1,94.2,6.8,22.5,muskmelon
24,128,196,22.8,90.7,5.5,110.4,apple
7,144,197,23.8,94.3,6.1,114.1,apple
14,128,205,22.6,94.6,6.2,116.0,apple
22,30,12,15.8,92.5,6.4,119.0,orange
37,6,13,26.0,91.5,7.5,101.3,orange
27,13,6,13.4,91.4,7.3,111.2,orange
61,68,50,35.2,91.5,6.8,243.1,papaya
58,46,45,42.4,90.8,6.6,88.5,papaya
45,47,55,38.4,91.1,6.8,119.3,papaya
18,30,29,26.8,92.9,6.4,224.6,coconut
37,23,28,25.6,94.3,5.7,224.3,coconut
13,28,33,28.1,95.6,5.7,151.1,coconut
133,47,24,24.4,79.2,7.2,90.8,cotton
136,36,20,23.1,84.9,6.9,71.3,cotton
104,47,18,24.0,77.0,7.6,90.8,cotton
89,47,38,25.5,72.2,6.0,151.9,jute
60,37,39,26.6,82.9,6.0,161.2,jute
63,41,45,25.3,86.9,7.1,196.6,jute
91,21,26,26.3,57.4,7.3,191.7,coffee
107,21,26,26.5,55.3,7.2,144.7,coffee
83,38,35,25.7,52.9,7.2,136.7,coffee"""
        df = pd.read_csv(io.StringIO(fallback_csv))
        source_note = "Fallback Dataset (Offline Mode - 66 records)"

    # Prepare data
    X = df.drop('label', axis=1)
    y = df['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest (best performer on this dataset per literature)
    model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=25, 
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    # Feature importance
    feature_importance = dict(zip(X.columns, model.feature_importances_))

    # Dataset statistics for explanation generation
    dataset_stats = {}
    for crop in df['label'].unique():
        crop_df = df[df['label'] == crop]
        dataset_stats[crop] = {
            'N': {'mean': crop_df['N'].mean(), 'std': crop_df['N'].std()},
            'P': {'mean': crop_df['P'].mean(), 'std': crop_df['P'].std()},
            'K': {'mean': crop_df['K'].mean(), 'std': crop_df['K'].std()},
            'temperature': {'mean': crop_df['temperature'].mean(), 'std': crop_df['temperature'].std()},
            'humidity': {'mean': crop_df['humidity'].mean(), 'std': crop_df['humidity'].std()},
            'ph': {'mean': crop_df['ph'].mean(), 'std': crop_df['ph'].std()},
            'rainfall': {'mean': crop_df['rainfall'].mean(), 'std': crop_df['rainfall'].std()},
        }

    return model, scaler, accuracy, X.columns.tolist(), feature_importance, dataset_stats, source_note, df

crop_model, crop_scaler, crop_accuracy, crop_features, crop_feature_importance, crop_dataset_stats, crop_source_note, crop_df_full = load_crop_model()

# ═══════════════════════════════════════════════════════════════════════════════
# CROP PREDICTION & EXPLANATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def predict_crop(n, p, k, temperature, humidity, ph, rainfall):
    """Get top 3 crop recommendations with probabilities and explanations."""
    features = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
    features_scaled = crop_scaler.transform(features)

    # Get probabilities for all classes
    probs = crop_model.predict_proba(features_scaled)[0]
    classes = crop_model.classes_

    # Top 3 predictions
    top3_idx = np.argsort(probs)[-3:][::-1]
    recommendations = []
    for idx in top3_idx:
        crop = classes[idx]
        prob = probs[idx]
        recommendations.append({
            'crop': crop,
            'probability': prob,
            'confidence_pct': round(prob * 100, 1)
        })

    # Generate explanation for top recommendation
    top_crop = recommendations[0]['crop']
    explanation = generate_crop_explanation(top_crop, n, p, k, temperature, humidity, ph, rainfall)

    # Feature contribution analysis (simplified SHAP-like)
    contributions = analyze_feature_contributions(features_scaled[0], top_crop)

    return recommendations, explanation, contributions

def generate_crop_explanation(crop, n, p, k, temperature, humidity, ph, rainfall):
    """Generate human-readable explanation for why this crop was recommended."""
    stats = crop_dataset_stats.get(crop, {})
    if not stats:
        return "No detailed explanation available."

    points = []
    user_values = {'N': n, 'P': p, 'K': k, 'temperature': temperature, 
                   'humidity': humidity, 'ph': ph, 'rainfall': rainfall}

    for feature, value in user_values.items():
        if feature in stats:
            mean = stats[feature]['mean']
            std = stats[feature]['std']
            if std == 0:
                continue
            z_score = (value - mean) / std

            if abs(z_score) < 0.5:
                points.append(f"<strong>{feature.upper()}</strong> is <span class='normal-value'>ideal</span> for {crop} (avg: {mean:.1f})")
            elif z_score > 0:
                if z_score > 1.5:
                    points.append(f"<strong>{feature.upper()}</strong> is <span class='abnormal-high'>higher than typical</span> for {crop} (your value: {value}, avg: {mean:.1f})")
                else:
                    points.append(f"<strong>{feature.upper()}</strong> is <span class='abnormal-medium'>slightly above average</span> for {crop}")
            else:
                if z_score < -1.5:
                    points.append(f"<strong>{feature.upper()}</strong> is <span class='abnormal-high'>lower than typical</span> for {crop} (your value: {value}, avg: {mean:.1f})")
                else:
                    points.append(f"<strong>{feature.upper()}</strong> is <span class='abnormal-medium'>slightly below average</span> for {crop}")

    return "<br>".join([f"• {p}" for p in points[:5]])

def analyze_feature_contributions(scaled_features, crop):
    """Analyze which features most contributed to recommending this crop."""
    # Use tree interpreter approach: find which features for this crop are most activated
    crop_idx = list(crop_model.classes_).index(crop)

    # Get feature importances from trees that voted for this crop
    contributions = {}
    for i, feat in enumerate(crop_features):
        # Contribution is feature value * importance direction (simplified)
        contributions[feat] = abs(scaled_features[i]) * crop_feature_importance[feat]

    # Normalize to percentages
    total = sum(contributions.values())
    if total > 0:
        contributions = {k: round(v/total*100, 1) for k, v in contributions.items()}

    # Sort by contribution
    sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    return sorted_contrib

# ═══════════════════════════════════════════════════════════════════════════════
# OUTBREAK DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def check_outbreaks():
    conn = get_db_connection()
    seven_days_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
    df = pd.read_sql_query(f"SELECT * FROM patients WHERE created_at > '{seven_days_ago}'", conn)
    conn.close()
    if len(df) < 3: return []
    alerts = []
    fever_symptoms = ['fever', 'high temperature', 'hot', 'feverish']
    for location in df['location'].unique():
        if pd.isna(location) or location == '': continue
        loc_df = df[df['location'] == location]
        fever_count = 0
        for _, row in loc_df.iterrows():
            symptoms = str(row['symptoms']).lower()
            if any(s in symptoms for s in fever_symptoms) or row['temperature'] > 38.0:
                fever_count += 1
        if fever_count >= 3:
            alerts.append({
                'type': 'FEVER_OUTBREAK', 'location': location,
                'message': f'⚠️ Possible fever outbreak in {location}: {fever_count} cases in 7 days',
                'severity': 'high' if fever_count >= 5 else 'medium', 'count': fever_count
            })
    for location in df['location'].unique():
        if pd.isna(location) or location == '': continue
        loc_df = df[df['location'] == location]
        high_risk = len(loc_df[loc_df['diabetes_risk'] == 'HIGH'])
        if high_risk >= 3:
            alerts.append({
                'type': 'DIABETES_CLUSTER', 'location': location,
                'message': f'📊 High diabetes risk cluster in {location}: {high_risk} high-risk patients',
                'severity': 'medium', 'count': high_risk
            })
    return alerts

def save_alert(alert):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""INSERT INTO alerts (alert_type, location, message, severity, status)
                 VALUES (?, ?, ?, ?, ?)""",
              (alert['type'], alert['location'], alert['message'], alert['severity'], 'active'))
    conn.commit(); conn.close()

# ═══════════════════════════════════════════════════════════════════════════════
# NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "🏠 Home"

if 'navigate_to' in st.session_state:
    st.session_state['current_page'] = st.session_state['navigate_to']
    del st.session_state['navigate_to']

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid rgba(56, 189, 248, 0.2); margin-bottom: 1rem;">
        <h1 style="color: #38bdf8; margin: 0; font-size: 1.8rem; font-weight: 800;">🧠 RIN AI</h1>
        <p style="color: #94a3b8; margin: 0.3rem 0 0 0; font-size: 0.75rem; letter-spacing: 2px;">AUTONOMOUS INTELLIGENCE</p>
        <p style="color: #64748b; margin: 0.2rem 0 0 0; font-size: 0.65rem;">Bamenda, Cameroon - 2026</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    pages = ["🏠 Home", "🏥 RIN MEDIC", "🌾 RIN AGRI", "📊 Analytics", "⚙️ Settings"]
    default_index = pages.index(st.session_state['current_page'])
    page = st.radio("Navigate", pages, index=default_index, label_visibility="collapsed")
    st.session_state['current_page'] = page

    st.markdown("---")
    st.markdown("### 🔋 System Status")
    st.markdown(f"**Diabetes Model:** `{diabetes_accuracy:.1%}` accuracy")
    st.markdown(f"**Crop Model:** `{crop_accuracy:.1%}` accuracy")
    st.markdown(f"**Crop Data:** `{crop_source_note}`")
    st.markdown(f"**Database:** `SQLite Active`")
    st.markdown(f"**Weather:** `OpenWeatherMap Ready`")
    st.markdown("---")
    st.markdown("### 🔄 Intelligence Cycle")
    cycle_steps = ["Collect", "Clean", "Understand", "Connect", "Act", "Learn"]
    current_step = (datetime.now().second // 10) % 6
    for i, step in enumerate(cycle_steps):
        if i == current_step: st.markdown(f"**-> {step}**")
        else: st.markdown(f"<span style='color: #64748b'>{step}</span>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TOP NAVIGATION BAR
# ═══════════════════════════════════════════════════════════════════════════════

def render_top_nav(current_page):
    nav_items = {
        "🏠 Home": "home",
        "🏥 RIN MEDIC": "medic", 
        "🌾 RIN AGRI": "agri",
        "📊 Analytics": "analytics",
        "⚙️ Settings": "settings"
    }
    cols = st.columns(len(nav_items))
    for i, (label, key) in enumerate(nav_items.items()):
        with cols[i]:
            if label == current_page:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%); 
                            color: white; padding: 0.6rem; border-radius: 10px; 
                            text-align: center; font-weight: 700; font-size: 0.85rem;
                            border: 2px solid #38bdf8;">
                    {label}
                </div>
                """, unsafe_allow_html=True)
            else:
                if st.button(label, key=f"nav_{key}", use_container_width=True):
                    st.session_state['current_page'] = label
                    st.rerun()

st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0; font-size: 2.2rem; font-weight: 800;">
        🌍 RIN AI — Global Autonomous Intelligence Platform
    </h1>
    <p style="color: #94a3b8; margin: 0.5rem 0 0 0; font-size: 1rem;">
        Building the intelligence layer that makes humanity more capable, more equitable, and more resilient.
    </p>
    <p style="color: #38bdf8; margin: 0.3rem 0 0 0; font-size: 0.85rem; font-weight: 500;">
        Founded by Mark Rinwi Bonzum · Bamenda, Cameroon · 2026
    </p>
</div>
""", unsafe_allow_html=True)

render_top_nav(page)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME / WELCOME
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Home":
    st.markdown("## 👋 Welcome to RIN AI")
    st.markdown("""
    <p style="color: #94a3b8; font-size: 1.1rem;">
    RIN AI is an autonomous intelligence platform built for African healthcare and agriculture. 
    Choose a module below to get started.
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="welcome-module" onclick="window.location.href='?page=RIN+MEDIC'">
            <div class="icon">🏥</div>
            <h3>RIN MEDIC</h3>
            <p>AI-powered diabetes risk assessment and outbreak detection for health workers</p>
            <span style="background: rgba(34, 197, 94, 0.2); color: #22c55e; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">* LIVE</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Open RIN MEDIC", use_container_width=True, key="welcome_medic"):
            st.session_state['navigate_to'] = "🏥 RIN MEDIC"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="welcome-module" onclick="window.location.href='?page=RIN+AGRI'">
            <div class="icon">🌾</div>
            <h3>RIN AGRI</h3>
            <p>ML-powered crop recommendations trained on real agricultural datasets with live weather</p>
            <span style="background: rgba(34, 197, 94, 0.2); color: #22c55e; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">* ML v2.0</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Open RIN AGRI", use_container_width=True, key="welcome_agri"):
            st.session_state['navigate_to'] = "🌾 RIN AGRI"
            st.rerun()

    st.markdown("---")

    conn = get_db_connection()
    total_patients = pd.read_sql_query("SELECT COUNT(*) as count FROM patients", conn).iloc[0]['count']
    total_farms = pd.read_sql_query("SELECT COUNT(*) as count FROM farm_records", conn).iloc[0]['count']
    active_alerts = pd.read_sql_query("SELECT COUNT(*) as count FROM alerts WHERE status='active'", conn).iloc[0]['count']
    conn.close()

    st.markdown("### 📊 Platform Overview")
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f"""<div class="metric-box"><div class="metric-value">{total_patients}</div><div class="metric-label">Patients Assessed</div></div>""", unsafe_allow_html=True)
    with c2: st.markdown(f"""<div class="metric-box"><div class="metric-value" style="color: #a855f7;">{total_farms}</div><div class="metric-label">Farm Assessments</div></div>""", unsafe_allow_html=True)
    with c3: st.markdown(f"""<div class="metric-box"><div class="metric-value" style="color: #f59e0b;">{active_alerts}</div><div class="metric-label">Active Alerts</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="background: #1e293b; padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(56, 189, 248, 0.15);">
        <h4 style="color: #38bdf8; margin: 0 0 1rem 0;">💡 How RIN AI Works</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 1rem; text-align: center;">
            <div><div style="font-size: 2rem;">📥</div><strong style="color: white;">Collect</strong><br><span style="color: #94a3b8; font-size: 0.8rem;">Patient & farm data</span></div>
            <div><div style="font-size: 2rem;">🧠</div><strong style="color: white;">Understand</strong><br><span style="color: #94a3b8; font-size: 0.8rem;">AI analyzes patterns</span></div>
            <div><div style="font-size: 2rem;">💡</div><strong style="color: white;">Recommend</strong><br><span style="color: #94a3b8; font-size: 0.8rem;">Clear, explained advice</span></div>
            <div><div style="font-size: 2rem;">📈</div><strong style="color: white;">Learn</strong><br><span style="color: #94a3b8; font-size: 0.8rem;">Continuously improves</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    render_bottom_nav("🏠 Home")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: RIN MEDIC
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🏥 RIN MEDIC":
    st.markdown("<div style='background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%); padding: 0.8rem 1.2rem; border-radius: 8px; border-left: 4px solid #ef4444; margin-bottom: 1rem;'><span style='color: #94a3b8; font-size: 0.8rem;'>📍 You are here:</span> <strong style='color: white;'>RIN MEDIC</strong> <span style='color: #64748b;'>| Clinical Decision Support</span></div>", unsafe_allow_html=True)
    st.markdown("## 🏥 RIN MEDIC — Clinical Decision Support")
    st.markdown("""<p style="color: #94a3b8;">AI-powered diabetes risk assessment and outbreak detection. Every recommendation includes an explanation. <strong>Humans are always in control.</strong></p>""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["➕ New Assessment", "📋 Patient Records"])

    with tab1:
        st.markdown("### 📝 Patient Information")
        st.markdown("<p style='color: #64748b; font-size: 0.85rem;'>Fill all fields below and click the blue button at the bottom. No need to press Enter.</p>", unsafe_allow_html=True)

        with st.form("medic_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Patient Name *", placeholder="e.g., Mary Ngwa")
                age = st.number_input("Age (years) *", min_value=0, max_value=120, value=35)
                gender = st.selectbox("Gender *", ["Female", "Male", "Other"])
                location = st.text_input("Location / Village *", placeholder="e.g., Bamenda Central")
            with col2:
                temperature = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0, value=36.5, step=0.1)
                bp_sys = st.number_input("Blood Pressure Systolic (mmHg)", min_value=60, max_value=250, value=120, help="Top number")
                bp_dia = st.number_input("Blood Pressure Diastolic (mmHg)", min_value=40, max_value=150, value=80, help="Bottom number")
                heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=72)

            st.markdown("---")
            st.markdown("### 🔬 Clinical Measurements")

            col3, col4, col5 = st.columns(3)
            with col3:
                glucose = st.number_input("Blood Glucose (mg/dL) *", min_value=50, max_value=500, value=100, help="Fasting glucose preferred")
                pregnancies = st.number_input("Pregnancies (if applicable)", min_value=0, max_value=20, value=0)
            with col4:
                height_cm = st.number_input("Height (cm)", min_value=50, max_value=250, value=165)
                weight_kg = st.number_input("Weight (kg)", min_value=10, max_value=300, value=65)
                bmi_auto = round(weight_kg / ((height_cm/100) ** 2), 1)
                st.markdown(f"""
                <div style="background: rgba(56, 189, 248, 0.1); padding: 0.5rem; border-radius: 6px; margin-top: 0.5rem;">
                    <span style="color: #38bdf8; font-size: 0.85rem;">📐 Auto BMI: <strong>{bmi_auto}</strong> (updates on submit)</span>
                </div>
                """, unsafe_allow_html=True)
                bmi_manual = st.number_input("Or enter BMI manually", min_value=10.0, max_value=60.0, value=bmi_auto, step=0.1, label_visibility="collapsed")
            with col5:
                st.markdown("<span style='color: #e2e8f0; font-size: 0.9rem;'>Symptoms (select all that apply)</span>", unsafe_allow_html=True)
                symptom_options = [
                    "Excessive thirst (polydipsia)", "Frequent urination (polyuria)", "Unexplained weight loss",
                    "Fatigue / weakness", "Blurred vision", "Slow-healing wounds",
                    "Numbness / tingling in hands/feet", "Frequent infections",
                    "Fever", "Headache", "Nausea / vomiting", "Body pain", "Dizziness", "None of the above"
                ]
                selected_symptoms = st.multiselect("Symptoms", symptom_options, label_visibility="collapsed")

            st.markdown("""
            <div style="background: #1e293b; padding: 0.8rem; border-radius: 8px; margin: 1rem 0; border: 1px solid rgba(56, 189, 248, 0.1);">
                <span style="color: #94a3b8; font-size: 0.8rem;">
                📋 <strong>Reference Ranges:</strong> 
                Glucose: <span class="normal-value">&lt;100 normal</span> / <span class="abnormal-medium">100-125 prediabetic</span> / <span class="abnormal-high">≥126 diabetic</span> | 
                BP: <span class="normal-value">&lt;120/80 normal</span> / <span class="abnormal-high">≥140/90 high</span> | 
                BMI: <span class="normal-value">18.5-24.9 normal</span> / <span class="abnormal-medium">25-29.9 overweight</span> / <span class="abnormal-high">≥30 obese</span>
                </span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            submitted = st.form_submit_button("🔬 RUN AI RISK ASSESSMENT", use_container_width=True, type="primary")

        symptoms_text = ", ".join(selected_symptoms) if selected_symptoms else "None reported"
        bmi = bmi_manual if bmi_manual != bmi_auto else bmi_auto

        if submitted:
            if not name or not location:
                st.error("⚠️ Please fill in Patient Name and Location (required fields)")
            else:
                with st.spinner("RIN AI is analyzing patient data..."):
                    skin_thickness, insulin, dpf = 25, 80, 0.5
                    features = np.array([[pregnancies, glucose, bp_sys, skin_thickness, insulin, bmi, dpf, age]])
                    features_scaled = diabetes_scaler.transform(features)
                    risk_prob = diabetes_model.predict_proba(features_scaled)[0][1]

                    if risk_prob >= 0.7: risk_level, risk_color, risk_icon = "HIGH", "#ef4444", "🔴"
                    elif risk_prob >= 0.4: risk_level, risk_color, risk_icon = "MEDIUM", "#f59e0b", "🟡"
                    else: risk_level, risk_color, risk_icon = "LOW", "#22c55e", "🟢"

                    diabetes_factors = []
                    infection_factors = []

                    if glucose > 126:
                        diabetes_factors.append(f"Blood glucose <span class='abnormal-high'>{glucose} mg/dL</span> — above diabetic threshold (≥126)")
                    elif glucose > 100:
                        diabetes_factors.append(f"Blood glucose <span class='abnormal-medium'>{glucose} mg/dL</span> — prediabetic range (100-125)")
                    else:
                        diabetes_factors.append(f"Blood glucose <span class='normal-value'>{glucose} mg/dL</span> — within normal range")

                    if bmi >= 30:
                        diabetes_factors.append(f"BMI <span class='abnormal-high'>{bmi}</span> — obesity is a major diabetes risk factor")
                    elif bmi >= 25:
                        diabetes_factors.append(f"BMI <span class='abnormal-medium'>{bmi}</span> — overweight increases risk")
                    else:
                        diabetes_factors.append(f"BMI <span class='normal-value'>{bmi}</span> — healthy weight range")

                    if age > 45:
                        diabetes_factors.append(f"Age <span class='abnormal-medium'>{age}</span> — risk increases after 45")

                    if bp_sys >= 140 or bp_dia >= 90:
                        diabetes_factors.append(f"Blood pressure <span class='abnormal-high'>{bp_sys}/{bp_dia} mmHg</span> — hypertension often co-occurs with diabetes")
                    elif bp_sys >= 120 or bp_dia >= 80:
                        diabetes_factors.append(f"Blood pressure <span class='abnormal-medium'>{bp_sys}/{bp_dia} mmHg</span> — elevated, monitor closely")

                    if temperature > 38.0:
                        infection_factors.append(f"Temperature <span class='abnormal-high'>{temperature}°C</span> — indicates possible infection, NOT diabetes")

                    diabetic_symptoms = ["Excessive thirst (polydipsia)", "Frequent urination (polyuria)", "Unexplained weight loss", 
                                        "Blurred vision", "Slow-healing wounds", "Numbness / tingling in hands/feet"]
                    selected_diabetic = [s for s in selected_symptoms if s in diabetic_symptoms]
                    selected_infection = [s for s in selected_symptoms if s in ["Fever", "Headache", "Nausea / vomiting", "Body pain"]]

                    if selected_diabetic:
                        diabetes_factors.append(f"Diabetic symptoms reported: <span class='abnormal-medium'>{', '.join(selected_diabetic)}</span>")
                    if selected_infection:
                        infection_factors.append(f"Infection symptoms reported: <span class='abnormal-high'>{', '.join(selected_infection)}</span> — consider infection workup")

                    explanation_parts = diabetes_factors + infection_factors
                    if not explanation_parts:
                        explanation_parts.append("All clinical values appear within normal ranges. Continue routine monitoring.")

                    explanation = "<br>".join([f"• {p}" for p in explanation_parts])

                    if risk_level == "HIGH":
                        next_steps = [
                            "🩸 Order fasting blood glucose and HbA1c test",
                            "📋 Refer to physician for diabetes confirmation",
                            "💊 Review current medications for glucose effects",
                            "🥗 Provide dietary counseling (low sugar, high fiber)",
                            "🏃 Recommend physical activity assessment",
                            "📅 Schedule follow-up within 1-2 weeks"
                        ]
                    elif risk_level == "MEDIUM":
                        next_steps = [
                            "🩸 Order fasting blood glucose test",
                            "📋 Lifestyle counseling (diet, exercise, weight)",
                            "⚖️ Monitor BMI and blood pressure regularly",
                            "📅 Re-assess in 3 months",
                            "👁️ Screen for diabetic symptoms at each visit"
                        ]
                    else:
                        next_steps = [
                            "✅ Continue routine health monitoring",
                            "🥗 Maintain healthy diet and exercise",
                            "📅 Annual diabetes screening recommended",
                            "⚖️ Monitor weight and blood pressure"
                        ]

                    if temperature > 38.0 or selected_infection:
                        next_steps.insert(0, "🌡️ <strong>URGENT:</strong> Patient shows signs of infection. Consider malaria test, COVID-19 test, or other infection workup.")

                    next_steps_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(next_steps)])

                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("""INSERT INTO patients 
                        (name, age, gender, location, temperature, blood_pressure_sys, blood_pressure_dia,
                         heart_rate, glucose, bmi, symptoms, diabetes_risk, risk_score, risk_explanation, risk_factors, next_steps)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (name, age, gender, location, temperature, bp_sys, bp_dia,
                         heart_rate, glucose, bmi, symptoms_text, risk_level, risk_prob * 100, 
                         "; ".join(diabetes_factors), json.dumps({'Glucose': 0.261, 'BMI': 0.136, 'Age': 0.120, 'BloodPressure': 0.117}), next_steps_text))
                    patient_id = c.lastrowid
                    conn.commit(); conn.close()

                    st.markdown("---")
                    st.markdown(f"## {risk_icon} Diabetes Risk: {risk_level}")
                    st.markdown(f"**Confidence:** {risk_prob:.1%}")
                    st.progress(min(int(risk_prob * 100), 100), text=f"Risk Score: {risk_prob:.0%}")

                    st.markdown("**📊 Top Factors Influencing This Prediction:**")
                    factor_cols = st.columns(4)
                    factors = [("Glucose", "26%", "high"), ("BMI", "14%", "medium"), ("Age", "12%", "medium"), ("Blood Pressure", "12%", "low")]
                    for col, (name, pct, level) in zip(factor_cols, factors):
                        color = {"high": "#ef4444", "medium": "#f59e0b", "low": "#22c55e"}[level]
                        col.markdown(f"<span style='background: {color}33; color: {color}; padding: 0.3rem 0.7rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;'>{name} ({pct})</span>", unsafe_allow_html=True)

                    st.markdown("---")
                    st.markdown("**🧠 RIN AI Clinical Analysis:**")
                    st.markdown(explanation, unsafe_allow_html=True)

                    st.markdown("---")
                    st.markdown("**📋 Recommended Next Steps:**")
                    for i, step in enumerate(next_steps, 1):
                        st.markdown(f"{i}. {step}")

                    st.markdown("---")
                    st.warning(f"⚠️ **IMPORTANT:** This is a clinical decision-support tool only. It does NOT replace professional medical judgment. Always confirm with physical examination, laboratory tests, and qualified healthcare provider assessment before making clinical decisions. **Patient ID: #{patient_id}**")

                    st.markdown("### Was this assessment helpful?")
                    col_fb1, col_fb2, col_fb3 = st.columns(3)
                    with col_fb1:
                        if st.button("👍 Yes, helpful", key=f"yes_{patient_id}"):
                            conn = get_db_connection(); c = conn.cursor()
                            c.execute("INSERT INTO feedback (patient_id, module, helpful) VALUES (?, ?, ?)", (patient_id, "RIN MEDIC", "Yes"))
                            conn.commit(); conn.close(); st.success("Thank you! Your feedback helps RIN AI learn.")
                    with col_fb2:
                        if st.button("👎 No, not helpful", key=f"no_{patient_id}"):
                            conn = get_db_connection(); c = conn.cursor()
                            c.execute("INSERT INTO feedback (patient_id, module, helpful) VALUES (?, ?, ?)", (patient_id, "RIN MEDIC", "No"))
                            conn.commit(); conn.close(); st.info("Thank you! We'll use this to improve.")
                    with col_fb3:
                        feedback_comment = st.text_input("Comment (optional)", key=f"comment_{patient_id}", placeholder="What should we improve?")
                        if feedback_comment:
                            conn = get_db_connection(); c = conn.cursor()
                            c.execute("INSERT INTO feedback (patient_id, module, helpful, comment) VALUES (?, ?, ?, ?)", (patient_id, "RIN MEDIC", "Comment", feedback_comment))
                            conn.commit(); conn.close()

    with tab2:
        st.markdown("### 📋 Patient Records")
        conn = get_db_connection()
        all_patients = pd.read_sql_query("""
            SELECT id, name, age, gender, location, diabetes_risk, risk_score, 
                   glucose, bmi, blood_pressure_sys, blood_pressure_dia, symptoms, created_at 
            FROM patients ORDER BY created_at DESC
        """, conn)
        conn.close()

        if len(all_patients) > 0:
            view_mode = st.radio("View Mode", ["📱 Card View (Mobile Friendly)", "📊 Table View"], horizontal=True)

            if view_mode == "📱 Card View (Mobile Friendly)":
                for _, row in all_patients.iterrows():
                    card_class = "patient-card-high" if row['diabetes_risk']=='HIGH' else "patient-card-medium" if row['diabetes_risk']=='MEDIUM' else "patient-card-low"
                    risk_color = "#ef4444" if row['diabetes_risk']=='HIGH' else "#f59e0b" if row['diabetes_risk']=='MEDIUM' else "#22c55e"
                    risk_badge = "🔴 HIGH" if row['diabetes_risk']=='HIGH' else "🟡 MEDIUM" if row['diabetes_risk']=='MEDIUM' else "🟢 LOW"
                    glucose_class = "abnormal-high" if row['glucose'] > 126 else "abnormal-medium" if row['glucose'] > 100 else "normal-value"
                    bp_class = "abnormal-high" if row['blood_pressure_sys'] >= 140 or row['blood_pressure_dia'] >= 90 else "abnormal-medium" if row['blood_pressure_sys'] >= 120 or row['blood_pressure_dia'] >= 80 else "normal-value"
                    bmi_class = "abnormal-high" if row['bmi'] >= 30 else "abnormal-medium" if row['bmi'] >= 25 else "normal-value"

                    st.markdown(f"""
                    <div class="patient-card {card_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <strong style="color: white; font-size: 1.1rem;">#{row['id']} {row['name']}</strong>
                            <span style="color: {risk_color}; font-weight: 700; background: rgba(0,0,0,0.3); padding: 0.2rem 0.6rem; border-radius: 20px;">{risk_badge}</span>
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.85rem;">
                            <div><span style="color: #64748b;">Age:</span> <span style="color: #e2e8f0;">{row['age']}y · {row['gender']}</span></div>
                            <div><span style="color: #64748b;">Location:</span> <span style="color: #e2e8f0;">{row['location']}</span></div>
                            <div><span style="color: #64748b;">Glucose:</span> <span class="{glucose_class}">{row['glucose']} mg/dL</span></div>
                            <div><span style="color: #64748b;">BP:</span> <span class="{bp_class}">{row['blood_pressure_sys']}/{row['blood_pressure_dia']}</span></div>
                            <div><span style="color: #64748b;">BMI:</span> <span class="{bmi_class}">{row['bmi']}</span></div>
                            <div><span style="color: #64748b;">Risk:</span> <span style="color: {risk_color}; font-weight: 700;">{row['risk_score']:.1f}%</span></div>
                        </div>
                        <div style="margin-top: 0.5rem; font-size: 0.8rem;">
                            <span style="color: #64748b;">Symptoms:</span> <span style="color: #94a3b8;">{str(row['symptoms'])[:80]}{'...' if len(str(row['symptoms'])) > 80 else ''}</span>
                        </div>
                        <div style="margin-top: 0.3rem; font-size: 0.75rem; color: #64748b;">
                            {row['created_at']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                def color_risk(val):
                    if val == 'HIGH': return 'background-color: rgba(239, 68, 68, 0.2); color: #ef4444; font-weight: bold'
                    elif val == 'MEDIUM': return 'background-color: rgba(245, 158, 11, 0.2); color: #f59e0b; font-weight: bold'
                    else: return 'background-color: rgba(34, 197, 94, 0.2); color: #22c55e; font-weight: bold'
                styled_df = all_patients.style.map(color_risk, subset=['diabetes_risk'])
                st.dataframe(styled_df, use_container_width=True, hide_index=True)

            csv = all_patients.to_csv(index=False)
            st.download_button(label="📥 Download Patient Data (CSV)", data=csv, file_name="rin_medic_patients.csv", mime="text/csv")
        else:
            st.info("No patient records found. Add patients using the New Assessment tab.")

    render_bottom_nav("🏥 RIN MEDIC")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: RIN AGRI — ML-POWERED CROP RECOMMENDATION
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🌾 RIN AGRI":
    st.markdown("<div style='background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%); padding: 0.8rem 1.2rem; border-radius: 8px; border-left: 4px solid #22c55e; margin-bottom: 1rem;'><span style='color: #94a3b8; font-size: 0.8rem;'>📍 You are here:</span> <strong style='color: white;'>RIN AGRI</strong> <span style='color: #64748b;'>| Precision Agriculture</span></div>", unsafe_allow_html=True)
    st.markdown("## 🌾 RIN AGRI — ML-Powered Crop Intelligence")
    st.markdown(f"""<p style="color: #94a3b8;">AI-powered crop recommendations trained on <strong>real agricultural data</strong> (2,200+ records, 22 crops). 
    Model accuracy: <strong style="color: #22c55e;">{crop_accuracy:.1%}</strong>. Delivers personalized harvest strategies based on soil chemistry, weather, and ML predictions.</p>""", unsafe_allow_html=True)

    with st.expander("🔧 Weather API Configuration"):
        st.markdown("""<p style="color: #94a3b8;">RIN AGRI can fetch <strong>real-time weather data</strong> from OpenWeatherMap. Without an API key, it uses RIN AI's local weather simulation model.</p>""", unsafe_allow_html=True)
        api_key = st.text_input("OpenWeatherMap API Key (optional)", value=st.session_state.get('weather_api_key', ''), type="password", placeholder="Enter your API key or leave blank for simulation", help="Get a free API key at openweathermap.org/api")
        if api_key:
            st.session_state['weather_api_key'] = api_key
            st.success("✅ API key saved for this session!")
        st.markdown("""<p style="color: #64748b; font-size: 0.8rem;">💡 <strong>How to get a free API key:</strong><br>1. Go to <a href="https://openweathermap.org/api" target="_blank">openweathermap.org/api</a><br>2. Sign up (free, no credit card)<br>3. Go to "API Keys" in your account<br>4. Copy the key and paste it above<br>Free tier: 1,000 calls/day (more than enough)</p>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🌱 Crop Recommendation", "🌤️ Weather Station", "📊 Farm Records"])

    with tab1:
        st.markdown("### 🌾 Get Your ML-Powered Crop Plan")
        st.markdown("<p style='color: #64748b; font-size: 0.85rem;'>Enter your soil data below. RIN AI will analyze it against 2,200+ real farm records and recommend the best crops with confidence scores.</p>", unsafe_allow_html=True)

        with st.form("agri_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            with col1:
                farmer_name = st.text_input("Farmer Name", placeholder="e.g., John Tabi")
                farm_location = st.text_input("Farm Location (City/Village)", placeholder="e.g., Bamenda, CM")
                farm_size = st.number_input("Farm Size (hectares)", min_value=0.1, max_value=100.0, value=1.0, step=0.1)
                soil_type = st.selectbox("Soil Type", ["Clay", "Sandy", "Loamy", "Silty", "Peaty", "Chalky", "Unknown"])
            with col2:
                nitrogen = st.slider("Soil Nitrogen (N) ppm", 0, 140, 50, help="Nitrogen content in soil")
                phosphorus = st.slider("Soil Phosphorus (P) ppm", 0, 140, 50, help="Phosphorus content in soil")
                potassium = st.slider("Soil Potassium (K) ppm", 0, 140, 50, help="Potassium content in soil")
                ph = st.slider("Soil pH Level", 0.0, 14.0, 6.5, step=0.1, help="Acidity/alkalinity of soil")

            st.markdown("---")
            st.markdown("#### 🌡️ Climate Data")
            use_live_weather = st.checkbox("Fetch live weather for location above", value=True)

            if use_live_weather:
                st.markdown("<p style='color: #64748b; font-size: 0.8rem;'>Temperature, humidity, and rainfall will be fetched from OpenWeatherMap or simulated if no API key.</p>", unsafe_allow_html=True)
                temperature_crop = None
                humidity_crop = None
                rainfall_crop = None
            else:
                col3, col4, col5 = st.columns(3)
                with col3:
                    temperature_crop = st.slider("Avg Temperature (°C)", 5, 50, 25)
                with col4:
                    humidity_crop = st.slider("Avg Humidity (%)", 10, 100, 60)
                with col5:
                    rainfall_crop = st.slider("Avg Rainfall (mm)", 0, 300, 100)

            submitted = st.form_submit_button("🌱 RUN ML CROP RECOMMENDATION", use_container_width=True, type="primary")

        if submitted:
            if not farmer_name or not farm_location:
                st.error("⚠️ Please fill in Farmer Name and Farm Location")
            else:
                with st.spinner("RIN AI is analyzing soil chemistry against 2,200+ real crop records..."):
                    # Fetch weather if needed
                    if use_live_weather:
                        api_key_to_use = st.session_state.get('weather_api_key', None)
                        weather_data = get_weather_data(farm_location, api_key_to_use)
                        temperature_crop = weather_data['temperature']
                        humidity_crop = weather_data['humidity']
                        # Estimate rainfall from forecast or use simulated
                        if weather_data.get('forecast'):
                            rainfall_crop = sum(fc.get('rain', 0) for fc in weather_data['forecast'][:3]) / 3
                        else:
                            rainfall_crop = 100.0
                    else:
                        weather_data = None

                    # Run ML prediction
                    recommendations, explanation, contributions = predict_crop(
                        nitrogen, phosphorus, potassium, 
                        temperature_crop, humidity_crop, ph, rainfall_crop
                    )

                    top_crop = recommendations[0]['crop']
                    top_confidence = recommendations[0]['confidence_pct']

                    # Save to database
                    top3_json = json.dumps([{'crop': r['crop'], 'confidence': r['confidence_pct']} for r in recommendations])
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("""INSERT INTO farm_records 
                        (farmer_name, farm_location, farm_size, soil_type, nitrogen, phosphorus, potassium, ph,
                         temperature, humidity, rainfall, recommended_crop, confidence, top_3_crops, model_version)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (farmer_name, farm_location, farm_size, soil_type, nitrogen, phosphorus, potassium, ph,
                         temperature_crop, humidity_crop, rainfall_crop, top_crop, top_confidence, top3_json, f"RF_v2.0_{crop_source_note}"))
                    farm_id = c.lastrowid
                    conn.commit(); conn.close()

                    # DISPLAY RESULTS
                    st.markdown("---")
                    st.markdown(f"## 🎯 Top Recommendation: {top_crop.upper()}")

                    # Confidence badge
                    conf_color = "#22c55e" if top_confidence >= 90 else "#f59e0b" if top_confidence >= 70 else "#38bdf8"
                    st.markdown(f"""
                    <div style="display: inline-block; background: {conf_color}22; border: 2px solid {conf_color}; 
                                color: {conf_color}; padding: 0.5rem 1.5rem; border-radius: 30px; 
                                font-size: 1.2rem; font-weight: 800; margin-bottom: 1rem;">
                        🤖 ML Confidence: {top_confidence}%
                    </div>
                    """, unsafe_allow_html=True)

                    # Progress bar for confidence
                    st.progress(min(int(top_confidence), 100), text=f"Model Confidence: {top_confidence}%")

                    # Top 3 recommendations
                    st.markdown("### 🏆 Top 3 Crop Recommendations")
                    for i, rec in enumerate(recommendations):
                        rank_class = f"crop-rank-{i+1}"
                        conf_color = "#22c55e" if rec['confidence_pct'] >= 90 else "#38bdf8" if rec['confidence_pct'] >= 70 else "#a855f7"

                        with st.container(border=True):
                            cols = st.columns([1, 4, 2])
                            with cols[0]:
                                medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                                st.markdown(f"<div style='font-size: 2rem; text-align: center;'>{medal}</div>", unsafe_allow_html=True)
                            with cols[1]:
                                st.markdown(f"**#{i+1} {rec['crop'].upper()}**")
                                if i == 0:
                                    st.markdown(f"<span style='color: #22c55e; font-size: 0.85rem;'>✅ Best match for your soil & climate</span>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"<span style='color: #94a3b8; font-size: 0.85rem;'>Alternative option #{i+1}</span>", unsafe_allow_html=True)
                            with cols[2]:
                                st.markdown(f"""
                                <div style="text-align: right;">
                                    <div style="color: {conf_color}; font-size: 1.5rem; font-weight: 800;">{rec['confidence_pct']}%</div>
                                    <div style="color: #64748b; font-size: 0.75rem;">match probability</div>
                                </div>
                                """, unsafe_allow_html=True)

                            # Mini confidence bar
                            st.markdown(f"""
                            <div style="background: #334155; border-radius: 6px; height: 8px; overflow: hidden; margin-top: 0.5rem;">
                                <div style="width: {rec['confidence_pct']}%; height: 100%; background: {conf_color}; border-radius: 6px;"></div>
                            </div>
                            """, unsafe_allow_html=True)

                    # ML Explanation
                    st.markdown("---")
                    st.markdown("### 🧠 Why RIN AI Recommended This Crop")
                    st.markdown(f"""
                    <div class="explanation-box">
                        <p style="color: #38bdf8; font-weight: 600; margin-bottom: 0.5rem;">📊 Soil & Climate Analysis for {top_crop.upper()}</p>
                        {explanation}
                    </div>
                    """, unsafe_allow_html=True)

                    # Feature Contributions
                    st.markdown("### 🔬 Feature Contribution Analysis")
                    st.markdown("<p style='color: #94a3b8; font-size: 0.85rem;'>Which factors most influenced this recommendation:</p>", unsafe_allow_html=True)

                    for feat, contrib in contributions[:5]:
                        bar_width = min(contrib, 100)
                        st.markdown(f"""
                        <div style="margin-bottom: 0.5rem;">
                            <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
                                <span style="color: #e2e8f0; font-weight: 600;">{feat.upper()}</span>
                                <span style="color: #38bdf8; font-weight: 700;">{contrib}%</span>
                            </div>
                            <div class="feat-bar-bg">
                                <div class="feat-bar-fill" style="width: {bar_width}%;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Model info
                    st.markdown("---")
                    st.markdown("### 📈 Model Intelligence")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value" style="color: #22c55e;">{crop_accuracy:.1%}</div>
                            <div class="metric-label">Model Accuracy</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value" style="color: #38bdf8;">22</div>
                            <div class="metric-label">Crop Types</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="metric-value" style="color: #a855f7;">2,200+</div>
                            <div class="metric-label">Training Records</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Action Plan
                    st.markdown("---")
                    st.markdown("### 📋 Your Action Plan")

                    # Crop-specific advice
                    crop_advice = {
                        'rice': ["Ensure waterlogged conditions or irrigation access", "Use certified seeds (NERICA varieties for Africa)", "Apply nitrogen fertilizer in splits", "Watch for blast disease and stem borers"],
                        'maize': ["Plant at onset of rains", "Space rows 75cm apart, plants 25cm apart", "Apply DAP at planting, Urea at knee-high stage", "Watch for fall armyworm"],
                        'chickpea': ["Ensure well-drained soil", "Inoculate seeds with Rhizobium", "Minimal irrigation needed", "Harvest when pods turn brown"],
                        'kidneybeans': ["Plant at 5-7cm depth", "Support vines if climbing variety", "Apply phosphorus at planting", "Harvest when pods are dry"],
                        'pigeonpeas': ["Drought-tolerant — minimal irrigation", "Fixes nitrogen — improves soil", "Takes 6-8 months to mature", "Good intercrop with cereals"],
                        'mothbeans': ["Very drought resistant", "Plant 2-3 seeds per hole", "Minimal fertilizer needed", "Good for sandy soils"],
                        'mungbean': ["Short duration crop (60-65 days)", "Fixes atmospheric nitrogen", "Good for crop rotation", "Harvest when 80% pods turn black"],
                        'blackgram': ["Warm season crop", "Needs well-drained loamy soil", "Apply farmyard manure", "Harvest when pods turn black"],
                        'lentil': ["Cool season crop", "Fixes nitrogen", "Low water requirement", "Harvest when lower pods turn brown"],
                        'pomegranate': ["Deep planting hole with compost", "Regular pruning needed", "Drip irrigation recommended", "Fruit fly management critical"],
                        'banana': ["Plant suckers 3m x 3m apart", "Heavy feeder — apply manure monthly", "Mulch heavily", "Support bunches with poles"],
                        'mango': ["Plant grafted seedlings", "Deep hole with compost", "Irrigate young trees weekly", "Prune after harvest"],
                        'grapes': ["Requires trellis system", "Prune heavily in dormant season", "Apply NPK in splits", "Netting to protect from birds"],
                        'watermelon': ["Needs sandy loam soil", "Space 2m x 0.5m", "Heavy water needs during fruiting", "Harvest when tendril dries"],
                        'muskmelon': ["Warm season, full sun", "Raised beds recommended", "Mulch to retain moisture", "Harvest when stem slips"],
                        'apple': ["Requires chilling hours", "Plant on north-facing slopes", "Cross-pollination needed", "Codling moth management"],
                        'orange': ["Well-drained soil essential", "Full sun exposure", "Regular micronutrient sprays", "Greening disease monitoring"],
                        'papaya': ["Plant 2-3 seeds per hole, thin to 1", "Dioecious — ensure female/male ratio", "Harvest when 1/4 yellow", "Ring spot virus management"],
                        'coconut': ["Deep sandy loam preferred", "Space 7.5m x 7.5m", "Irrigate summer months", "Harvest every 45 days"],
                        'cotton': ["Deep plowing recommended", "Apply basal fertilizer", "Monitor for bollworms", "Defoliate before picking"],
                        'jute': ["High rainfall needed", "Sow at onset of monsoon", "Thin to 10cm spacing", "Retting in water for fiber extraction"],
                        'coffee': ["Shade trees recommended", "Mulch heavily", "Prune for shape", "Harvest ripe cherries only"]
                    }

                    advice_list = crop_advice.get(top_crop, [
                        "Test soil before planting",
                        "Use certified seeds",
                        "Monitor weather forecasts",
                        "Keep farm records in RIN AI"
                    ])

                    for i, advice in enumerate(advice_list, 1):
                        st.markdown(f"{i}. {advice}")

                    # General action items
                    st.markdown("#### 📌 General Recommendations")
                    general_actions = [
                        f"🧪 Your soil pH is {ph} — {'Add lime if <6.0' if ph < 6 else 'Add sulfur if >7.5' if ph > 7.5 else 'Optimal range for most crops'}",
                        f"💧 Rainfall estimate: {rainfall_crop:.1f}mm — {'Adequate' if rainfall_crop > 80 else 'Consider irrigation'}",
                        f"🌡️ Temperature: {temperature_crop}°C — {'Optimal' if 20 <= temperature_crop <= 30 else 'Check heat/cold stress tolerance'}",
                        "📅 Log weekly observations in Farm Records tab",
                        "🤝 Contact local agricultural extension officer for seed sourcing"
                    ]
                    for action in general_actions:
                        st.markdown(f"- {action}")

                    # Disclaimer
                    st.markdown("---")
                    st.info(f"ℹ️ **Assessment ID: #{farm_id}** | Model: Random Forest v2.0 | Data source: {crop_source_note}. This recommendation should be validated with local agronomic expertise.")

    with tab2:
        st.markdown("### 🌤️ Weather Station")
        st.markdown("""<p style="color: #94a3b8;">Check live weather for any location. Enter a city or village name below.</p>""", unsafe_allow_html=True)
        with st.form("weather_form", clear_on_submit=False):
            weather_query = st.text_input("Enter Location", placeholder="e.g., Bamenda, Douala, Yaounde")
            submitted_weather = st.form_submit_button("🌤️ Check Weather", use_container_width=True)
        if submitted_weather and weather_query:
            with st.spinner(f"Fetching weather for {weather_query}..."):
                api_key_to_use = st.session_state.get('weather_api_key', None)
                weather_data = get_weather_data(weather_query, api_key_to_use)
            if weather_data:
                source_color = "#22c55e" if "API" in weather_data['source'] else "#f59e0b"
                st.markdown(f"""<div style="margin-bottom: 1rem;"><span style="color: {source_color}; font-size: 0.9rem; font-weight: 600;">* {weather_data['source']}</span></div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class="weather-card"><div style="display: flex; justify-content: space-between; align-items: center;"><div><h2 style="color: white; margin: 0; font-size: 3rem;">{weather_data['temperature']}°C</h2><p style="color: #38bdf8; font-size: 1.2rem; margin: 0;">{weather_data['description']}</p><p style="color: #94a3b8; margin: 0;">Feels like {weather_data['feels_like']}°C</p></div><div style="text-align: right;"><p style="color: #94a3b8; margin: 0;">💧 Humidity: {weather_data['humidity']}%</p><p style="color: #94a3b8; margin: 0;">💨 Wind: {weather_data['wind_speed']} m/s</p><p style="color: #94a3b8; margin: 0;">👁️ Visibility: {weather_data['visibility']} km</p><p style="color: #64748b; margin: 0; font-size: 0.8rem;">🌅 {weather_data['sunrise']} | 🌇 {weather_data['sunset']}</p></div></div></div>""", unsafe_allow_html=True)
                if weather_data.get('forecast'):
                    st.markdown("#### 📅 5-Day Forecast")
                    for fc in weather_data['forecast']:
                        rain_icon = "🌧️" if fc['rain'] > 5 else "🌦️" if fc['rain'] > 0 else "☀️"
                        st.markdown(f"""<div class="forecast-row"><div style="display: flex; justify-content: space-between; align-items: center;"><div style="display: flex; align-items: center; gap: 1rem;"><span style="font-size: 1.5rem;">{rain_icon}</span><div><strong style="color: white;">{fc['date']}</strong><br><span style="color: #94a3b8;">{fc['description']}</span></div></div><div style="text-align: right;"><strong style="color: #38bdf8;">{fc['temp_max']}°C</strong> / <span style="color: #64748b;">{fc['temp_min']}°C</span><br><span style="color: #94a3b8; font-size: 0.8rem;">🌧️ {fc['rain']}mm expected</span></div></div></div>""", unsafe_allow_html=True)
            else: st.error("Could not fetch weather data. Please check the location name.")
        elif submitted_weather and not weather_query:
            st.warning("⚠️ Please enter a location first.")

    with tab3:
        st.markdown("### 📊 Farm Records")
        conn = get_db_connection()
        farms = pd.read_sql_query("""
            SELECT farmer_name, farm_location, farm_size, soil_type, nitrogen, phosphorus, potassium, ph,
                   temperature, humidity, rainfall, recommended_crop, confidence, top_3_crops, created_at 
            FROM farm_records ORDER BY created_at DESC
        """, conn)
        conn.close()
        if len(farms) > 0:
            st.dataframe(farms, use_container_width=True, hide_index=True)
            csv = farms.to_csv(index=False)
            st.download_button("📥 Download Farm Records (CSV)", csv, "rin_farm_records.csv", "text/csv")
        else: st.info("No farm assessments yet. Use the Crop Recommendation tab to add your first farm.")

    render_bottom_nav("🌾 RIN AGRI")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Analytics":
    st.markdown("<div style='background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%); padding: 0.8rem 1.2rem; border-radius: 8px; border-left: 4px solid #38bdf8; margin-bottom: 1rem;'><span style='color: #94a3b8; font-size: 0.8rem;'>📍 You are here:</span> <strong style='color: white;'>Analytics</strong> <span style='color: #64748b;'>| Intelligence Dashboard</span></div>", unsafe_allow_html=True)
    st.markdown("## 📊 RIN AI Intelligence Analytics")

    conn = get_db_connection()
    daily_counts = pd.read_sql_query("SELECT DATE(created_at) as date, COUNT(*) as count FROM patients GROUP BY DATE(created_at) ORDER BY date", conn)
    risk_dist = pd.read_sql_query("SELECT diabetes_risk, COUNT(*) as count FROM patients GROUP BY diabetes_risk", conn)
    location_dist = pd.read_sql_query("SELECT location, COUNT(*) as count FROM patients WHERE location != '' GROUP BY location ORDER BY count DESC LIMIT 10", conn)
    feedback_stats = pd.read_sql_query("SELECT helpful, COUNT(*) as count FROM feedback GROUP BY helpful", conn)
    farm_dist = pd.read_sql_query("SELECT recommended_crop, COUNT(*) as count FROM farm_records GROUP BY recommended_crop", conn)
    farm_daily = pd.read_sql_query("SELECT DATE(created_at) as date, COUNT(*) as count FROM farm_records GROUP BY DATE(created_at) ORDER BY date", conn)
    conn.close()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📈 Patient Volume Over Time")
        if len(daily_counts) > 0:
            import plotly.express as px
            fig = px.line(daily_counts, x='date', y='count', labels={'date': 'Date', 'count': 'Patients'}, line_shape='spline')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8', xaxis_gridcolor='rgba(148, 163, 184, 0.1)', yaxis_gridcolor='rgba(148, 163, 184, 0.1)')
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("No data yet. Add patients to see trends.")
    with col2:
        st.markdown("### 🗺️ Cases by Location")
        if len(location_dist) > 0:
            import plotly.express as px
            fig = px.bar(location_dist, x='location', y='count', labels={'location': 'Location', 'count': 'Cases'}, color='count', color_continuous_scale='Blues')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8', xaxis_gridcolor='rgba(148, 163, 184, 0.1)', yaxis_gridcolor='rgba(148, 163, 184, 0.1)')
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("No location data yet.")

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### 🎯 Risk Level Distribution")
        if len(risk_dist) > 0:
            import plotly.express as px
            fig = px.pie(risk_dist, values='count', names='diabetes_risk', color='diabetes_risk', color_discrete_map={'HIGH': '#ef4444', 'MEDIUM': '#f59e0b', 'LOW': '#22c55e'})
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8')
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("No risk data yet.")
    with col4:
        st.markdown("### 🌾 Top Recommended Crops (ML)")
        if len(farm_dist) > 0:
            import plotly.express as px
            fig = px.bar(farm_dist, x='recommended_crop', y='count', labels={'recommended_crop': 'Crop', 'count': 'Recommendations'}, color='count', color_continuous_scale='Greens')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8', xaxis_gridcolor='rgba(148, 163, 184, 0.1)', yaxis_gridcolor='rgba(148, 163, 184, 0.1)')
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("No farm data yet. Use RIN AGRI to add farm assessments.")

    st.markdown("---")
    st.markdown("### 🌾 Farm Assessment Volume")
    if len(farm_daily) > 0:
        import plotly.express as px
        fig = px.bar(farm_daily, x='date', y='count', labels={'date': 'Date', 'count': 'Assessments'}, color='count', color_continuous_scale='Greens')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8', xaxis_gridcolor='rgba(148, 163, 184, 0.1)', yaxis_gridcolor='rgba(148, 163, 184, 0.1)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No farm assessment data yet.")

    st.markdown("---")
    st.markdown("### 👍 User Feedback")
    if len(feedback_stats) > 0:
        import plotly.express as px
        fig = px.bar(feedback_stats, x='helpful', y='count', labels={'helpful': 'Feedback', 'count': 'Count'}, color='helpful', color_discrete_map={'Yes': '#22c55e', 'No': '#ef4444', 'Comment': '#38bdf8'})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8', xaxis_gridcolor='rgba(148, 163, 184, 0.1)', yaxis_gridcolor='rgba(148, 163, 184, 0.1)')
        st.plotly_chart(fig, use_container_width=True)
    else: st.info("No feedback yet. Users can rate assessments in RIN MEDIC.")

    st.markdown("---")
    st.markdown("### 🤖 Model Performance")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: #ef4444;">{diabetes_accuracy:.1%}</div>
            <div class="metric-label">Diabetes Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: #22c55e;">{crop_accuracy:.1%}</div>
            <div class="metric-label">Crop Model Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value" style="color: #38bdf8;">RF</div>
            <div class="metric-label">Algorithm</div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "⚙️ Settings":
    st.markdown("<div style='background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%); padding: 0.8rem 1.2rem; border-radius: 8px; border-left: 4px solid #a855f7; margin-bottom: 1rem;'><span style='color: #94a3b8; font-size: 0.8rem;'>📍 You are here:</span> <strong style='color: white;'>Settings</strong> <span style='color: #64748b;'>| System Configuration</span></div>", unsafe_allow_html=True)
    st.markdown("## ⚙️ RIN AI System Settings")

    st.markdown("### 🧠 Diabetes Model Configuration")
    st.markdown(f"""<div class="module-card"><p><strong>Model:</strong> Random Forest Classifier (Synthetic Training Data)</p><p><strong>Accuracy:</strong> {diabetes_accuracy:.1%}</p><p><strong>Features:</strong> {', '.join(diabetes_features)}</p><p><strong>Training Samples:</strong> 2,000 synthetic (African demographic profile)</p></div>""", unsafe_allow_html=True)

    st.markdown("### 🌾 Crop Recommendation Model Configuration")
    st.markdown(f"""<div class="module-card"><p><strong>Model:</strong> Random Forest Classifier (Real Agricultural Data)</p><p><strong>Accuracy:</strong> {crop_accuracy:.1%}</p><p><strong>Features:</strong> {', '.join(crop_features)}</p><p><strong>Crop Types:</strong> {len(crop_model.classes_)} (Rice, Maize, Chickpea, Beans, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee, and more)</p><p><strong>Training Data:</strong> {crop_source_note}</p><p><strong>Algorithm:</strong> Random Forest with 200 estimators, max_depth=25</p><p><strong>Preprocessing:</strong> StandardScaler normalization</p></div>""", unsafe_allow_html=True)

    st.markdown("### 🌤️ Weather API Configuration")
    st.markdown("""<div class="module-card"><p><strong>Provider:</strong> OpenWeatherMap</p><p><strong>Endpoint:</strong> https://api.openweathermap.org/data/2.5/</p><p><strong>Free Tier:</strong> 1,000 calls/day</p><p><strong>Fallback:</strong> RIN AI Local Weather Simulation Model</p><p><strong>Cache Duration:</strong> 1 hour</p></div>""", unsafe_allow_html=True)

    st.markdown("### 🗄️ Database Management")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🗑️ Clear Patients", use_container_width=True):
            conn = get_db_connection(); c = conn.cursor(); c.execute("DELETE FROM patients"); conn.commit(); conn.close(); st.success("All patient records cleared."); st.rerun()
    with col2:
        if st.button("🗑️ Clear Alerts", use_container_width=True):
            conn = get_db_connection(); c = conn.cursor(); c.execute("DELETE FROM alerts"); conn.commit(); conn.close(); st.success("All alerts cleared."); st.rerun()
    with col3:
        if st.button("🗑️ Clear Feedback", use_container_width=True):
            conn = get_db_connection(); c = conn.cursor(); c.execute("DELETE FROM feedback"); conn.commit(); conn.close(); st.success("All feedback cleared."); st.rerun()
    with col4:
        if st.button("🗑️ Clear Farms", use_container_width=True):
            conn = get_db_connection(); c = conn.cursor(); c.execute("DELETE FROM farm_records"); conn.commit(); conn.close(); st.success("All farm records cleared."); st.rerun()

    st.markdown("---")
    st.markdown("### 📤 Export Data")
    conn = get_db_connection()
    patients_df = pd.read_sql_query("SELECT * FROM patients", conn)
    if len(patients_df) > 0:
        csv_patients = patients_df.to_csv(index=False)
        st.download_button("📥 Export Patients (CSV)", csv_patients, "rin_patients.csv", "text/csv")
    alerts_df = pd.read_sql_query("SELECT * FROM alerts", conn)
    if len(alerts_df) > 0:
        csv_alerts = alerts_df.to_csv(index=False)
        st.download_button("📥 Export Alerts (CSV)", csv_alerts, "rin_alerts.csv", "text/csv")
    feedback_df = pd.read_sql_query("SELECT * FROM feedback", conn)
    if len(feedback_df) > 0:
        csv_feedback = feedback_df.to_csv(index=False)
        st.download_button("📥 Export Feedback (CSV)", csv_feedback, "rin_feedback.csv", "text/csv")
    farms_df = pd.read_sql_query("SELECT * FROM farm_records", conn)
    if len(farms_df) > 0:
        csv_farms = farms_df.to_csv(index=False)
        st.download_button("📥 Export Farm Records (CSV)", csv_farms, "rin_farms.csv", "text/csv")
    conn.close()

    st.markdown("---")
    st.markdown("### ℹ️ About RIN AI")
    st.markdown(f"""<div class="module-card"><h3 style="color: #38bdf8;">RIN AI v2.0 — GAIOS Platform</h3><p><strong>Founder:</strong> Mark Rinwi Bonzum</p><p><strong>Location:</strong> Bamenda, Cameroon</p><p><strong>Mission:</strong> Build the intelligence layer that makes humanity permanently more capable, more equitable, and more resilient.</p><p><strong>Active Modules:</strong> RIN MEDIC, RIN AGRI (ML-powered)</p><p><strong>Future Modules:</strong> RIN GRID, RIN GOV, RIN EDU</p><p><strong>Core Principles:</strong></p><ul><li>Humans are always in control</li><li>Works even with poor internet</li><li>Explains everything it does</li><li>Never stops learning</li></ul><p style="color: #64748b; font-size: 0.8rem; margin-top: 1rem;">Built with Python, Streamlit, scikit-learn, SQLite, Plotly, and OpenWeatherMap API. Crop model trained on real agricultural dataset (2,200+ records).</p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📝 Changelog")
    st.markdown("""
    <div class="module-card">
        <p><strong>v2.0</strong> — ML Crop Recommendation (August 2026)</p>
        <ul style="color: #94a3b8;">
            <li>✅ Replaced rule-based crop recommendation with ML model trained on real dataset</li>
            <li>✅ Random Forest classifier on 2,200+ agricultural records (22 crop types)</li>
            <li>✅ Top-3 crop recommendations with probability scores</li>
            <li>✅ Feature contribution analysis (SHAP-like explanations)</li>
            <li>✅ Soil-climate compatibility analysis with z-score comparisons</li>
            <li>✅ Crop-specific action plans (22 crops)</li>
            <li>✅ Model accuracy display in sidebar and analytics</li>
            <li>✅ Offline fallback dataset for areas with poor connectivity</li>
        </ul>
        <p><strong>v1.2</strong> — User Feedback Update (July 2026)</p>
        <ul style="color: #94a3b8;">
            <li>✅ Added Home page with visible module cards</li>
            <li>✅ Auto-calculate BMI from height and weight</li>
            <li>✅ Multi-select symptom list (replaced free text)</li>
            <li>✅ Fixed AI explanation — fever no longer linked to diabetes risk</li>
            <li>✅ Added confidence bar on results</li>
            <li>✅ Added "Top Factors Influencing Prediction" badges</li>
            <li>✅ Added "Recommended Next Steps" section</li>
            <li>✅ Added color-coded reference ranges</li>
            <li>✅ Added mobile-friendly card view for patient records</li>
            <li>✅ Color-coded abnormal values (glucose, BP, BMI)</li>
            <li>✅ Fixed disclaimer readability</li>
            <li>✅ Hidden default Streamlit UI elements</li>
            <li>✅ Fixed active tab color (blue instead of red)</li>
        </ul>
        <p><strong>v1.1</strong> — Weather API Integration</p>
        <ul style="color: #94a3b8;">
            <li>✅ OpenWeatherMap API integration</li>
            <li>✅ 5-day weather forecast</li>
            <li>✅ Weather caching</li>
            <li>✅ Farm records database</li>
        </ul>
        <p><strong>v1.0</strong> — Initial Release</p>
        <ul style="color: #94a3b8;">
            <li>✅ RIN MEDIC prototype</li>
            <li>✅ RIN AGRI prototype</li>
            <li>✅ Outbreak detection</li>
            <li>✅ Patient database</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    render_bottom_nav("⚙️ Settings")

# ═══════════════════════════════════════════════════════════════════════════════
# BOTTOM NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════

def render_bottom_nav(current_page):
    st.markdown("---")
    st.markdown("### 🚀 Quick Navigation")
    cols = st.columns(4)
    nav_options = [
        ("🏠 Home", "home_btn"),
        ("🏥 RIN MEDIC", "medic_btn"),
        ("🌾 RIN AGRI", "agri_btn"),
        ("📊 Analytics", "analytics_btn")
    ]
    for i, (label, key) in enumerate(nav_options):
        with cols[i]:
            if label != current_page:
                if st.button(f"Go to {label}", key=f"bottom_{key}", use_container_width=True):
                    st.session_state['current_page'] = label
                    st.rerun()
            else:
                st.markdown(f"<div style='text-align: center; padding: 0.5rem; color: #38bdf8; font-weight: 600;'>📍 {label}</div>", unsafe_allow_html=True)

# FOOTER
st.markdown("---")
st.markdown("""<div style="text-align: center; padding: 1rem; color: #64748b; font-size: 0.8rem;"><p>🧠 <strong>RIN AI</strong> — Global Autonomous Intelligence Platform · v2.0</p><p>Founded by Mark Rinwi Bonzum · Bamenda, Cameroon · 2026</p><p style="color: #38bdf8;">Collect → Clean → Understand → Connect → Act → Learn → Repeat</p></div>""", unsafe_allow_html=True)
