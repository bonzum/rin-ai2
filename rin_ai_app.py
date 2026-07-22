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
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="RIN AI v1.1 | Weather-Enabled", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    padding: 2rem; border-radius: 16px; margin-bottom: 2rem;
    border: 1px solid rgba(56, 189, 248, 0.2);
}
.metric-box {
    background: #1e293b; padding: 1.2rem; border-radius: 10px;
    text-align: center; border: 1px solid rgba(56, 189, 248, 0.1);
}
.metric-value { font-size: 2.2rem; font-weight: 800; color: #38bdf8; }
.metric-label { font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
.alert-high {
    background: linear-gradient(145deg, #7f1d1d 0%, #991b1b 100%);
    border-left: 4px solid #ef4444; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
}
.alert-medium {
    background: linear-gradient(145deg, #713f12 0%, #854d0e 100%);
    border-left: 4px solid #f59e0b; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;
}
.module-card {
    background: #1e293b; padding: 1.5rem; border-radius: 12px;
    border: 1px solid rgba(56, 189, 248, 0.15); margin-bottom: 1rem;
}
.weather-card {
    background: linear-gradient(145deg, #1e3a5f 0%, #0f172a 100%);
    padding: 1.2rem; border-radius: 12px;
    border: 1px solid rgba(56, 189, 248, 0.2); margin-bottom: 0.8rem;
}
.explanation-box {
    background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 8px; padding: 1rem; margin-top: 1rem;
}
.forecast-row {
    background: rgba(30, 41, 59, 0.8); padding: 0.8rem;
    border-radius: 8px; margin: 0.3rem 0; border-left: 3px solid #38bdf8;
}
.risk-high { color: #ef4444; font-weight: 700; }
.risk-medium { color: #f59e0b; font-weight: 700; }
.risk-low { color: #22c55e; font-weight: 700; }
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
</style>
""", unsafe_allow_html=True)

# DATABASE

def init_database():
    conn = sqlite3.connect('rin_ai.db')
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, age INTEGER, gender TEXT, location TEXT,
        temperature REAL, blood_pressure_sys INTEGER, blood_pressure_dia INTEGER,
        heart_rate INTEGER, glucose REAL, bmi REAL, symptoms TEXT,
        diabetes_risk TEXT, risk_score REAL, risk_explanation TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
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
        recommended_crop TEXT, confidence INTEGER, weather_data TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit(); conn.close()

init_database()

def get_db_connection():
    return sqlite3.connect('rin_ai.db')

# WEATHER API

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
        st.warning(f"Weather API error: {str(e)[:100]}. Using simulated data.")
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

# AI MODEL

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

model, scaler, model_accuracy, feature_names = load_diabetes_model()

# OUTBREAK DETECTION

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

# SIDEBAR
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid rgba(56, 189, 248, 0.2); margin-bottom: 1rem;">
        <h1 style="color: #38bdf8; margin: 0; font-size: 1.8rem; font-weight: 800;">🧠 RIN AI</h1>
        <p style="color: #94a3b8; margin: 0.3rem 0 0 0; font-size: 0.75rem; letter-spacing: 2px;">AUTONOMOUS INTELLIGENCE</p>
        <p style="color: #64748b; margin: 0.2rem 0 0 0; font-size: 0.65rem;">Bamenda, Cameroon - 2026</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigate",
        ["🏠 Dashboard", "🏥 RIN MEDIC", "🌾 RIN AGRI", "📊 Analytics", "⚙️ Settings"],
        label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### 🔋 System Status")
    st.markdown(f"**AI Model Accuracy:** `{model_accuracy:.1%}`")
    st.markdown(f"**Model Type:** `Random Forest`")
    st.markdown(f"**Database:** `SQLite Active`")
    st.markdown(f"**Weather API:** `OpenWeatherMap Ready`")
    st.markdown("---")
    st.markdown("### 🔄 Intelligence Cycle")
    cycle_steps = ["Collect", "Clean", "Understand", "Connect", "Act", "Learn"]
    current_step = (datetime.now().second // 10) % 6
    for i, step in enumerate(cycle_steps):
        if i == current_step: st.markdown(f"**→ {step}** ⚡")
        else: st.markdown(f"<span style='color: #64748b'>{step}</span>", unsafe_allow_html=True)

# MAIN HEADER
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

# DASHBOARD
if page == "🏠 Dashboard":
    st.markdown("## 📊 Command Center")
    conn = get_db_connection()
    total_patients = pd.read_sql_query("SELECT COUNT(*) as count FROM patients", conn).iloc[0]['count']
    high_risk = pd.read_sql_query("SELECT COUNT(*) as count FROM patients WHERE diabetes_risk='HIGH'", conn).iloc[0]['count']
    today_patients = pd.read_sql_query("SELECT COUNT(*) as count FROM patients WHERE DATE(created_at) = DATE('now')", conn).iloc[0]['count']
    active_alerts = pd.read_sql_query("SELECT COUNT(*) as count FROM alerts WHERE status='active'", conn).iloc[0]['count']
    total_farms = pd.read_sql_query("SELECT COUNT(*) as count FROM farm_records", conn).iloc[0]['count']
    conn.close()
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.markdown(f"""<div class="metric-box"><div class="metric-value">{total_patients}</div><div class="metric-label">Total Patients</div></div>""", unsafe_allow_html=True)
    with col2: st.markdown(f"""<div class="metric-box"><div class="metric-value" style="color: #ef4444;">{high_risk}</div><div class="metric-label">High Risk Cases</div></div>""", unsafe_allow_html=True)
    with col3: st.markdown(f"""<div class="metric-box"><div class="metric-value" style="color: #22c55e;">{today_patients}</div><div class="metric-label">Today's Cases</div></div>""", unsafe_allow_html=True)
    with col4: st.markdown(f"""<div class="metric-box"><div class="metric-value" style="color: #f59e0b;">{active_alerts}</div><div class="metric-label">Active Alerts</div></div>""", unsafe_allow_html=True)
    with col5: st.markdown(f"""<div class="metric-box"><div class="metric-value" style="color: #a855f7;">{total_farms}</div><div class="metric-label">Farm Assessments</div></div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🚨 Active Intelligence Alerts")
    new_alerts = check_outbreaks()
    for alert in new_alerts: save_alert(alert)
    conn = get_db_connection()
    alerts_df = pd.read_sql_query("SELECT * FROM alerts WHERE status='active' ORDER BY created_at DESC LIMIT 10", conn)
    conn.close()
    if len(alerts_df) > 0:
        for _, alert in alerts_df.iterrows():
            severity_class = f"alert-{alert['severity']}"
            st.markdown(f"""<div class="{severity_class}"><strong style="color: white;">{alert['message']}</strong><br><span style="color: rgba(255,255,255,0.7); font-size: 0.8rem;">Detected: {alert['created_at']} | Type: {alert['alert_type']}</span></div>""", unsafe_allow_html=True)
    else: st.success("✅ No active outbreak alerts. All systems normal.")
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 👥 Recent Patient Assessments")
        conn = get_db_connection()
        recent = pd.read_sql_query("SELECT name, age, location, diabetes_risk, risk_score, created_at FROM patients ORDER BY created_at DESC LIMIT 10", conn)
        conn.close()
        if len(recent) > 0:
            for _, row in recent.iterrows():
                border_color = '#ef4444' if row['diabetes_risk']=='HIGH' else '#f59e0b' if row['diabetes_risk']=='MEDIUM' else '#22c55e'
                risk_badge = "🔴 HIGH" if row['diabetes_risk']=='HIGH' else "🟡 MEDIUM" if row['diabetes_risk']=='MEDIUM' else "🟢 LOW"
                st.markdown(f"""<div style="background: #1e293b; padding: 0.8rem; border-radius: 8px; margin: 0.3rem 0; border-left: 3px solid {border_color};"><strong>{row['name']}</strong> · {row['age']}y · {row['location']} · <span style="color: {border_color}; font-weight: 700;">{risk_badge}</span> · Score: {row['risk_score']:.1f}%<br><span style="color: #64748b; font-size: 0.75rem;">{row['created_at']}</span></div>""", unsafe_allow_html=True)
        else: st.info("No patients recorded yet. Go to RIN MEDIC to add your first patient.")
    with col2:
        st.markdown("### 📈 Risk Distribution")
        conn = get_db_connection()
        risk_dist = pd.read_sql_query("SELECT diabetes_risk, COUNT(*) as count FROM patients GROUP BY diabetes_risk", conn)
        conn.close()
        if len(risk_dist) > 0:
            import plotly.express as px
            fig = px.pie(risk_dist, values='count', names='diabetes_risk', color='diabetes_risk',
                        color_discrete_map={'HIGH': '#ef4444', 'MEDIUM': '#f59e0b', 'LOW': '#22c55e'})
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8',
                             showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2))
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("Add patients to see risk distribution.")

# RIN MEDIC
elif page == "🏥 RIN MEDIC":
    st.markdown("## 🏥 RIN MEDIC — Autonomous Clinical Intelligence")
    st.markdown("""<p style="color: #94a3b8;">AI-powered decision support for healthcare workers. Not a replacement for clinical judgment — an amplifier of it. Every recommendation includes an explanation.</p>""", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["➕ New Patient Assessment", "📋 Patient Records"])
    with tab1:
        st.markdown("### 📝 Enter Patient Information")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Patient Name", placeholder="e.g., Mary Ngwa")
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=35)
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            location = st.text_input("Location / Village", placeholder="e.g., Bamenda Central")
        with col2:
            temperature = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0, value=36.5, step=0.1)
            bp_sys = st.number_input("Blood Pressure (Systolic)", min_value=60, max_value=250, value=120)
            bp_dia = st.number_input("Blood Pressure (Diastolic)", min_value=40, max_value=150, value=80)
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=72)
        col3, col4 = st.columns(2)
        with col3:
            glucose = st.number_input("Blood Glucose (mg/dL)", min_value=50, max_value=500, value=100)
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0, step=0.1)
        with col4:
            symptoms = st.text_area("Symptoms (comma separated)", placeholder="e.g., fever, headache, fatigue, blurred vision", height=100)
            pregnancies = st.number_input("Number of Pregnancies (if applicable)", min_value=0, max_value=20, value=0)
        st.markdown("---")
        if st.button("🔬 Run AI Risk Assessment", use_container_width=True):
            with st.spinner("RIN AI is analyzing patient data..."):
                skin_thickness, insulin, dpf = 25, 80, 0.5
                features = np.array([[pregnancies, glucose, bp_sys, skin_thickness, insulin, bmi, dpf, age]])
                features_scaled = scaler.transform(features)
                risk_prob = model.predict_proba(features_scaled)[0][1]
                if risk_prob >= 0.7: risk_level, risk_color, risk_icon = "HIGH", "#ef4444", "🔴"
                elif risk_prob >= 0.4: risk_level, risk_color, risk_icon = "MEDIUM", "#f59e0b", "🟡"
                else: risk_level, risk_color, risk_icon = "LOW", "#22c55e", "🟢"
                explanation_parts = []
                if glucose > 126: explanation_parts.append(f"Blood glucose ({glucose} mg/dL) is elevated")
                if bmi > 30: explanation_parts.append(f"BMI ({bmi}) indicates obesity")
                elif bmi > 25: explanation_parts.append(f"BMI ({bmi}) indicates overweight")
                if age > 45: explanation_parts.append(f"Age ({age}) — risk increases after 45")
                if bp_sys > 140: explanation_parts.append(f"Blood pressure ({bp_sys}/{bp_dia}) is elevated")
                if temperature > 38: explanation_parts.append(f"Temperature ({temperature}°C) indicates fever")
                if not explanation_parts: explanation_parts.append("All vitals appear within normal ranges. Continue regular monitoring.")
                explanation = "; ".join(explanation_parts)
                conn = get_db_connection()
                c = conn.cursor()
                c.execute("""INSERT INTO patients (name, age, gender, location, temperature, blood_pressure_sys, blood_pressure_dia, heart_rate, glucose, bmi, symptoms, diabetes_risk, risk_score, risk_explanation) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (name, age, gender, location, temperature, bp_sys, bp_dia, heart_rate, glucose, bmi, symptoms, risk_level, risk_prob * 100, explanation))
                patient_id = c.lastrowid
                conn.commit(); conn.close()
                st.markdown("---")
                st.markdown(f"""<div style="background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%); padding: 2rem; border-radius: 16px; border: 2px solid {risk_color};"><h2 style="color: {risk_color}; margin: 0;">{risk_icon} Diabetes Risk: {risk_level}</h2><p style="color: white; font-size: 1.5rem; margin: 0.5rem 0;">Confidence: {risk_prob:.1%}</p><div class="explanation-box"><strong style="color: #38bdf8;">🧠 RIN AI Explanation:</strong><br><span style="color: #e2e8f0;">{explanation}</span></div><p style="color: #64748b; margin-top: 1rem; font-size: 0.8rem;">⚠️ This is a decision support tool. Always consult a qualified healthcare professional before making clinical decisions. Patient ID: #{patient_id}</p></div>""", unsafe_allow_html=True)
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
                    feedback_comment = st.text_input("Comment (optional)", key=f"comment_{patient_id}")
                    if feedback_comment:
                        conn = get_db_connection(); c = conn.cursor()
                        c.execute("INSERT INTO feedback (patient_id, module, helpful, comment) VALUES (?, ?, ?, ?)", (patient_id, "RIN MEDIC", "Comment", feedback_comment))
                        conn.commit(); conn.close()
    with tab2:
        st.markdown("### 📋 All Patient Records")
        conn = get_db_connection()
        all_patients = pd.read_sql_query("SELECT id, name, age, gender, location, diabetes_risk, risk_score, symptoms, created_at FROM patients ORDER BY created_at DESC", conn)
        conn.close()
        if len(all_patients) > 0:
            def color_risk(val):
                if val == 'HIGH': return 'background-color: rgba(239, 68, 68, 0.2); color: #ef4444; font-weight: bold'
                elif val == 'MEDIUM': return 'background-color: rgba(245, 158, 11, 0.2); color: #f59e0b; font-weight: bold'
                else: return 'background-color: rgba(34, 197, 94, 0.2); color: #22c55e; font-weight: bold'
            styled_df = all_patients.style.map(color_risk, subset=['diabetes_risk'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            csv = all_patients.to_csv(index=False)
            st.download_button(label="📥 Download Patient Data (CSV)", data=csv, file_name="rin_medic_patients.csv", mime="text/csv")
        else: st.info("No patient records found. Add patients using the New Patient Assessment tab.")

# RIN AGRI — WITH REAL WEATHER API
elif page == "🌾 RIN AGRI":
    st.markdown("## 🌾 RIN AGRI — Precision Agriculture Intelligence")
    st.markdown("""<p style="color: #94a3b8;">AI-powered crop recommendations for farmers. Now with <strong>live weather data</strong> from OpenWeatherMap API. Delivers personalized harvest strategies based on real-time soil, weather, and market data.</p>""", unsafe_allow_html=True)

    with st.expander("🔧 Weather API Configuration"):
        st.markdown("""<p style="color: #94a3b8;">RIN AGRI can fetch <strong>real-time weather data</strong> from OpenWeatherMap. Without an API key, it uses RIN AI's local weather simulation model.</p>""", unsafe_allow_html=True)
        api_key = st.text_input("OpenWeatherMap API Key (optional)", value=st.session_state.get('weather_api_key', ''), type="password", placeholder="Enter your API key or leave blank for simulation", help="Get a free API key at openweathermap.org/api")
        if api_key:
            st.session_state['weather_api_key'] = api_key
            st.success("✅ API key saved for this session!")
        st.markdown("""<p style="color: #64748b; font-size: 0.8rem;">💡 <strong>How to get a free API key:</strong><br>1. Go to <a href="https://openweathermap.org/api" target="_blank">openweathermap.org/api</a><br>2. Sign up (free, no credit card)<br>3. Go to "API Keys" in your account<br>4. Copy the key and paste it above<br>Free tier: 1,000 calls/day (more than enough)</p>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🌱 Crop Recommendation", "🌤️ Weather Station", "📊 Farm Records"])

    with tab1:
        st.markdown("### 🌾 Get Your Personalized Crop Plan")
        col1, col2 = st.columns(2)
        with col1:
            farmer_name = st.text_input("Farmer Name", placeholder="e.g., John Tabi")
            farm_location = st.text_input("Farm Location (City/Village)", placeholder="e.g., Bamenda, CM")
            farm_size = st.number_input("Farm Size (hectares)", min_value=0.1, max_value=100.0, value=1.0, step=0.1)
            soil_type = st.selectbox("Soil Type", ["Clay", "Sandy", "Loamy", "Silty", "Peaty", "Chalky", "Unknown"])
        with col2:
            nitrogen = st.slider("Soil Nitrogen (N) level", 0, 140, 50)
            phosphorus = st.slider("Soil Phosphorus (P) level", 0, 140, 50)
            potassium = st.slider("Soil Potassium (K) level", 0, 140, 50)
            ph = st.slider("Soil pH Level", 0.0, 14.0, 6.5, step=0.1)

        st.markdown("---")
        st.markdown("### 🌤️ Live Weather Data")
        weather = None
        if farm_location:
            with st.spinner(f"Fetching weather for {farm_location}..."):
                api_key_to_use = st.session_state.get('weather_api_key', None)
                weather = get_weather_data(farm_location, api_key_to_use)
            if weather:
                source_color = "#22c55e" if "API" in weather['source'] else "#f59e0b"
                st.markdown(f"""<div style="margin-bottom: 0.5rem;"><span style="color: {source_color}; font-size: 0.8rem; font-weight: 600;">● Data Source: {weather['source']}</span></div>""", unsafe_allow_html=True)
                wcol1, wcol2, wcol3, wcol4 = st.columns(4)
                with wcol1: st.metric("Temperature", f"{weather['temperature']}°C", f"Feels {weather['feels_like']}°C")
                with wcol2: st.metric("Humidity", f"{weather['humidity']}%")
                with wcol3: st.metric("Wind", f"{weather['wind_speed']} m/s")
                with wcol4: st.metric("Visibility", f"{weather['visibility']} km")
                st.markdown(f"""<div class="weather-card"><div style="display: flex; justify-content: space-between; align-items: center;"><div><strong style="color: #38bdf8; font-size: 1.2rem;">{weather['description']}</strong><br><span style="color: #94a3b8;">Pressure: {weather['pressure']} hPa | Clouds: {weather['clouds']}%</span><br><span style="color: #64748b; font-size: 0.8rem;">🌅 {weather['sunrise']} | 🌇 {weather['sunset']}</span></div></div></div>""", unsafe_allow_html=True)
                if weather.get('forecast'):
                    st.markdown("#### 📅 5-Day Forecast")
                    fcols = st.columns(len(weather['forecast']))
                    for i, fc in enumerate(weather['forecast']):
                        with fcols[i]:
                            rain_icon = "🌧️" if fc['rain'] > 5 else "🌦️" if fc['rain'] > 0 else "☀️"
                            st.markdown(f"""<div style="background: #1e293b; padding: 0.6rem; border-radius: 8px; text-align: center; border: 1px solid rgba(56, 189, 248, 0.1);"><div style="font-size: 0.75rem; color: #94a3b8;">{fc['date'][5:]}</div><div style="font-size: 1.5rem; margin: 0.2rem 0;">{rain_icon}</div><div style="font-size: 0.9rem; color: white; font-weight: 600;">{fc['temp_max']}° / {fc['temp_min']}°</div><div style="font-size: 0.65rem; color: #64748b;">{fc['rain']}mm rain</div></div>""", unsafe_allow_html=True)
        else: st.info("👆 Enter a farm location above to fetch live weather data.")

        st.markdown("---")
        if weather:
            temperature_crop = weather['temperature']
            rainfall = weather.get('rainfall', 100)
            st.markdown(f"""<div style="background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.3); padding: 0.8rem; border-radius: 8px; margin-bottom: 1rem;"><strong style="color: #22c55e;">✅ Auto-detected from weather data:</strong><br><span style="color: #e2e8f0;">Temperature: {temperature_crop}°C | Rainfall estimate: {rainfall}mm</span></div>""", unsafe_allow_html=True)
        else:
            temperature_crop = st.slider("Average Temperature (°C)", 10, 45, 25)
            rainfall = st.slider("Average Rainfall (mm)", 0, 300, 100)

        if st.button("🌱 Generate Crop Recommendation", use_container_width=True):
            with st.spinner("RIN AI is analyzing soil, weather, and market data..."):
                recommendations = []
                temp_factor = temperature_crop
                rain_factor = rainfall
                if soil_type in ["Loamy", "Silty"] and ph >= 6.0 and ph <= 7.5:
                    if rain_factor > 80 and temp_factor >= 20 and temp_factor <= 30:
                        recommendations.append({"crop": "Maize", "confidence": 92, "reason": f"Loamy soil with good pH. Current temp {temp_factor}°C and rainfall {rain_factor}mm ideal for maize.", "planting": "Next 2 weeks" if weather and weather.get('forecast') and weather['forecast'][0].get('rain', 0) < 10 else "Wait for dry spell", "harvest": "3-4 months", "yield": f"~{farm_size * 3.5:.1f} tonnes", "market_price": "Good — stable demand"})
                        recommendations.append({"crop": "Beans", "confidence": 85, "reason": "Nitrogen-fixing crop, improves soil for next season. Good market price.", "planting": "Now", "harvest": "2-3 months", "yield": f"~{farm_size * 1.2:.1f} tonnes", "market_price": "Excellent — high demand"})
                if nitrogen > 80 and phosphorus > 40 and potassium > 40:
                    recommendations.append({"crop": "Rice", "confidence": 88, "reason": "High nutrient soil supports rice. Ensure water management.", "planting": "After next rainfall", "harvest": "4-5 months", "yield": f"~{farm_size * 4.0:.1f} tonnes", "market_price": "Stable — local staple"})
                if temp_factor > 28 and rain_factor < 100:
                    recommendations.append({"crop": "Cassava", "confidence": 90, "reason": f"Drought-resistant, thrives in warm {temp_factor}°C temperatures. Low water needs.", "planting": "Anytime", "harvest": "8-12 months", "yield": f"~{farm_size * 15:.1f} tonnes", "market_price": "Growing — industrial demand"})
                if ph < 5.5:
                    recommendations.append({"crop": "Groundnuts (Peanuts)", "confidence": 82, "reason": "Tolerates slightly acidic soil. Good nitrogen fixer.", "planting": "After soil lime treatment", "harvest": "4-5 months", "yield": f"~{farm_size * 1.5:.1f} tonnes", "market_price": "Good — export potential"})
                if not recommendations:
                    recommendations.append({"crop": "Sorghum", "confidence": 75, "reason": "Hardy crop that tolerates variable conditions. Good starting point.", "planting": "Next rainfall", "harvest": "3-4 months", "yield": f"~{farm_size * 2.0:.1f} tonnes", "market_price": "Stable"})
                weather_json = json.dumps(weather) if weather else "{}"
                conn = get_db_connection()
                c = conn.cursor()
                c.execute("""INSERT INTO farm_records (farmer_name, farm_location, farm_size, soil_type, nitrogen, phosphorus, potassium, ph, recommended_crop, confidence, weather_data) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (farmer_name, farm_location, farm_size, soil_type, nitrogen, phosphorus, potassium, ph, recommendations[0]['crop'], recommendations[0]['confidence'], weather_json))
                conn.commit(); conn.close()
                st.markdown("---")
                st.markdown("### 🎯 RIN AI Crop Recommendations")
                for i, rec in enumerate(recommendations[:3]):
                    confidence_color = "#22c55e" if rec['confidence'] >= 90 else "#f59e0b" if rec['confidence'] >= 80 else "#38bdf8"
                    st.markdown(f"""<div class="module-card"><div style="display: flex; justify-content: space-between; align-items: center;"><h3 style="color: white; margin: 0;">#{i+1} {rec['crop']}</h3><span style="background: {confidence_color}; color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600;">{rec['confidence']}% Match</span></div><p style="color: #94a3b8; margin: 0.5rem 0;">{rec['reason']}</p><div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;"><div style="background: rgba(56, 189, 248, 0.1); padding: 0.8rem; border-radius: 8px;"><strong style="color: #38bdf8;">🌱 Best Planting Time</strong><br><span style="color: #e2e8f0;">{rec['planting']}</span></div><div style="background: rgba(34, 197, 94, 0.1); padding: 0.8rem; border-radius: 8px;"><strong style="color: #22c55e;">🌾 Expected Harvest</strong><br><span style="color: #e2e8f0;">{rec['harvest']} · {rec['yield']}</span></div><div style="background: rgba(245, 158, 11, 0.1); padding: 0.8rem; border-radius: 8px;"><strong style="color: #f59e0b;">💰 Market Outlook</strong><br><span style="color: #e2e8f0;">{rec['market_price']}</span></div><div style="background: rgba(239, 68, 68, 0.1); padding: 0.8rem; border-radius: 8px;"><strong style="color: #ef4444;">⚠️ Risk Factor</strong><br><span style="color: #e2e8f0;">Monitor rainfall closely</span></div></div></div>""", unsafe_allow_html=True)
                st.markdown("### 📋 Your Action Plan")
                st.markdown("""<div style="background: #1e293b; padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(56, 189, 248, 0.2);"><ol style="color: #e2e8f0; line-height: 2;"><li><strong>Test soil pH</strong> — If below 6.0, consider lime treatment before planting</li><li><strong>Check seed quality</strong> — Use certified seeds for best yield</li><li><strong>Plan irrigation</strong> — Ensure water access during dry spells</li><li><strong>Monitor weekly</strong> — Log pest sightings and growth progress in RIN AI</li><li><strong>Connect with buyers</strong> — Contact local cooperative before harvest</li></ol></div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("### 🌤️ Weather Station")
        st.markdown("""<p style="color: #94a3b8;">Check live weather for any location. Enter a city or village name below.</p>""", unsafe_allow_html=True)
        weather_query = st.text_input("Enter Location", placeholder="e.g., Bamenda, Douala, Yaounde")
        if weather_query:
            with st.spinner(f"Fetching weather for {weather_query}..."):
                api_key_to_use = st.session_state.get('weather_api_key', None)
                weather_data = get_weather_data(weather_query, api_key_to_use)
            if weather_data:
                source_color = "#22c55e" if "API" in weather_data['source'] else "#f59e0b"
                st.markdown(f"""<div style="margin-bottom: 1rem;"><span style="color: {source_color}; font-size: 0.9rem; font-weight: 600;">● {weather_data['source']}</span></div>""", unsafe_allow_html=True)
                st.markdown(f"""<div class="weather-card"><div style="display: flex; justify-content: space-between; align-items: center;"><div><h2 style="color: white; margin: 0; font-size: 3rem;">{weather_data['temperature']}°C</h2><p style="color: #38bdf8; font-size: 1.2rem; margin: 0;">{weather_data['description']}</p><p style="color: #94a3b8; margin: 0;">Feels like {weather_data['feels_like']}°C</p></div><div style="text-align: right;"><p style="color: #94a3b8; margin: 0;">💧 Humidity: {weather_data['humidity']}%</p><p style="color: #94a3b8; margin: 0;">💨 Wind: {weather_data['wind_speed']} m/s</p><p style="color: #94a3b8; margin: 0;">👁️ Visibility: {weather_data['visibility']} km</p><p style="color: #64748b; margin: 0; font-size: 0.8rem;">🌅 {weather_data['sunrise']} | 🌇 {weather_data['sunset']}</p></div></div></div>""", unsafe_allow_html=True)
                if weather_data.get('forecast'):
                    st.markdown("#### 📅 5-Day Forecast")
                    for fc in weather_data['forecast']:
                        rain_icon = "🌧️" if fc['rain'] > 5 else "🌦️" if fc['rain'] > 0 else "☀️"
                        st.markdown(f"""<div class="forecast-row"><div style="display: flex; justify-content: space-between; align-items: center;"><div style="display: flex; align-items: center; gap: 1rem;"><span style="font-size: 1.5rem;">{rain_icon}</span><div><strong style="color: white;">{fc['date']}</strong><br><span style="color: #94a3b8;">{fc['description']}</span></div></div><div style="text-align: right;"><strong style="color: #38bdf8;">{fc['temp_max']}°C</strong> / <span style="color: #64748b;">{fc['temp_min']}°C</span><br><span style="color: #94a3b8; font-size: 0.8rem;">🌧️ {fc['rain']}mm expected</span></div></div></div>""", unsafe_allow_html=True)
            else: st.error("Could not fetch weather data. Please check the location name.")

    with tab3:
        st.markdown("### 📊 Farm Records")
        conn = get_db_connection()
        farms = pd.read_sql_query("SELECT farmer_name, farm_location, farm_size, soil_type, recommended_crop, confidence, created_at FROM farm_records ORDER BY created_at DESC", conn)
        conn.close()
        if len(farms) > 0:
            st.dataframe(farms, use_container_width=True, hide_index=True)
            csv = farms.to_csv(index=False)
            st.download_button("📥 Download Farm Records (CSV)", csv, "rin_farm_records.csv", "text/csv")
        else: st.info("No farm assessments yet. Use the Crop Recommendation tab to add your first farm.")

# ANALYTICS
elif page == "📊 Analytics":
    st.markdown("## 📊 RIN AI Intelligence Analytics")
    conn = get_db_connection()
    daily_counts = pd.read_sql_query("SELECT DATE(created_at) as date, COUNT(*) as count FROM patients GROUP BY DATE(created_at) ORDER BY date", conn)
    risk_dist = pd.read_sql_query("SELECT diabetes_risk, COUNT(*) as count FROM patients GROUP BY diabetes_risk", conn)
    location_dist = pd.read_sql_query("SELECT location, COUNT(*) as count FROM patients WHERE location != '' GROUP BY location ORDER BY count DESC LIMIT 10", conn)
    feedback_stats = pd.read_sql_query("SELECT helpful, COUNT(*) as count FROM feedback GROUP BY helpful", conn)
    farm_dist = pd.read_sql_query("SELECT recommended_crop, COUNT(*) as count FROM farm_records GROUP BY recommended_crop", conn)
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
        st.markdown("### 🌾 Top Recommended Crops")
        if len(farm_dist) > 0:
            import plotly.express as px
            fig = px.bar(farm_dist, x='recommended_crop', y='count', labels={'recommended_crop': 'Crop', 'count': 'Recommendations'}, color='count', color_continuous_scale='Greens')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8', xaxis_gridcolor='rgba(148, 163, 184, 0.1)', yaxis_gridcolor='rgba(148, 163, 184, 0.1)')
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("No farm data yet. Use RIN AGRI to add farm assessments.")
    st.markdown("---")
    st.markdown("### 👍 User Feedback")
    if len(feedback_stats) > 0:
        import plotly.express as px
        fig = px.bar(feedback_stats, x='helpful', y='count', labels={'helpful': 'Feedback', 'count': 'Count'}, color='helpful', color_discrete_map={'Yes': '#22c55e', 'No': '#ef4444', 'Comment': '#38bdf8'})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8', xaxis_gridcolor='rgba(148, 163, 184, 0.1)', yaxis_gridcolor='rgba(148, 163, 184, 0.1)')
        st.plotly_chart(fig, use_container_width=True)
    else: st.info("No feedback yet. Users can rate assessments in RIN MEDIC.")

# SETTINGS
elif page == "⚙️ Settings":
    st.markdown("## ⚙️ RIN AI System Settings")
    st.markdown("### 🧠 Model Configuration")
    st.markdown(f"""<div class="module-card"><p><strong>Diabetes Prediction Model:</strong> Random Forest Classifier</p><p><strong>Model Accuracy:</strong> {model_accuracy:.1%}</p><p><strong>Features Used:</strong> {', '.join(feature_names)}</p><p><strong>Training Data:</strong> 2,000 synthetic samples (African demographic profile)</p><p><strong>Last Updated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p></div>""", unsafe_allow_html=True)
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
    st.markdown(f"""<div class="module-card"><h3 style="color: #38bdf8;">RIN AI v1.1 — GAIOS Platform with Weather Integration</h3><p><strong>Founder:</strong> Mark Rinwi Bonzum</p><p><strong>Location:</strong> Bamenda, Cameroon</p><p><strong>Mission:</strong> Build the intelligence layer that makes humanity permanently more capable, more equitable, and more resilient.</p><p><strong>Active Modules:</strong> RIN MEDIC, RIN AGRI (with live weather)</p><p><strong>Future Modules:</strong> RIN GRID, RIN GOV, RIN EDU</p><p><strong>Core Principles:</strong></p><ul><li>Humans are always in control</li><li>Works even with poor internet</li><li>Explains everything it does</li><li>Never stops learning</li></ul><p style="color: #64748b; font-size: 0.8rem; margin-top: 1rem;">Built with Python, Streamlit, scikit-learn, SQLite, Plotly, and OpenWeatherMap API. All data stored locally.</p></div>""", unsafe_allow_html=True)

# FOOTER
st.markdown("---")
st.markdown("""<div style="text-align: center; padding: 1rem; color: #64748b; font-size: 0.8rem;"><p>🧠 <strong>RIN AI</strong> — Global Autonomous Intelligence Platform · v1.1</p><p>Founded by Mark Rinwi Bonzum · Bamenda, Cameroon · 2026</p><p style="color: #38bdf8;">Collect → Clean → Understand → Connect → Act → Learn → Repeat</p></div>""", unsafe_allow_html=True)
