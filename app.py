import os
import json
import io  # Added for reliable binary streaming
from datetime import datetime
from flask import Flask, render_template, request, session, flash, redirect, url_for, g, make_response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from fpdf import FPDF 

# --- App Configuration ---
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "india_heart_key_2026")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Database Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    predictions = db.relationship('Prediction', backref='user', lazy=True)
    
    def set_password(self, password): 
        self.password_hash = generate_password_hash(password)
    def check_password(self, password): 
        return check_password_hash(self.password_hash, password)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    features_json = db.Column(db.String(500))
    prediction_result = db.Column(db.String(50))
    confidence = db.Column(db.Float)

# --- Asset Management ---
def get_data():
    if 'df' not in g:
        try:
            g.df = pd.read_csv('heart.csv')
            g.model = joblib.load('heart_model.pkl')
            if os.path.exists('feature_importances.csv'):
                g.feat_imp_df = pd.read_csv('feature_importances.csv')
            else:
                g.feat_imp_df = pd.DataFrame({'Feature': ['Age', 'CP', 'Chol', 'Trestbps', 'Thalach'], 'Importance': [0.2, 0.3, 0.15, 0.1, 0.25]})
            g.feature_cols = g.df.drop('target', axis=1).columns.tolist() if 'target' in g.df.columns else g.df.columns.tolist()
        except Exception as e:
            print(f"Asset Load Error: {e}")
            g.df = g.model = g.feature_cols = g.feat_imp_df = None

@app.before_request
def load_assets(): 
    get_data()

# --- Auth Routes ---
@app.route("/")
def index():
    """Entry point: Redirects to login if not authenticated, otherwise home."""
    if "user_id" in session:
        return redirect(url_for("home"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(username=request.form["username"]).first()
        if user and user.check_password(request.form["password"]):
            session.update({"user_id": user.id, "username": user.username})
            return redirect(url_for("home"))
        flash("Invalid username or password", "danger")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        if User.query.filter_by(username=request.form["username"]).first():
            flash("Username already exists", "warning")
            return redirect(url_for("register"))
        user = User(username=request.form["username"])
        user.set_password(request.form["password"])
        db.session.add(user)
        db.session.commit()
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/logout")
def logout(): 
    session.clear()
    return redirect(url_for("login"))

# --- Functional Logic Routes ---
@app.route("/home")
def home(): 
    if "user_id" not in session: return redirect(url_for("login"))
    return render_template("home.html", feature_cols=g.feature_cols or [])

@app.route("/predict", methods=["POST"])
def predict():
    if "user_id" not in session: return redirect(url_for("login"))
    try:
        features = {feat: float(request.form[feat]) for feat in g.feature_cols}
        input_df = pd.DataFrame([features])
        prob = g.model.predict_proba(input_df)[0]
        pred = g.model.predict(input_df)[0]
        res = "Heart Disease Detected" if pred == 1 else "No Heart Disease Detected"
        
        new_pred = Prediction(
            user_id=session["user_id"], 
            features_json=json.dumps(features), 
            prediction_result=res, 
            confidence=round(prob[pred]*100, 2)
        )
        db.session.add(new_pred)
        db.session.commit()
        return redirect(url_for("report", prediction_id=new_pred.id))
    except Exception as e: 
        flash(f"Prediction Error: {e}", "danger")
        return redirect(url_for("home"))

@app.route("/history")
def history():
    if "user_id" not in session: return redirect(url_for("login"))
    user_preds = Prediction.query.filter_by(user_id=session["user_id"]).order_by(Prediction.timestamp.desc()).all()
    return render_template("history.html", predictions=user_preds)

@app.route("/report/<int:prediction_id>")
def report(prediction_id):
    if "user_id" not in session: return redirect(url_for("login"))
    pred = Prediction.query.filter_by(id=prediction_id, user_id=session["user_id"]).first_or_404()
    features = json.loads(pred.features_json)
    is_positive = "Detected" in pred.prediction_result
    
    if is_positive:
        med_info = [
            {"type": "Statins", "purpose": "Cholesterol", "desc": "Lowers LDL cholesterol."},
            {"type": "Beta-Blockers", "purpose": "BP Control", "desc": "Reduces heart workload."},
            {"type": "Antiplatelets", "purpose": "Prevention", "desc": "Prevents arterial clotting."}
        ]
        diet_plan = ["Low-sodium DASH diet", "Fiber-rich oats and beans", "Limit saturated fats"]
    else:
        med_info = [
            {"type": "Omega-3", "purpose": "Maintenance", "desc": "Supports vascular health."},
            {"type": "Antioxidants", "purpose": "Protection", "desc": "Reduces oxidative stress."}
        ]
        diet_plan = ["Mediterranean diet", "Leafy greens", "Lean protein sources"]

    analysis_data = {
        "bp": {"label": "Blood Pressure", "val": features.get('trestbps', 0), "med": "120/80", "risk": features.get('trestbps', 0) > 135},
        "chol": {"label": "Cholesterol", "val": features.get('chol', 0), "med": "< 200", "risk": features.get('chol', 0) > 230},
        "hr": {"label": "Max Heart Rate", "val": features.get('thalach', 0), "med": "150-180", "risk": features.get('thalach', 0) < 110}
    }

    real_age = int(features.get('age', 40))
    heart_age = real_age + (8 if is_positive else -2)
    
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number", 
        value=pred.confidence,
        gauge={'bar': {'color': "#dc3545" if is_positive else "#198754"}, 'axis': {'range': [0, 100]}}
    ))
    gauge_fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10))

    return render_template("report.html", 
                           prediction=pred, is_positive=is_positive, 
                           real_age=real_age, heart_age=heart_age, 
                           analysis=analysis_data, med_info=med_info,
                           diet_plan=diet_plan, gauge_html=pio.to_html(gauge_fig, full_html=False))

# --- Analytics & Visualizations ---
@app.route("/analytics")
def analytics():
    if "user_id" not in session: return redirect(url_for("login"))
    user_preds = Prediction.query.filter_by(user_id=session["user_id"]).order_by(Prediction.timestamp.asc()).all()
    
    if not user_preds: 
        flash("No history found. Please make a prediction first.", "info")
        return redirect(url_for("home"))
    
    latest_features = json.loads(user_preds[-1].features_json)
    stats = {
        "hr": latest_features.get('thalach', 'N/A'),
        "avg_conf": round(sum(p.confidence for p in user_preds) / len(user_preds), 1),
        "total": len(user_preds)
    }
    
    fig = px.line(x=[p.timestamp for p in user_preds], 
                  y=[p.confidence for p in user_preds], 
                  title="Cardiac Health Risk Trend", markers=True)
    
    return render_template("analytics.html", trend_plot=pio.to_html(fig, full_html=False), stats=stats)

@app.route("/feature-importance")
def plotly_plot():
    if g.feat_imp_df is not None:
        fig = px.bar(g.feat_imp_df, x=g.feat_imp_df.columns[1], y=g.feat_imp_df.columns[0], 
                     orientation='h', title="Feature Importance Analysis")
        return render_template("plotly_plot.html", plot_html=pio.to_html(fig, full_html=False))
    return redirect(url_for("home"))

@app.route("/heatmap")
def heatmap_plot():
    if g.df is not None:
        corr = g.df.select_dtypes(include=[np.number]).corr()
        fig = px.imshow(corr, text_auto=".2f", title="Parameter Correlation Heatmap")
        return render_template("heatmap.html", heatmap_html=pio.to_html(fig, full_html=False))
    return redirect(url_for("home"))

@app.route("/distributions")
def distributions():
    if g.df is not None:
        fig = px.histogram(g.df, x="age", color="target", barmode="overlay", marginal="violin", 
                           title="Dataset Age Distribution")
        return render_template("distributions.html", plot_html=pio.to_html(fig, full_html=False))
    return redirect(url_for("home"))

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session: return redirect(url_for("login"))
    
    user_preds = Prediction.query.filter_by(user_id=session["user_id"]).all()
    pos = sum(1 for p in user_preds if "Detected" in p.prediction_result)
    neg = len(user_preds) - pos
    total_count = Prediction.query.count()
    
    stats = {'total': total_count, 'user_total': len(user_preds), 'positive': pos, 'negative': neg}
    feat_imp_labels = g.feat_imp_df.iloc[:, 0].tolist() if g.feat_imp_df is not None else []
    feat_imp_data = g.feat_imp_df.iloc[:, 1].tolist() if g.feat_imp_df is not None else []
    bar_labels = ["Total Records", "Your History", "High Risk", "Low Risk"]
    bar_data = [total_count, len(user_preds), pos, neg]

    return render_template("dashboard.html", stats=stats, pie_values=[neg, pos], pie_labels=["Healthy", "Risk"],
                           bar_labels=bar_labels, bar_data=bar_data,
                           feat_imp_labels=feat_imp_labels, feat_imp_data=feat_imp_data)

@app.route("/referrals")
def referrals():
    if "user_id" not in session: return redirect(url_for("login"))
    
    hospitals = [
        {"id": 1, "name": "Srinivas Multispeciality", "location": "Bangalore", "doctor": "Dr. V. K. Srinivas"},
        {"name": "Medanta - The Medicity", "location": "Gurugram", "doctor": "Dr. Naresh Trehan", "id": 2},
        {"name": "Apollo Hospitals", "location": "Chennai", "doctor": "Dr. Y. V. C. Reddy", "id": 3},
        {"name": "Fortis Memorial Research Institute", "location": "Gurugram", "doctor": "Dr. T. S. Kler", "id": 4}
    ]
    
    latest = Prediction.query.filter_by(user_id=session["user_id"]).order_by(Prediction.timestamp.desc()).first()
    return render_template("referrals.html", hospitals=hospitals, prediction=latest)

@app.route("/generate_referral/<int:hospital_id>")
def generate_referral(hospital_id):
    if "user_id" not in session: return redirect(url_for("login"))
    
    user = User.query.get(session["user_id"])
    latest_pred = Prediction.query.filter_by(user_id=user.id).order_by(Prediction.timestamp.desc()).first()
    
    hospitals = {
        1: {"name": "Srinivas Multispeciality", "doc": "Dr. V. K. Srinivas"},
        2: {"name": "Medanta - The Medicity", "doc": "Dr. Naresh Trehan"},
        3: {"name": "Apollo Hospitals", "doc": "Dr. Y. V. C. Reddy"},
        4: {"name": "Fortis Memorial", "doc": "Dr. T. S. Kler"}
    }
    hosp = hospitals.get(hospital_id, hospitals[1])

    # Data Prep for detailed referral
    features = json.loads(latest_pred.features_json) if latest_pred else {}
    is_positive = "Detected" in (latest_pred.prediction_result if latest_pred else "")

    # PDF Creation
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 18)
    pdf.set_text_color(180, 0, 0)
    pdf.cell(0, 10, "MEDICAL REFERRAL LETTER", ln=True, align='C')
    pdf.ln(5)
    
    # Details Section
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 7, f"Date: {datetime.now().strftime('%d-%m-%Y')}", ln=True)
    pdf.cell(0, 7, f"To: {hosp['doc']}", ln=True)
    pdf.cell(0, 7, f"Hospital: {hosp['name']}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 10, f" Patient: {user.username.upper()}", ln=True, fill=True)
    pdf.ln(5)
    
    # Clinical Diagnosis & Vitals
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Clinical Assessment & Observations:", ln=True)
    
    # Table Header
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(60, 8, "Parameter", 1)
    pdf.cell(60, 8, "Value", 1)
    pdf.cell(60, 8, "Reference", 1, ln=True)
    
    # Table Content
    pdf.set_font("Arial", '', 10)
    pdf.cell(60, 8, "Resting Blood Pressure", 1)
    pdf.cell(60, 8, f"{features.get('trestbps', 'N/A')} mmHg", 1)
    pdf.cell(60, 8, "120/80 mmHg", 1, ln=True)
    
    pdf.cell(60, 8, "Serum Cholesterol", 1)
    pdf.cell(60, 8, f"{features.get('chol', 'N/A')} mg/dl", 1)
    pdf.cell(60, 8, "< 200 mg/dl", 1, ln=True)
    
    pdf.cell(60, 8, "Max Heart Rate", 1)
    pdf.cell(60, 8, f"{features.get('thalach', 'N/A')} bpm", 1)
    pdf.cell(60, 8, "150-180 bpm", 1, ln=True)
    pdf.ln(10)
    
    # AI Insight & Management
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Diagnosis & Management Plan:", ln=True)
    pdf.set_font("Arial", '', 11)
    
    if latest_pred:
        analysis_text = (f"Screening Result: {latest_pred.prediction_result}\n"
                         f"AI Confidence: {latest_pred.confidence}%\n\n"
                         f"Recommendations:\n")
        if is_positive:
            analysis_text += ("- Immediate Cardiologist consultation advised.\n"
                              "- Potential initiation of Statins and Beta-Blockers.\n"
                              "- Low-sodium DASH diet recommended.")
        else:
            analysis_text += ("- Routine maintenance of heart health.\n"
                              "- Mediterranean diet and regular aerobic exercise.\n"
                              "- Omega-3 supplements for vascular support.")
    else:
        analysis_text = "Routine cardiac consultation and clinical check-up requested."
    
    pdf.multi_cell(0, 8, analysis_text)
    
    pdf.ln(20)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, "__________________________", ln=True)
    pdf.cell(0, 10, "Authorized AI System Signature", ln=True)

    # Output to binary stream
    pdf_output = pdf.output(dest='S')
    if isinstance(pdf_output, str):
        pdf_output = pdf_output.encode('latin-1')
        
    buffer = io.BytesIO(pdf_output)
    buffer.seek(0)

    response = make_response(buffer.getvalue())
    response.headers.set('Content-Disposition', 'attachment', filename=f'Referral_{user.username}.pdf')
    response.headers.set('Content-Type', 'application/pdf')
    return response

if __name__ == '__main__':
    with app.app_context(): 
        db.create_all()
    app.run(debug=True) give updated app.py
