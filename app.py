"""
MediRisk · Diabetic Readmission Dashboard
==========================================
User uploads ONE file: diabetic_data.csv

The two output files from the notebook are bundled at the top of this script.
Paste the paths to your local files in the CONFIG section below.

  model_predictions.xlsx          → loaded from disk (place in same folder)
  readmission_model_reloaded.joblib → loaded from disk (place in same folder)

No re-training happens. The app just reads your notebook outputs directly.

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import joblib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score
)

# ─────────────────────────────────────────────────────────
#  CONFIG — put your two output files in the same folder
#  as this app.py, that's all
# ─────────────────────────────────────────────────────────
XLSX_PATH   = "model_predictions.xlsx"
JOBLIB_PATH = "readmission_model_reloaded.joblib"

# Known metrics from your notebook run
KNOWN_METRICS = {
    "KNN":                 {"Accuracy": 0.5461, "AUC": 0.5528},
    "Decision Tree":       {"Accuracy": 0.5833, "AUC": 0.5814},
    "Random Forest":       {"Accuracy": 0.6611, "AUC": 0.7182},
    "Logistic Regression": {"Accuracy": 0.6272, "AUC": 0.6696},
    "GBM":                 {"Accuracy": 0.6632, "AUC": 0.7218},
    "XGBoost":             {"Accuracy": 0.6685, "AUC": 0.7304},
    "Naive Bayes":         {"Accuracy": 0.5894, "AUC": 0.6452},
    "Voting Classifier":   {"Accuracy": 0.6620, "AUC": 0.7195},
}

MODEL_COL = {
    "KNN":                 "KNN_Pred",
    "Decision Tree":       "DecisionTree_Pred",
    "Random Forest":       "RandomForest_Pred",
    "Logistic Regression": "LogisticRegression_Pred",
    "GBM":                 "GBM_Pred",
    "XGBoost":             "XGBoost_Pred",
    "Naive Bayes":         "NaiveBayes_Pred",
    "Voting Classifier":   "VotingClassifier_Pred",
}

# ─────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediRisk · Diabetic Readmission AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

html,body,[class*="css"]    { font-family:'DM Sans',sans-serif; background:#0d1117; color:#e2e8f0; }
.stApp                       { background:#0d1117; }
[data-testid="stSidebar"]    { background:linear-gradient(180deg,#111827 0%,#0d1117 100%); border-right:1px solid #1e293b; }
[data-testid="stSidebar"] *  { color:#cbd5e1 !important; }

.hero { background:linear-gradient(135deg,#0f2027 0%,#1a3a4a 50%,#0f2027 100%);
        border:1px solid #1e3a4a; border-radius:16px; padding:2.2rem 3rem;
        margin-bottom:1.8rem; position:relative; overflow:hidden; }
.hero::before { content:''; position:absolute; inset:0;
                background:radial-gradient(ellipse at 30% 50%,rgba(56,189,248,.09) 0%,transparent 70%);
                pointer-events:none; }
.hero-badge { display:inline-block; background:rgba(56,189,248,.12); border:1px solid rgba(56,189,248,.3);
              color:#38bdf8; font-family:'JetBrains Mono',monospace; font-size:.68rem;
              padding:3px 10px; border-radius:20px; margin-bottom:.7rem;
              letter-spacing:.1em; text-transform:uppercase; }
.hero-title { font-family:'DM Serif Display',serif; font-size:2.4rem; color:#f0f9ff; margin:0 0 .25rem; line-height:1.1; }
.hero-title span { color:#38bdf8; font-style:italic; }
.hero-sub { font-size:.95rem; color:#94a3b8; font-weight:300; margin:0; letter-spacing:.02em; }

.sg  { display:flex; gap:1rem; margin-bottom:1.8rem; flex-wrap:wrap; }
.sc  { flex:1; min-width:130px; background:#111827; border:1px solid #1e293b;
       border-radius:12px; padding:1.1rem 1.4rem; position:relative; overflow:hidden; }
.sc::after { content:''; position:absolute; top:0; left:0; width:3px; height:100%;
             background:var(--a,#38bdf8); border-radius:3px 0 0 3px; }
.sl { font-size:.7rem; color:#64748b; text-transform:uppercase; letter-spacing:.1em; margin-bottom:.35rem; }
.sv { font-family:'DM Serif Display',serif; font-size:1.9rem; color:#f1f5f9; line-height:1; }
.ss { font-size:.72rem; color:#475569; margin-top:.2rem; }

.sec-lbl   { font-family:'JetBrains Mono',monospace; font-size:.67rem; color:#38bdf8;
             letter-spacing:.15em; text-transform:uppercase; margin-bottom:.35rem; }
.sec-title { font-family:'DM Serif Display',serif; font-size:1.45rem; color:#f1f5f9; margin:0 0 1.3rem; }

.mc      { display:flex; align-items:center; background:#111827; border:1px solid #1e293b;
           border-radius:10px; padding:.85rem 1.1rem; margin-bottom:.5rem; gap:.9rem; }
.mc.best { border-color:#38bdf8; background:rgba(56,189,248,.06); }
.mc-name { flex:2; font-weight:500; font-size:.88rem; color:#e2e8f0; }
.mc-bw   { flex:3; }
.mc-bl   { font-size:.68rem; color:#64748b; margin-bottom:2px; }
.mc-tr   { background:#1e293b; border-radius:4px; height:7px; overflow:hidden; }
.mc-fi   { height:100%; border-radius:4px; }
.mc-sc   { font-family:'JetBrains Mono',monospace; font-size:.82rem; color:#94a3b8; min-width:46px; text-align:right; }

.rp     { border-radius:14px; padding:1.8rem; text-align:center; margin-bottom:1.4rem; }
.rp.hi  { background:linear-gradient(135deg,#2d1515,#3d1a1a); border:1px solid #7f1d1d; }
.rp.lo  { background:linear-gradient(135deg,#0d2818,#1a3d27); border:1px solid #14532d; }
.rp-lbl { font-size:.72rem; color:#94a3b8; letter-spacing:.12em; text-transform:uppercase; margin-bottom:.4rem; }
.rp-hi  { font-family:'DM Serif Display',serif; font-size:2rem; color:#fca5a5; }
.rp-lo  { font-family:'DM Serif Display',serif; font-size:2rem; color:#86efac; }
.rp-p   { font-family:'JetBrains Mono',monospace; font-size:.95rem; color:#94a3b8; margin-top:.4rem; }

.ebox   { background:#111827; border:1px solid #1e293b; border-radius:12px; padding:1.4rem; margin-top:.8rem; }
.ebox-t { font-family:'DM Serif Display',serif; font-size:1.15rem; color:#f1f5f9; margin-bottom:.25rem; }
.ebox-s { font-size:.8rem; color:#64748b; margin-bottom:.8rem; }

.stButton>button { background:linear-gradient(135deg,#0369a1,#0ea5e9)!important; color:#fff!important;
                   border:none!important; border-radius:8px!important; font-weight:600!important;
                   letter-spacing:.04em!important; padding:.6rem 1.5rem!important; }
.stTabs [data-baseweb="tab-list"] { background:#111827; border-radius:10px; padding:4px; gap:4px; }
.stTabs [data-baseweb="tab"]      { background:transparent; color:#64748b; border-radius:8px; font-size:.84rem; padding:8px 16px; }
.stTabs [aria-selected="true"]    { background:#1e3a4a!important; color:#38bdf8!important; }
.stSelectbox label,.stSlider label,.stTextInput label { color:#94a3b8!important; font-size:.8rem!important; }
hr { border-color:#1e293b; }
[data-testid="stFileUploader"] { background:#111827; border:2px dashed #1e3a4a; border-radius:12px; padding:.8rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
#  MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":"#111827","axes.facecolor":"#111827","axes.edgecolor":"#1e293b",
    "axes.labelcolor":"#94a3b8","xtick.color":"#64748b","ytick.color":"#64748b",
    "text.color":"#e2e8f0","grid.color":"#1e293b",
    "axes.spines.top":False,"axes.spines.right":False,
})

# ─────────────────────────────────────────────────────────
#  LOAD NOTEBOOK OUTPUT FILES  (no training)
# ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_xlsx():
    return pd.read_excel(XLSX_PATH)

@st.cache_data(show_spinner=False)
def load_joblib():
    return np.array(joblib.load(JOBLIB_PATH)).flatten()

def files_ready():
    return os.path.exists(XLSX_PATH) and os.path.exists(JOBLIB_PATH)

# ─────────────────────────────────────────────────────────
#  EDA HELPERS
# ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))

# ─────────────────────────────────────────────────────────
#  EMAIL
# ─────────────────────────────────────────────────────────
def send_email(to_email, name, from_email, app_pw):
    subject = "⚕️ Important: High Readmission Risk — Follow-up Care Recommended"
    body = f"""Dear {name},

Our AI clinical risk model has identified you as HIGH RISK for hospital readmission.

We strongly recommend scheduling a follow-up consultation within the next 7 days to:
  • Review your medications and insulin management
  • Monitor blood glucose and A1C levels
  • Address any emerging symptoms early

Please contact our scheduling team at your earliest convenience.

This is an automated message from the MediRisk Clinical AI System.

Warm regards,
Your Healthcare Team — MediRisk AI
"""
    try:
        msg = MIMEMultipart()
        msg["From"] = from_email; msg["To"] = to_email; msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        srv = smtplib.SMTP("smtp.gmail.com", 587)
        srv.starttls(); srv.login(from_email, app_pw)
        srv.sendmail(from_email, to_email, msg.as_string()); srv.quit()
        return True, "Sent ✓"
    except Exception as e:
        return False, str(e)

# ─────────────────────────────────────────────────────────
#  GAUGE CHART
# ─────────────────────────────────────────────────────────
def gauge(prob):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    t = np.linspace(np.pi, 0, 300); r, cx, cy = 0.35, 0.5, 0.28
    ax.fill_between(cx+r*np.cos(t), cy, cy+r*np.sin(t), color="#1e293b")
    ft = np.linspace(np.pi, np.pi-prob*np.pi, 300)
    col = "#f87171" if prob>.5 else "#fbbf24" if prob>.35 else "#4ade80"
    ax.fill_between(cx+r*np.cos(ft), cy, cy+r*np.sin(ft), color=col, alpha=.92)
    na = np.pi - prob*np.pi
    ax.annotate("", xy=(cx+.3*np.cos(na), cy+.3*np.sin(na)), xytext=(cx,cy),
                arrowprops=dict(arrowstyle="->", color="white", lw=2.5))
    ax.text(cx, cy+.28, f"{prob*100:.1f}%", ha="center", va="bottom",
            fontsize=20, fontweight="bold", color=col)
    ax.text(cx, cy+.14, "Readmission Risk", ha="center", fontsize=9, color="#64748b")
    ax.text(cx-r-.05, cy-.06, "0%",   ha="center", fontsize=8, color="#475569")
    ax.text(cx+r+.05, cy-.06, "100%", ha="center", fontsize=8, color="#475569")
    plt.tight_layout(); return fig

# ─────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sec-lbl">MediRisk · v2.0</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:DM Serif Display,serif;font-size:1.25rem;color:#f1f5f9;margin-bottom:1.2rem;">Clinical AI Dashboard</div>', unsafe_allow_html=True)
    csv_file = st.file_uploader("📂  Upload diabetic_data.csv", type=["csv"])
    st.markdown("---")

    if not files_ready():
        st.warning(f"Place these two files in the **same folder** as `app.py`:\n\n• `{XLSX_PATH}`\n• `{JOBLIB_PATH}`")

    page = st.radio("Navigate", [
        "🏠  Overview & EDA",
        "📊  Model Performance",
        "🔬  Prediction Explorer",
        "📧  Email Alert Centre",
    ])
    st.markdown("---")
    st.markdown('<div style="font-size:.71rem;color:#475569;line-height:1.8;">Dataset: UCI Diabetic Readmission<br>Records: 101,766 patients<br>Best model: XGBoost (AUC 0.7304)<br>Predictions loaded from .xlsx<br>RF alerts loaded from .joblib</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">🩺 AI-Powered Clinical Decision Support</div>
  <div class="hero-title">Medi<span>Risk</span></div>
  <p class="hero-sub">Diabetic patient readmission analysis &nbsp;·&nbsp; 8 ML models &nbsp;·&nbsp; Automated care alerts</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
#  GATE
# ─────────────────────────────────────────────────────────
if not csv_file:
    st.markdown("""
    <div style="background:#111827;border:1px solid #1e293b;border-radius:16px;
         padding:3rem;text-align:center;max-width:520px;margin:0 auto;">
      <div style="font-size:3rem;margin-bottom:1rem;">🩺</div>
      <div style="font-family:'DM Serif Display',serif;font-size:1.4rem;color:#f1f5f9;margin-bottom:.6rem;">
        Upload <code style="background:#1e293b;padding:2px 8px;border-radius:5px;color:#38bdf8;">diabetic_data.csv</code>
      </div>
      <div style="color:#64748b;font-size:.88rem;margin-bottom:1.8rem;">
        Only the raw dataset is needed here. Model predictions and alerts
        are loaded automatically from the notebook output files.
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:.8rem;text-align:left;max-width:400px;margin:0 auto;">
        <div style="background:#0f2027;border:1px solid #1e3a4a;border-radius:8px;padding:.85rem 1rem;">
          <div style="color:#38bdf8;font-size:.78rem;font-weight:600;margin-bottom:.3rem;">📊 EDA & Insights</div>
          <div style="color:#64748b;font-size:.73rem;">From diabetic_data.csv</div>
        </div>
        <div style="background:#0f2027;border:1px solid #1e3a4a;border-radius:8px;padding:.85rem 1rem;">
          <div style="color:#38bdf8;font-size:.78rem;font-weight:600;margin-bottom:.3rem;">🤖 Model Results</div>
          <div style="color:#64748b;font-size:.73rem;">From model_predictions.xlsx</div>
        </div>
        <div style="background:#0f2027;border:1px solid #1e3a4a;border-radius:8px;padding:.85rem 1rem;">
          <div style="color:#38bdf8;font-size:.78rem;font-weight:600;margin-bottom:.3rem;">🔬 Explorer</div>
          <div style="color:#64748b;font-size:.73rem;">Row-level prediction analysis</div>
        </div>
        <div style="background:#0f2027;border:1px solid #1e3a4a;border-radius:8px;padding:.85rem 1rem;">
          <div style="color:#38bdf8;font-size:.78rem;font-weight:600;margin-bottom:.3rem;">📧 Email Alerts</div>
          <div style="color:#64748b;font-size:.73rem;">From readmission_model_reloaded.joblib</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────
raw_df = load_csv(csv_file.read())

# Load notebook outputs if files exist
if files_ready():
    pred_df  = load_xlsx()
    rf_preds = load_joblib()
else:
    pred_df  = None
    rf_preds = None

# ─────────────────────────────────────────────────────────
#  BUILD METRICS TABLE  from pred_df + KNOWN_METRICS
# ─────────────────────────────────────────────────────────
def build_metrics(pred_df):
    rows = []
    true = pred_df["True_Label"].values
    for name, col in MODEL_COL.items():
        if col not in pred_df.columns:
            continue
        pred = pred_df[col].values
        rows.append({
            "Model":     name,
            "Accuracy":  KNOWN_METRICS[name]["Accuracy"],
            "AUC":       KNOWN_METRICS[name]["AUC"],
            "Precision": round(precision_score(true, pred, average="binary", zero_division=0), 4),
            "Recall":    round(recall_score(true, pred, average="binary", zero_division=0), 4),
            "F1":        round(f1_score(true, pred, average="binary", zero_division=0), 4),
        })
    return pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)

if pred_df is not None:
    metrics_df = build_metrics(pred_df)
    best_model = metrics_df.iloc[0]["Model"]
    true_labels = pred_df["True_Label"].values
else:
    metrics_df  = None
    best_model  = "XGBoost"
    true_labels = None

# ═════════════════════════════════════════════════════════
#  PAGE: OVERVIEW & EDA
# ═════════════════════════════════════════════════════════
if page == "🏠  Overview & EDA":
    total  = len(raw_df)
    n_feat = raw_df.shape[1] - 1
    rc     = raw_df["readmitted"].value_counts()

    st.markdown('<div class="sec-lbl">Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">At a Glance</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sg">
      <div class="sc" style="--a:#38bdf8;">
        <div class="sl">Total Records</div>
        <div class="sv">{total:,}</div>
        <div class="ss">diabetic_data.csv</div>
      </div>
      <div class="sc" style="--a:#f87171;">
        <div class="sl">Readmitted (&lt;30 + &gt;30)</div>
        <div class="sv">{rc.get('<30',0) + rc.get('>30',0):,}</div>
        <div class="ss">{(rc.get('<30',0)+rc.get('>30',0))/total*100:.1f}% of patients</div>
      </div>
      <div class="sc" style="--a:#4ade80;">
        <div class="sl">Not Readmitted (NO)</div>
        <div class="sv">{rc.get('NO',0):,}</div>
        <div class="ss">{rc.get('NO',0)/total*100:.1f}% of patients</div>
      </div>
      <div class="sc" style="--a:#a78bfa;">
        <div class="sl">Features</div>
        <div class="sv">{n_feat}</div>
        <div class="ss">Clinical variables</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["  Distributions  ", "  Target Analysis  ", "  Correlation  "])

    with tab1:
        num_feats = [c for c in ["time_in_hospital","num_lab_procedures","num_medications",
                                  "number_diagnoses","num_procedures",
                                  "number_outpatient","number_inpatient","number_emergency"]
                     if c in raw_df.columns]
        colors = ["#38bdf8","#818cf8","#4ade80","#f472b6","#fb923c","#a78bfa","#34d399","#fbbf24"]
        fig, axes = plt.subplots(2, 4, figsize=(16, 7))
        axes = axes.flatten()
        for i, (feat, color) in enumerate(zip(num_feats, colors)):
            axes[i].hist(raw_df[feat].dropna(), bins=25, color=color, alpha=.85, edgecolor="none")
            axes[i].set_title(feat.replace("_"," ").title(), fontsize=10, pad=8)
            axes[i].axvline(raw_df[feat].mean(), color="white", linestyle="--", lw=1, alpha=.5)
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout(pad=2); st.pyplot(fig)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5,4))
            vals  = raw_df["readmitted"].value_counts()
            bars  = ax.bar(vals.index, vals.values,
                           color=["#f87171","#fbbf24","#4ade80"], edgecolor="none", width=.45)
            ax.set_ylabel("Count", labelpad=8); ax.set_title("Readmission Categories", pad=10)
            ax.yaxis.grid(True, alpha=.3); ax.set_axisbelow(True)
            for b,v in zip(bars, vals.values):
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+300,
                        f"{v:,}", ha="center", fontsize=10, fontweight="bold")
            plt.tight_layout(); st.pyplot(fig)

        with c2:
            if "age" in raw_df.columns:
                fig, ax = plt.subplots(figsize=(5,4))
                age_order = ["[0-10)","[10-20)","[20-30)","[30-40)","[40-50)",
                             "[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"]
                age_read = (raw_df[raw_df["readmitted"]!="NO"]["age"]
                            .value_counts().reindex(age_order, fill_value=0))
                ax.bar(range(len(age_order)), age_read.values,
                       color="#38bdf8", edgecolor="none", width=.65)
                ax.set_xticks(range(len(age_order)))
                ax.set_xticklabels(age_order, rotation=35, ha="right", fontsize=8)
                ax.set_ylabel("Readmitted Count", labelpad=8)
                ax.set_title("Readmissions by Age Group", pad=10)
                ax.yaxis.grid(True, alpha=.3); ax.set_axisbelow(True)
                plt.tight_layout(); st.pyplot(fig)

        if "time_in_hospital" in raw_df.columns:
            fig, ax = plt.subplots(figsize=(12,4))
            for label, color, name in [("NO","#4ade80","Not Readmitted"),
                                        ("<30","#fbbf24","Readmitted <30 days"),
                                        (">30","#f87171","Readmitted >30 days")]:
                ax.hist(raw_df[raw_df["readmitted"]==label]["time_in_hospital"].dropna(),
                        bins=14, alpha=.6, color=color, label=name, edgecolor="none")
            ax.set_xlabel("Days in Hospital", labelpad=8); ax.set_ylabel("Count", labelpad=8)
            ax.set_title("Time in Hospital vs Readmission", pad=10)
            ax.legend(framealpha=0); ax.yaxis.grid(True, alpha=.3); ax.set_axisbelow(True)
            plt.tight_layout(); st.pyplot(fig)

    with tab3:
        # label-encode for correlation
        df_enc = raw_df.copy().replace("?", np.nan)
        le = LabelEncoder()
        for col in df_enc.select_dtypes(include="object").columns:
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        top_feats = df_enc.corr()["readmitted"].abs().nlargest(14).index.tolist()
        fig, ax = plt.subplots(figsize=(11, 8))
        sns.heatmap(df_enc[top_feats].corr(), annot=True, fmt=".2f",
                    cmap=sns.diverging_palette(220,20,as_cmap=True),
                    linewidths=.5, linecolor="#0d1117", ax=ax,
                    annot_kws={"size":8}, vmin=-1, vmax=1, cbar_kws={"shrink":.8})
        ax.set_title("Top 14 Features — Correlation Matrix", pad=12)
        plt.tight_layout(); st.pyplot(fig)

    st.markdown('<div class="sec-lbl" style="margin-top:.5rem;">Raw Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(raw_df.head(20), use_container_width=True, height=350)

# ═════════════════════════════════════════════════════════
#  PAGE: MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════
elif page == "📊  Model Performance":
    if pred_df is None:
        st.error(f"Place `{XLSX_PATH}` in the same folder as app.py and restart.")
        st.stop()

    st.markdown('<div class="sec-lbl">Results from model_predictions.xlsx</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">8 Models — Head to Head</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    for _, row in metrics_df.iterrows():
        is_best = row["Model"] == best_model
        cls  = "mc best" if is_best else "mc"
        icon = "👑 " if is_best else ""
        aw   = int(row["Accuracy"] * 200)
        auew = int(row["AUC"]      * 200)
        ac   = "#38bdf8" if is_best else "#334155"
        rc   = "#4ade80" if is_best else "#334155"
        st.markdown(f"""
        <div class="{cls}">
          <div class="mc-name">{icon}{row["Model"]}</div>
          <div class="mc-bw">
            <div class="mc-bl">Accuracy</div>
            <div class="mc-tr"><div class="mc-fi" style="width:{aw}px;max-width:100%;background:{ac};"></div></div>
          </div>
          <div class="mc-sc">{row["Accuracy"]}</div>
          <div class="mc-bw">
            <div class="mc-bl">AUC-ROC</div>
            <div class="mc-tr"><div class="mc-fi" style="width:{auew}px;max-width:100%;background:{rc};"></div></div>
          </div>
          <div class="mc-sc">{row["AUC"]}</div>
          <div class="mc-sc" style="color:#64748b;">F1 {row["F1"]}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["  Confusion Matrix  ", "  Metrics Chart  ", "  Agreement Matrix  "])

    with tab1:
        sel = st.selectbox("Select model", list(MODEL_COL.keys()),
                           index=list(MODEL_COL.keys()).index(best_model))
        preds = pred_df[MODEL_COL[sel]].values
        c1, c2 = st.columns(2)
        with c1:
            cm = confusion_matrix(true_labels, preds)
            fig, ax = plt.subplots(figsize=(5,4.2))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        linewidths=2, linecolor="#0d1117",
                        annot_kws={"size":14,"weight":"bold"},
                        xticklabels=["Not Readmitted","Readmitted"],
                        yticklabels=["Not Readmitted","Readmitted"])
            ax.set_ylabel("Actual",labelpad=8); ax.set_xlabel("Predicted",labelpad=8)
            ax.set_title(f"Confusion Matrix — {sel}", pad=12)
            plt.tight_layout(); st.pyplot(fig)
        with c2:
            cr = classification_report(true_labels, preds,
                                       target_names=["Not Readmitted","Readmitted"],
                                       output_dict=True)
            st.dataframe(pd.DataFrame(cr).T.round(4), use_container_width=True)

    with tab2:
        fig, axes = plt.subplots(1, 4, figsize=(16,5))
        for ax, metric in zip(axes, ["Accuracy","AUC","F1","Recall"]):
            s = metrics_df.sort_values(metric)
            cols_ = ["#38bdf8" if n==best_model else "#1e3a4a" for n in s["Model"]]
            ax.barh(s["Model"], s[metric], color=cols_, edgecolor="none", height=.55)
            ax.set_title(metric, pad=8)
            ax.xaxis.grid(True, alpha=.3); ax.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig)

    with tab3:
        ml = list(MODEL_COL.keys())
        agree = pd.DataFrame(index=ml, columns=ml, dtype=float)
        for a in ml:
            for b in ml:
                agree.loc[a,b] = (pred_df[MODEL_COL[a]].values==pred_df[MODEL_COL[b]].values).mean()
        fig, ax = plt.subplots(figsize=(9,7))
        sns.heatmap(agree.astype(float), annot=True, fmt=".2f", cmap="YlGnBu",
                    linewidths=1, linecolor="#0d1117", ax=ax,
                    annot_kws={"size":9}, vmin=.4, vmax=1.0, cbar_kws={"shrink":.7})
        ax.set_title("Model Prediction Agreement Matrix", pad=12)
        plt.tight_layout(); st.pyplot(fig)

# ═════════════════════════════════════════════════════════
#  PAGE: PREDICTION EXPLORER
# ═════════════════════════════════════════════════════════
elif page == "🔬  Prediction Explorer":
    if pred_df is None:
        st.error(f"Place `{XLSX_PATH}` in the same folder as app.py and restart.")
        st.stop()

    st.markdown('<div class="sec-lbl">Row-level Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Prediction Explorer</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1: tf = st.selectbox("True Label", ["All","Readmitted (1)","Not Readmitted (0)"])
    with c2: fm = st.selectbox("Focus Model", list(MODEL_COL.keys()),
                               index=list(MODEL_COL.keys()).index(best_model))
    with c3: cf = st.selectbox("Correctness", ["All","Correct only","Wrong only"])

    vdf = pred_df.copy()
    vdf["Correct"] = (vdf["True_Label"]==vdf[MODEL_COL[fm]]).map({True:"✓",False:"✗"})
    if tf=="Readmitted (1)":       vdf = vdf[vdf["True_Label"]==1]
    elif tf=="Not Readmitted (0)": vdf = vdf[vdf["True_Label"]==0]
    if cf=="Correct only": vdf = vdf[vdf["Correct"]=="✓"]
    elif cf=="Wrong only": vdf = vdf[vdf["Correct"]=="✗"]

    st.markdown(f'<div style="color:#64748b;font-size:.82rem;margin-bottom:.6rem;">Showing {len(vdf):,} rows</div>', unsafe_allow_html=True)
    dcols = ["True_Label",MODEL_COL[fm],"Correct"]+[c for c in MODEL_COL.values() if c!=MODEL_COL[fm]]
    st.dataframe(vdf[dcols].head(500), use_container_width=True, height=400)

    st.markdown('<div class="sec-lbl" style="margin-top:1rem;">Inspect Single Row</div>', unsafe_allow_html=True)
    idx   = st.slider("Row index", 0, len(pred_df)-1, 0)
    row   = pred_df.iloc[idx]
    votes = {n: int(row[c]) for n,c in MODEL_COL.items()}
    n_hi  = sum(v==1 for v in votes.values())
    cons  = n_hi/len(votes)
    tl    = int(row["True_Label"])

    ca, cb = st.columns(2)
    with ca:
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1e293b;border-radius:12px;padding:1.2rem 1.5rem;">
          <div class="sec-lbl">Row {idx} — All Model Votes</div>
          <div style="display:flex;gap:2rem;margin:.8rem 0;">
            <div><div style="font-size:.7rem;color:#64748b;">True Label</div>
              <div style="font-family:JetBrains Mono,monospace;font-weight:600;font-size:1rem;color:{'#f87171' if tl==1 else '#4ade80'};">
                {'Readmitted' if tl==1 else 'Not Readmitted'}</div></div>
            <div><div style="font-size:.7rem;color:#64748b;">Voting HIGH</div>
              <div style="font-family:JetBrains Mono,monospace;font-weight:600;font-size:1rem;color:#38bdf8;">{n_hi}/{len(votes)}</div></div>
            <div><div style="font-size:.7rem;color:#64748b;">Consensus</div>
              <div style="font-family:JetBrains Mono,monospace;font-weight:600;font-size:1rem;color:{'#f87171' if cons>.5 else '#4ade80'};">{cons*100:.0f}%</div></div>
          </div>
        """, unsafe_allow_html=True)
        for mname, vote in votes.items():
            col_ = "#f87171" if vote==1 else "#4ade80"
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;background:#0d1117;
                 border:1px solid #1e293b;border-radius:7px;padding:.5rem .9rem;margin-bottom:.35rem;">
              <span style="font-size:.84rem;">{mname}</span>
              <span style="font-family:JetBrains Mono,monospace;font-size:.8rem;color:{col_};font-weight:600;">
                {'HIGH' if vote==1 else 'LOW'}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with cb:
        st.pyplot(gauge(cons))

# ═════════════════════════════════════════════════════════
#  PAGE: EMAIL ALERT CENTRE
# ═════════════════════════════════════════════════════════
elif page == "📧  Email Alert Centre":
    if rf_preds is None:
        st.error(f"Place `{JOBLIB_PATH}` in the same folder as app.py and restart.")
        st.stop()

    st.markdown('<div class="sec-lbl">Powered by readmission_model_reloaded.joblib</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">High-Risk Patient Email Alerts</div>', unsafe_allow_html=True)

    # rf_preds = rf_pred array saved by joblib.dump(rf_pred, ...)
    total_  = len(rf_preds)
    n_high  = int((rf_preds==1).sum())
    n_low   = total_ - n_high

    # align with pred_df true labels if available
    if pred_df is not None:
        n = min(len(rf_preds), len(pred_df))
        alert_df       = pred_df.iloc[:n].copy()
        alert_df["RF_Pred"] = rf_preds[:n].astype(int)
        high_df        = alert_df[alert_df["RF_Pred"]==1].reset_index(drop=True)
        tp = int(((rf_preds[:n]==1)&(true_labels[:n]==1)).sum())
        fp = int(((rf_preds[:n]==1)&(true_labels[:n]==0)).sum())
        prec = round(precision_score(true_labels[:n], rf_preds[:n],
                                     average="binary", zero_division=0), 4)
    else:
        high_df = pd.DataFrame({"RF_Pred": rf_preds[rf_preds==1]})
        tp = fp = 0; prec = 0.0

    st.markdown(f"""
    <div class="sg">
      <div class="sc" style="--a:#38bdf8;">
        <div class="sl">Test Patients</div>
        <div class="sv">{total_:,}</div>
        <div class="ss">from .joblib predictions</div>
      </div>
      <div class="sc" style="--a:#f87171;">
        <div class="sl">High-Risk (RF=1)</div>
        <div class="sv">{n_high:,}</div>
        <div class="ss">{n_high/total_*100:.1f}% — alert candidates</div>
      </div>
      <div class="sc" style="--a:#4ade80;">
        <div class="sl">Low-Risk (RF=0)</div>
        <div class="sv">{n_low:,}</div>
        <div class="ss">{n_low/total_*100:.1f}% — no alert needed</div>
      </div>
      <div class="sc" style="--a:#a78bfa;">
        <div class="sl">RF Precision</div>
        <div class="sv">{prec}</div>
        <div class="ss">Of alerts, truly readmitted</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5,3.8))
        ax.bar(["High Risk","Low Risk"], [n_high,n_low],
               color=["#f87171","#4ade80"], edgecolor="none", width=.45)
        ax.set_title("RF Prediction Distribution",pad=10)
        ax.yaxis.grid(True,alpha=.3); ax.set_axisbelow(True)
        for x,v in enumerate([n_high,n_low]):
            ax.text(x, v+50, f"{v:,}", ha="center", fontsize=10, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(5,3.8))
        ax.bar(["True Positives\n(correctly flagged)","False Positives\n(over-flagged)"],
               [tp,fp], color=["#4ade80","#fbbf24"], edgecolor="none", width=.45)
        ax.set_title("High-Risk Alert Breakdown",pad=10)
        ax.yaxis.grid(True,alpha=.3); ax.set_axisbelow(True)
        for x,v in enumerate([tp,fp]):
            ax.text(x, v+30, f"{v:,}", ha="center", fontsize=10, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig)

    st.markdown('<div class="sec-lbl" style="margin-top:.5rem;">High-Risk Records</div>', unsafe_allow_html=True)
    st.dataframe(high_df.head(200), use_container_width=True, height=280)

    # ── Email section ──────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div class="ebox">
      <div class="ebox-t">📧 Send Automated Alert Emails</div>
      <div class="ebox-s">Configure Gmail credentials and send personalised care alerts to high-risk patients.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    g1, g2 = st.columns(2)
    with g1: sender = st.text_input("Sender Gmail", placeholder="clinic@gmail.com")
    with g2: apppw  = st.text_input("Gmail App Password", type="password",
                                    placeholder="xxxx xxxx xxxx xxxx")
    st.markdown('<div style="font-size:.75rem;color:#475569;margin-bottom:1rem;">ℹ️ Use a Gmail App Password — not your regular password. <a href="https://myaccount.google.com/apppasswords" target="_blank" style="color:#38bdf8;">Generate here ↗</a></div>', unsafe_allow_html=True)

    mode = st.radio("Alert mode", ["Single patient","All high-risk patients"], horizontal=True)

    if mode == "Single patient":
        s1, s2 = st.columns(2)
        with s1:
            sp_name  = st.text_input("Patient Name",  placeholder="e.g. Ravi Kumar")
            sp_email = st.text_input("Patient Email", placeholder="patient@example.com")
        with s2:
            sp_idx = st.number_input("Row from high-risk table", min_value=0,
                                     max_value=max(n_high-1,0), value=0)
            tl_val = int(high_df.iloc[sp_idx]["True_Label"]) if "True_Label" in high_df.columns else "?"
            st.markdown(f"""
            <div style="background:#111827;border:1px solid #1e293b;border-radius:8px;padding:.75rem 1rem;margin-top:.4rem;">
              <div style="font-size:.74rem;color:#64748b;">
                Row {sp_idx} · RF Pred = 1 · True Label = {tl_val}
              </div>
            </div>""", unsafe_allow_html=True)

        if st.button("🚨 Send Alert to This Patient", use_container_width=True):
            if not all([sp_name, sp_email, sender, apppw]):
                st.warning("Fill in all fields.")
            else:
                with st.spinner(f"Sending to {sp_name}..."):
                    ok, msg = send_email(sp_email, sp_name, sender, apppw)
                if ok:
                    st.success(f"✅ Alert sent to **{sp_name}** ({sp_email})")
                    st.balloons()
                else:
                    st.error(f"❌ {msg}")
    else:
        has_name  = "name"  in high_df.columns
        has_email = "email" in high_df.columns
        if not (has_name and has_email):
            st.info("Bulk mode needs `name` and `email` columns in the predictions file. Add them to your dataset before running the notebook.")
        else:
            st.dataframe(high_df[["name","email","True_Label","RF_Pred"]].head(20), use_container_width=True)
            if st.button(f"🚨 Send to All {n_high:,} High-Risk Patients", use_container_width=True):
                if not all([sender, apppw]):
                    st.warning("Enter Gmail credentials first.")
                else:
                    prog = st.progress(0); log = []
                    for i, r in high_df.iterrows():
                        ok, msg = send_email(r["email"], r["name"], sender, apppw)
                        log.append({"Name":r["name"],"Email":r["email"],
                                    "Status":"✅ Sent" if ok else f"❌ {msg}"})
                        prog.progress((i+1)/n_high)
                    sent = sum(1 for r in log if "✅" in r["Status"])
                    st.success(f"✅ {sent}/{n_high} emails sent.")
                    st.dataframe(pd.DataFrame(log), use_container_width=True)