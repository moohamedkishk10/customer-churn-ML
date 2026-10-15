import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.predict import THRESHOLD, predict_customer  # noqa: E402

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root { color-scheme: light; }
html, body, .stApp { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(circle at 10% 6%, rgba(8,145,178,.20), transparent 38%),
        radial-gradient(circle at 92% 12%, rgba(249,115,22,.18), transparent 36%),
        radial-gradient(circle at 78% 94%, rgba(132,204,22,.16), transparent 42%),
        #F6F1E7 !important;
}
[data-testid="stAppViewContainer"], [data-testid="stMain"] { background: transparent !important; }
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 960px; padding-top: 2rem; padding-bottom: 4rem; }

.hero-title {
    font-size: 2.2rem; font-weight: 800; letter-spacing: -.03em;
    margin: 1.4rem 0 .2rem;
    background: linear-gradient(90deg, #0E7490, #0891B2 45%, #F97316);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub { color: #5B6472; margin-bottom: 1.6rem; }

[class*="st-key-card-"] {
    background: rgba(255,255,255,.62);
    border: 1px solid rgba(31,41,55,.08);
    border-radius: 18px;
    padding: 1.3rem 1.6rem 1.1rem;
    margin-bottom: 1rem;
    box-shadow: 0 10px 30px rgba(31,41,55,.06);
}
.section-title {
    display: flex; align-items: center; gap: .6rem;
    font-weight: 700; font-size: 1.05rem; color: #1F2937;
    margin-bottom: .6rem;
}
.section-title .dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: linear-gradient(135deg, #0891B2, #F97316);
}

.stApp label p { color: #1F2937 !important; font-weight: 600; font-size: .88rem; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: #5B6472 !important; }

/* select boxes: fixed background AND fixed text color, in any theme */
div[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border: 1px solid #D9CDB6 !important;
    border-radius: 10px !important;
}
div[data-baseweb="select"] div,
div[data-baseweb="select"] span,
div[data-baseweb="select"] input {
    color: #1F2937 !important;
    -webkit-text-fill-color: #1F2937 !important;
}
div[data-baseweb="select"] svg { fill: #5B6472 !important; }

/* dropdown list (rendered outside the page container) */
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] li,
div[data-baseweb="menu"] {
    background: #FFFFFF !important;
    color: #1F2937 !important;
}
div[data-baseweb="popover"] li:hover { background: #E6F4F7 !important; }

/* sliders */
[data-testid="stSliderThumbValue"] { color: #0E7490 !important; font-weight: 700; }
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color: #5B6472 !important; }

.stButton > button { width: 100%; }
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #0891B2, #14B8A6) !important;
    border: none !important; border-radius: 12px !important;
    padding: .7rem 1.4rem; font-weight: 700;
    box-shadow: 0 8px 20px rgba(8,145,178,.28);
    transition: transform .15s ease, box-shadow .15s ease;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 26px rgba(8,145,178,.36);
}
.stButton > button[kind="primary"] p { color: #FFFFFF !important; }

.result {
    border-radius: 20px; padding: 1.5rem 1.7rem; margin-top: 1.2rem;
    border: 1px solid rgba(31,41,55,.08);
    box-shadow: 0 14px 34px rgba(31,41,55,.08);
}
.result.stay { background: linear-gradient(135deg, rgba(20,184,166,.18), rgba(255,255,255,.85)); }
.result.leave { background: linear-gradient(135deg, rgba(249,115,22,.20), rgba(255,255,255,.85)); }
.result-top { display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap; }
.result-label { font-size: .8rem; letter-spacing: .08em; text-transform: uppercase; color: #5B6472; font-weight: 600; }
.result-value { font-size: 3rem; font-weight: 800; letter-spacing: -.03em; line-height: 1.1; }
.result.stay .result-value { color: #0F766E; }
.result.leave .result-value { color: #C2410C; }
.result-badge { padding: .45rem 1rem; border-radius: 999px; font-weight: 700; font-size: .9rem; color: #FFFFFF; }
.result.stay .result-badge { background: #0F766E; }
.result.leave .result-badge { background: #C2410C; }
.bar { position: relative; height: 12px; border-radius: 999px; background: rgba(31,41,55,.10); margin-top: 1.2rem; }
.bar-fill { height: 100%; border-radius: 999px; }
.result.stay .bar-fill { background: linear-gradient(90deg, #5EEAD4, #0F766E); }
.result.leave .bar-fill { background: linear-gradient(90deg, #FDBA74, #C2410C); }
.bar-mark { position: absolute; top: -4px; width: 2px; height: 20px; background: #1F2937; }
.bar-scale { display: flex; justify-content: space-between; font-size: .75rem; color: #5B6472; margin-top: .4rem; }
.result-sub { margin-top: .9rem; color: #374151; font-size: .92rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

LOGO_HTML = """
<style>body{margin:0;background:transparent}</style>
<div style="margin:0;padding:0;background:#050508;border-radius:16px;display:flex;align-items:center;justify-content:center;height:200px;overflow:hidden;font-family:'Inter',sans-serif;">
  <style>
    .wordmark{font-size:56px;font-weight:700;letter-spacing:-2px;display:inline-flex;align-items:baseline;direction:ltr}
    .wm-K{color:#e2e8f0;font-weight:900;opacity:0;animation:revUp .7s ease .2s forwards}
    .wm-e{font-weight:300;color:#06b6d4;margin-left:-3px;opacity:0;animation:revUp .7s ease .3s forwards}
    .wm-Y{font-weight:900;color:#06b6d4;opacity:0;animation:revUp .7s ease .4s forwards}
    .s-wrap{position:relative;display:inline-block;width:52px;height:78px;margin:0 6px;vertical-align:baseline;opacity:0;animation:revUp .7s ease .5s forwards}
    .s-top{position:absolute;top:0;left:0;width:100%;height:42%;overflow:hidden;font-size:56px;font-weight:700;line-height:1;color:#1e40af;clip-path:polygon(0 0,100% 0,100% 70%,0 100%)}
    .s-top::before{content:'S';position:absolute;top:0;left:0}
    .s-bot{position:absolute;bottom:0;left:0;width:100%;height:42%;overflow:hidden;font-size:56px;font-weight:700;line-height:1;color:#0d9488;clip-path:polygon(0 30%,100% 0,100% 100%,0 100%)}
    .s-bot::before{content:'S';position:absolute;bottom:0;left:0}
    .glow-line{position:absolute;top:50%;left:0;width:100%;height:2px;background:linear-gradient(90deg,transparent,rgba(6,182,212,.9),rgba(20,184,166,.9),transparent);transform:translateY(-50%);animation:glowPulse 2s ease-in-out infinite}
    .the-key{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:70px;height:20px;opacity:0;animation:keySlide 1s cubic-bezier(.34,1.56,.64,1) .7s forwards}
    .key-head{position:absolute;left:0;top:50%;transform:translateY(-50%);width:20px;height:20px;border:3px solid #06b6d4;border-radius:50%;background:rgba(6,182,212,.15);box-shadow:0 0 20px rgba(6,182,212,.4)}
    .key-shaft{position:absolute;left:38%;top:50%;transform:translateY(-50%);width:50%;height:5px;background:linear-gradient(90deg,#06b6d4,#14b8a6,#06b6d4);border-radius:4px;box-shadow:0 0 18px rgba(6,182,212,.5)}
    .key-teeth{position:absolute;right:0;top:50%;transform:translateY(-50%);display:flex;gap:2px;align-items:flex-end}
    .key-teeth span{display:block;width:5px;background:linear-gradient(180deg,#14b8a6,#06b6d4);border-radius:0 0 2px 2px}
    .key-teeth span:nth-child(1){height:10px} .key-teeth span:nth-child(2){height:16px}
    .key-teeth span:nth-child(3){height:7px} .key-teeth span:nth-child(4){height:13px}
    .wm-h{font-weight:600;color:#06b6d4;margin-left:-3px;opacity:0;animation:revUp .7s ease .85s forwards}
    .wm-k{font-weight:900;color:#e2e8f0;opacity:0;animation:revUp .7s ease 1s forwards}
    .tagline{display:block;text-align:center;margin-top:18px;font-size:10px;letter-spacing:8px;text-transform:uppercase;color:#334155;opacity:0;animation:revUp .6s ease 1.2s forwards}
    @keyframes revUp{0%{opacity:0;transform:translateY(24px)}100%{opacity:1;transform:none}}
    @keyframes keySlide{0%{opacity:0;transform:translate(-140%,-50%) scale(.7)}100%{opacity:1;transform:translate(-50%,-50%) scale(1)}}
    @keyframes glowPulse{0%,100%{opacity:.4}50%{opacity:1}}
  </style>
  <div style="text-align:center">
    <div class="wordmark">
      <span class="wm-K">K</span><span class="wm-e">e</span><span class="wm-Y">Y</span>
      <div class="s-wrap">
        <div class="s-top"></div><div class="glow-line"></div><div class="s-bot"></div>
        <div class="the-key"><div class="key-head"></div><div class="key-shaft"></div>
          <div class="key-teeth"><span></span><span></span><span></span><span></span></div>
        </div>
      </div>
      <span class="wm-h">h</span><span class="wm-k">k</span>
    </div>
    <span class="tagline">Key Check Tech</span>
  </div>
</div>
"""


def section_title(text):
    st.markdown(
        f'<div class="section-title"><span class="dot"></span>{text}</div>',
        unsafe_allow_html=True,
    )


def render_result(probability, will_churn):
    pct = probability * 100
    thr = THRESHOLD * 100
    tone = "leave" if will_churn else "stay"
    headline = "Likely to leave" if will_churn else "Likely to stay"
    if will_churn:
        sub = "The model flags this customer as at risk of churning."
    else:
        sub = "The model does not flag this customer as at risk of churning."

    html = (
        f'<div class="result {tone}">'
        f'<div class="result-top"><div>'
        f'<div class="result-label">Churn probability</div>'
        f'<div class="result-value">{pct:.1f}%</div></div>'
        f'<div class="result-badge">{headline}</div></div>'
        f'<div class="bar"><div class="bar-fill" style="width:{pct:.1f}%"></div>'
        f'<div class="bar-mark" style="left:{thr:.0f}%"></div></div>'
        f'<div class="bar-scale"><span>0%</span><span>100%</span></div>'
        f'<div class="result-sub">{sub} The dark marker on the bar is the decision threshold ({thr:.0f}%).</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


components.html(LOGO_HTML, height=200)

st.markdown('<div class="hero-title">Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Enter the customer details to estimate the probability that the customer will leave.</div>',
    unsafe_allow_html=True,
)

with st.container(key="card-customer"):
    section_title("Customer")
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        partner = st.selectbox("Partner", ["No", "Yes"])
    with c2:
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])

with st.container(key="card-account"):
    section_title("Account")
    c1, c2 = st.columns(2)
    with c1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    with c2:
        monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0, step=0.5, format="$%.1f")
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
    st.caption("Total charges are estimated as tenure × monthly charges.")

with st.container(key="card-services"):
    section_title("Services")
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    if phone == "Yes":
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
    else:
        multiple_lines = "No phone service"

    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    addon_names = [
        "Online Security", "Online Backup", "Device Protection",
        "Tech Support", "Streaming TV", "Streaming Movies",
    ]
    addons = {}
    if internet == "No":
        for name in addon_names:
            addons[name] = "No internet service"
    else:
        cols = st.columns(3)
        for i, name in enumerate(addon_names):
            with cols[i % 3]:
                addons[name] = st.selectbox(name, ["No", "Yes"])

if st.button("Predict", type="primary"):
    customer = {
        "Gender": gender,
        "Senior Citizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure Months": tenure,
        "Phone Service": phone,
        "Multiple Lines": multiple_lines,
        "Internet Service": internet,
        **addons,
        "Contract": contract,
        "Paperless Billing": paperless,
        "Payment Method": payment,
        "Monthly Charges": monthly,
        "Total Charges": tenure * monthly,
    }

    result = predict_customer(customer)
    render_result(result["churn_probability"], result["will_churn"])

st.caption("This tool estimates risk from historical patterns. It is a prediction, not a guarantee.")