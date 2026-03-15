import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from pathlib import Path

st.set_page_config(
    page_title="CreditScore · Risk Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════
STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Barlow:wght@300;400;500;600;700&family=Barlow+Condensed:wght@400;600;700&display=swap');

:root {
  --bg-base:       #0d0f14;
  --bg-surface:    #13161e;
  --bg-elevated:   #1a1e28;
  --bg-input:      #1e2230;
  --border-sub:    #252a38;
  --border-main:   #2e3448;
  --gold:          #c9a84c;
  --gold-light:    #e2c47a;
  --gold-dim:      rgba(201,168,76,0.18);
  --gold-glow:     rgba(201,168,76,0.08);
  --text-primary:  #f0ede6;
  --text-secondary:#8e95a8;
  --text-muted:    #5a6070;
  --red:           #e05c5c;
  --amber:         #d4943a;
  --green:         #4caf82;
  --radius-sm:     8px;
  --radius-md:     14px;
  --radius-lg:     20px;
}

html, body,
.stApp,
.stApp > div,
section[data-testid="stAppViewContainer"],
.main,
.main .block-container {
  background: var(--bg-base) !important;
  color: var(--text-primary) !important;
  font-family: 'Barlow', sans-serif !important;
}

.main .block-container {
  padding-top: 2rem !important;
  max-width: 1120px !important;
}

[data-testid="stSidebar"] { display: none !important; }

p, span, li, td, th, div, label {
  font-family: 'Barlow', sans-serif !important;
}

/* HERO */
.cv-hero {
  position: relative;
  padding: 52px 56px 48px;
  margin-bottom: 20px;
  background: var(--bg-surface);
  border: 1px solid var(--border-main);
  border-radius: var(--radius-lg);
  overflow: hidden;
}
.cv-hero::before {
  content: '';
  position: absolute; inset: 0;
  background:
    radial-gradient(ellipse 60% 80% at 90% 20%, rgba(201,168,76,0.10) 0%, transparent 60%),
    radial-gradient(ellipse 40% 60% at 10% 80%, rgba(201,168,76,0.04) 0%, transparent 50%);
  pointer-events: none;
}
.cv-hero::after {
  content: '';
  position: absolute;
  top: 0; left: 56px; right: 56px;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--gold), transparent);
  opacity: 0.6;
}
.cv-eyebrow {
  font-family: 'Barlow Condensed', sans-serif !important;
  font-size: 0.70rem; font-weight: 700;
  letter-spacing: 0.22em; text-transform: uppercase;
  color: var(--gold) !important;
  margin: 0 0 18px 0;
  display: flex; align-items: center; gap: 10px;
}
.cv-eyebrow::before {
  content: '';
  display: inline-block;
  width: 24px; height: 1px;
  background: var(--gold);
  opacity: 0.7;
}
.cv-hero-title {
  font-family: 'Playfair Display', serif !important;
  font-size: 3.2rem; font-weight: 700;
  color: var(--text-primary) !important;
  margin: 0 0 6px 0; line-height: 1.05;
  letter-spacing: -0.02em;
}
.cv-hero-title span { color: var(--gold) !important; font-style: italic; }
.cv-hero-sub {
  font-size: 1rem; font-weight: 300;
  color: var(--text-secondary) !important;
  max-width: 620px; line-height: 1.7;
  margin: 16px 0 0 0;
}
.cv-hero-decor {
  position: absolute; right: 56px; top: 50%;
  transform: translateY(-50%);
  font-size: 7rem; line-height: 1;
  color: var(--gold) !important;
  opacity: 0.06;
  font-family: 'Playfair Display', serif !important;
  font-weight: 700;
  pointer-events: none;
  user-select: none;
}

/* INFO NOTE */
.cv-note {
  margin: 0 0 28px 0;
  padding: 12px 16px;
  background: rgba(201,168,76,0.08);
  border: 1px solid rgba(201,168,76,0.18);
  border-radius: var(--radius-sm);
  color: var(--text-secondary) !important;
  font-size: 0.82rem;
}

/* SECTION */
.cv-section {
  background: var(--bg-surface);
  border: 1px solid var(--border-main);
  border-radius: var(--radius-lg);
  padding: 26px 26px 20px;
  margin-bottom: 18px;
}
.sec-header {
  display: flex; align-items: flex-start; gap: 14px;
  margin-bottom: 18px; padding-bottom: 14px;
  border-bottom: 1px solid var(--border-sub);
}
.sec-num {
  font-family: 'Barlow Condensed', sans-serif !important;
  font-size: 0.68rem; font-weight: 700;
  letter-spacing: 0.15em; text-transform: uppercase;
  color: var(--gold) !important;
  background: var(--gold-dim);
  border: 1px solid rgba(201,168,76,0.3);
  border-radius: 4px;
  padding: 4px 8px;
  margin-top: 3px;
  flex-shrink: 0;
}
.sec-title {
  font-family: 'Playfair Display', serif !important;
  font-size: 1.18rem; font-weight: 600;
  color: var(--text-primary) !important;
  margin: 0 0 4px 0;
}
.sec-desc {
  font-size: 0.84rem;
  color: var(--text-muted) !important;
  margin: 0;
}

/* FORM */
.stNumberInput label,
.stSelectbox label,
.stMultiSelect label,
.stTextInput label {
  font-family: 'Barlow Condensed', sans-serif !important;
  font-size: 0.72rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.10em !important;
  text-transform: uppercase !important;
  color: var(--text-secondary) !important;
  margin-bottom: 6px !important;
}

div[data-testid="column"] {
  padding-bottom: 0.35rem !important;
}

.stNumberInput input {
  background: var(--bg-input) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border-main) !important;
  border-radius: var(--radius-sm) !important;
  font-size: 0.95rem !important;
  font-weight: 500 !important;
  padding: 10px 14px !important;
}
.stNumberInput input:focus {
  border-color: var(--gold) !important;
  box-shadow: 0 0 0 3px var(--gold-glow) !important;
  outline: none !important;
}
.stNumberInput button {
  background: var(--bg-elevated) !important;
  color: var(--gold) !important;
  border-color: var(--border-main) !important;
}
.stNumberInput button:hover {
  background: var(--gold-dim) !important;
}

.stSelectbox > div > div,
.stSelectbox > div > div > div,
.stMultiSelect > div > div {
  background: var(--bg-input) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border-main) !important;
  border-radius: var(--radius-sm) !important;
}

.stSelectbox [data-baseweb="select"] span,
.stMultiSelect input {
  color: var(--text-primary) !important;
}

[data-baseweb="popover"],
[data-baseweb="menu"] {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border-main) !important;
  border-radius: var(--radius-sm) !important;
}

[data-baseweb="menu"] li,
[data-baseweb="menu"] [role="option"] {
  background: var(--bg-elevated) !important;
  color: var(--text-primary) !important;
}
[data-baseweb="menu"] li:hover,
[data-baseweb="menu"] [role="option"]:hover,
[data-baseweb="menu"] [aria-selected="true"] {
  background: var(--gold-dim) !important;
  color: var(--gold-light) !important;
}

.stMultiSelect [data-baseweb="tag"] {
  background: var(--gold-dim) !important;
  border: 1px solid rgba(201,168,76,0.35) !important;
  border-radius: 5px !important;
}
.stMultiSelect [data-baseweb="tag"] span,
.stMultiSelect [data-baseweb="tag"] button {
  color: var(--gold-light) !important;
}

.cv-hint {
  font-size: 0.76rem;
  color: var(--text-muted) !important;
  margin: 4px 0 4px 2px;
  font-style: italic;
}

/* BUTTON */
.stFormSubmitButton > button {
  background: linear-gradient(135deg, #b8922a 0%, var(--gold) 50%, #d4a84c 100%) !important;
  color: #0d0f14 !important;
  font-family: 'Barlow Condensed', sans-serif !important;
  font-weight: 700 !important;
  font-size: 1rem !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  padding: 15px 40px !important;
  border-radius: var(--radius-sm) !important;
  border: none !important;
  width: 100% !important;
  box-shadow: 0 4px 24px rgba(201,168,76,0.25) !important;
}
.stFormSubmitButton > button:hover {
  box-shadow: 0 8px 32px rgba(201,168,76,0.40) !important;
  transform: translateY(-1px) !important;
}

/* RESULT */
.cv-result {
  padding: 34px 40px;
  border-radius: var(--radius-lg);
  margin: 30px 0 20px;
  position: relative; overflow: hidden;
}
.cv-result::after {
  content: '';
  position: absolute; top: 0; left: 40px; right: 40px; height: 1px;
}
.cv-result-bad {
  background: linear-gradient(135deg, #1f1015 0%, #1a1014 100%);
  border: 1px solid rgba(224,92,92,0.35);
}
.cv-result-bad::after { background: linear-gradient(90deg,transparent,#e05c5c,transparent); }

.cv-result-standard {
  background: linear-gradient(135deg, #1a1508 0%, #161208 100%);
  border: 1px solid rgba(212,148,58,0.35);
}
.cv-result-standard::after { background: linear-gradient(90deg,transparent,#d4943a,transparent); }

.cv-result-good {
  background: linear-gradient(135deg, #0b1712 0%, #091410 100%);
  border: 1px solid rgba(76,175,130,0.35);
}
.cv-result-good::after { background: linear-gradient(90deg,transparent,#4caf82,transparent); }

.cv-result-eyebrow {
  font-family: 'Barlow Condensed', sans-serif !important;
  font-size: 0.68rem; font-weight: 700;
  letter-spacing: 0.20em; text-transform: uppercase;
  margin: 0 0 12px 0;
}
.cv-result-verdict {
  font-family: 'Playfair Display', serif !important;
  font-size: 2.5rem; font-weight: 700;
  margin: 0 0 8px 0; line-height: 1.1;
}
.cv-result-conf {
  font-size: 0.9rem;
  color: var(--text-secondary) !important;
  margin: 0;
}
.cv-result-conf strong {
  color: var(--text-primary) !important;
}

.cv-probs-title {
  font-family: 'Barlow Condensed', sans-serif !important;
  font-size: 0.72rem; font-weight: 700;
  letter-spacing: 0.18em; text-transform: uppercase;
  color: var(--text-muted) !important;
  margin: 28px 0 16px;
}
.cv-prob-row {
  display: flex; align-items: center; gap: 16px;
  margin-bottom: 13px;
}
.cv-prob-name {
  font-family: 'Barlow Condensed', sans-serif !important;
  font-size: 0.80rem; font-weight: 600;
  letter-spacing: 0.04em; text-transform: uppercase;
  color: var(--text-secondary) !important;
  width: 190px; flex-shrink: 0;
}
.cv-prob-track {
  flex: 1; height: 6px;
  background: var(--border-sub);
  border-radius: 100px; overflow: hidden;
}
.cv-prob-fill { height: 100%; border-radius: 100px; }
.cv-prob-pct {
  font-family: 'Barlow Condensed', sans-serif !important;
  font-size: 0.85rem; font-weight: 700;
  color: var(--text-primary) !important;
  width: 48px; text-align: right;
}

[data-testid="metric-container"] {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-main) !important;
  border-radius: var(--radius-md) !important;
  padding: 18px 20px !important;
}

[data-testid="stExpander"] {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border-main) !important;
  border-radius: var(--radius-md) !important;
}

[data-testid="stExpander"] summary {
  font-family: 'Barlow Condensed', sans-serif !important;
  font-size: 0.82rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  color: var(--gold) !important;
}

[data-testid="stDataFrame"] {
  border: 1px solid var(--border-main) !important;
  border-radius: var(--radius-sm) !important;
}

[data-testid="stAlert"] {
  background: rgba(224,92,92,0.10) !important;
  border: 1px solid rgba(224,92,92,0.3) !important;
  border-radius: var(--radius-sm) !important;
  color: #f4a0a0 !important;
}

.cv-footer {
  margin-top: 52px;
  padding: 24px 0 14px;
  border-top: 1px solid var(--border-sub);
  display: flex; justify-content: space-between; align-items: center;
  flex-wrap: wrap; gap: 12px;
}
.cv-footer-brand {
  font-family: 'Playfair Display', serif !important;
  font-size: 0.95rem; font-weight: 600;
  color: var(--gold) !important;
}
.cv-footer-note {
  font-size: 0.74rem;
  color: var(--text-muted) !important;
}
.cv-footer-note code {
  color: var(--gold) !important;
  background: var(--gold-dim);
  padding: 1px 6px;
  border-radius: 4px;
}

/* responsive */
@media (max-width: 900px) {
  .cv-hero { padding: 36px 28px; }
  .cv-hero-decor { display: none; }
  .cv-hero-title { font-size: 2.4rem; }
}
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════
MODEL_PATH = "modelo_riesgo_credito.keras"
SCALER_PATH = "minmax_scaler.joblib"
PCA_PATH = "pca_8_componentes.joblib"
ENCODERS_PATH = "label_encoders.joblib"

TARGET_META = {
    0: dict(
        label="Bad",
        sublabel="Alto Riesgo",
        css="cv-result-bad",
        color="#e05c5c",
        bar="#e05c5c",
    ),
    1: dict(
        label="Standard",
        sublabel="Riesgo Medio",
        css="cv-result-standard",
        color="#d4943a",
        bar="#d4943a",
    ),
    2: dict(
        label="Good",
        sublabel="Bajo Riesgo",
        css="cv-result-good",
        color="#4caf82",
        bar="#4caf82",
    ),
}

FEATURE_COLUMNS = [
    "Num_Cuentas_Bancarias",
    "Num_Tarjetas_Credito",
    "Tasa_Interes",
    "Num_Prestamos",
    "Retraso_Desde_Vencimiento",
    "Num_Pagos_Retrasados",
    "Cambio_Limite_Credito",
    "Num_Consultas_Credito",
    "Mezcla_Credito",
    "Deuda_Pendiente",
    "Ratio_Utilizacion_Credito",
    "Antiguedad_Historial_Crediticio",
    "Pago_Monto_Minimo",
    "Cuota_Mensual_Total",
    "Balance_Mensual",
]

DEFAULTS = {
    "Num_Cuentas_Bancarias": 5.0,
    "Num_Tarjetas_Credito": 5.0,
    "Tasa_Interes": 13.0,
    "Num_Prestamos": 3.0,
    "Retraso_Desde_Vencimiento": 17.875,
    "Num_Pagos_Retrasados": 13.75,
    "Cambio_Limite_Credito": 9.37,
    "Num_Consultas_Credito": 5.25,
    "Deuda_Pendiente": 1166.155,
    "Ratio_Utilizacion_Credito": 45.394818,
    "Antiguedad_Historial_Crediticio": 18.238095,
    "Cuota_Mensual_Total": 53.506477,
    "Balance_Mensual": 338.49158,
}


# ══════════════════════════════════════════════════════════════════
#  CARGA
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    return model, scaler, pca, encoders


def section_open(num, title, desc):
    st.markdown(
        f"""
        <div class="cv-section">
          <div class="sec-header">
            <span class="sec-num">{num}</span>
            <div>
              <p class="sec-title">{title}</p>
              <p class="sec-desc">{desc}</p>
            </div>
          </div>
        """,
        unsafe_allow_html=True,
    )


def section_close():
    st.markdown("</div>", unsafe_allow_html=True)


def build_feature_row(inputs: dict) -> pd.DataFrame:
    row = pd.DataFrame([inputs], columns=FEATURE_COLUMNS)
    return row


def encode_inputs(row: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    row = row.copy()
    row["Mezcla_Credito"] = encoders["Credit_Mix"].transform(
        row["Mezcla_Credito"].astype(str)
    )
    row["Pago_Monto_Minimo"] = encoders["Payment_of_Min_Amount"].transform(
        row["Pago_Monto_Minimo"].astype(str)
    )
    return row


# ══════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════
st.markdown(
    """
<div class="cv-hero">
  <p class="cv-eyebrow">Risk Intelligence Platform</p>
  <h1 class="cv-hero-title">Credit<span>Score</span></h1>
  <p class="cv-hero-sub">
    Sistema de evaluación crediticia basado en una Red Neuronal Artificial con reducción
    de dimensionalidad PCA. Ingresa el perfil financiero del cliente para obtener su
    clasificación de riesgo en tiempo real.
  </p>
  <div class="cv-hero-decor">◆</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="cv-note">
      Los campos numéricos están restringidos a los rangos observados durante el entrenamiento del modelo,
      con el fin de mantener la coherencia de las predicciones.
    </div>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════
#  VALIDAR ARCHIVOS
# ══════════════════════════════════════════════════════════════════
required_files = [MODEL_PATH, SCALER_PATH, PCA_PATH, ENCODERS_PATH]
missing = [p for p in required_files if not Path(p).exists()]
if missing:
    st.error(f"Faltan artefactos del modelo: {', '.join(missing)}")
    st.stop()

model, scaler, pca, encoders = load_artifacts()

credit_mix_options = list(encoders["Credit_Mix"].classes_)
payment_min_options = list(encoders["Payment_of_Min_Amount"].classes_)

# ══════════════════════════════════════════════════════════════════
#  FORMULARIO
# ══════════════════════════════════════════════════════════════════
with st.form("cv_form"):

    section_open(
        "01",
        "Productos financieros",
        "Cantidad de cuentas, tarjetas, préstamos y consultas crediticias.",
    )
    c1, c2 = st.columns(2, gap="large")
    with c1:
        num_cuentas_bancarias = st.number_input(
            "Nº cuentas bancarias — Num_Cuentas_Bancarias",
            min_value=0.0,
            max_value=10.5,
            value=float(DEFAULTS["Num_Cuentas_Bancarias"]),
            step=0.5,
        )
        num_tarjetas_credito = st.number_input(
            "Nº tarjetas de crédito — Num_Tarjetas_Credito",
            min_value=0.5,
            max_value=10.875,
            value=float(DEFAULTS["Num_Tarjetas_Credito"]),
            step=0.5,
        )
        num_prestamos = st.number_input(
            "Nº de préstamos — Num_Prestamos",
            min_value=0.0,
            max_value=9.0,
            value=float(DEFAULTS["Num_Prestamos"]),
            step=1.0,
        )
    with c2:
        num_consultas_credito = st.number_input(
            "Consultas al buró — Num_Consultas_Credito",
            min_value=0.0,
            max_value=16.375,
            value=float(DEFAULTS["Num_Consultas_Credito"]),
            step=1.0,
        )
        mezcla_credito = st.selectbox(
            "Mezcla de crédito — Mezcla_Credito",
            credit_mix_options,
            index=(
                credit_mix_options.index("Standard")
                if "Standard" in credit_mix_options
                else 0
            ),
        )
        pago_monto_minimo = st.selectbox(
            "Pago del monto mínimo — Pago_Monto_Minimo",
            payment_min_options,
            index=(
                payment_min_options.index("Yes") if "Yes" in payment_min_options else 0
            ),
        )
    section_close()

    section_open(
        "02",
        "Costo del crédito y endeudamiento",
        "Tasa de interés, deuda pendiente y cuota mensual total.",
    )
    c1, c2 = st.columns(2, gap="large")
    with c1:
        tasa_interes = st.number_input(
            "Tasa de interés % — Tasa_Interes",
            min_value=1.0,
            max_value=34.0,
            value=float(DEFAULTS["Tasa_Interes"]),
            step=0.5,
        )
        deuda_pendiente = st.number_input(
            "Deuda pendiente $ — Deuda_Pendiente",
            min_value=0.23,
            max_value=4998.07,
            value=float(DEFAULTS["Deuda_Pendiente"]),
            step=10.0,
        )
    with c2:
        cuota_mensual_total = st.number_input(
            "Cuota mensual total $ — Cuota_Mensual_Total",
            min_value=2.04613,
            max_value=130.33341,
            value=float(DEFAULTS["Cuota_Mensual_Total"]),
            step=0.1,
            format="%.4f",
        )
        cambio_limite_credito = st.number_input(
            "Cambio en límite de crédito — Cambio_Limite_Credito",
            min_value=0.5,
            max_value=31.115,
            value=float(DEFAULTS["Cambio_Limite_Credito"]),
            step=0.1,
            format="%.3f",
        )
    section_close()

    section_open(
        "03",
        "Historial y comportamiento crediticio",
        "Uso de crédito, retrasos y antigüedad del historial.",
    )
    c1, c2 = st.columns(2, gap="large")
    with c1:
        retraso_desde_vencimiento = st.number_input(
            "Retraso desde vencimiento (días) — Retraso_Desde_Vencimiento",
            min_value=-2.0,
            max_value=63.25,
            value=float(DEFAULTS["Retraso_Desde_Vencimiento"]),
            step=1.0,
        )
        num_pagos_retrasados = st.number_input(
            "Nº pagos retrasados — Num_Pagos_Retrasados",
            min_value=0.0,
            max_value=26.375,
            value=float(DEFAULTS["Num_Pagos_Retrasados"]),
            step=1.0,
        )
        ratio_utilizacion_credito = st.number_input(
            "Utilización de crédito % — Ratio_Utilizacion_Credito",
            min_value=3.15708,
            max_value=125.23911,
            value=float(DEFAULTS["Ratio_Utilizacion_Credito"]),
            step=0.1,
            format="%.5f",
        )
    with c2:
        antiguedad_historial_crediticio = st.number_input(
            "Antigüedad historial crediticio — Antiguedad_Historial_Crediticio",
            min_value=0.375,
            max_value=33.380952,
            value=float(DEFAULTS["Antiguedad_Historial_Crediticio"]),
            step=0.1,
            format="%.6f",
        )
        balance_mensual = st.number_input(
            "Balance mensual $ — Balance_Mensual",
            min_value=92.841401,
            max_value=1349.264887,
            value=float(DEFAULTS["Balance_Mensual"]),
            step=1.0,
            format="%.6f",
        )
    section_close()

    submitted = st.form_submit_button("◆  Ejecutar análisis crediticio")

# ══════════════════════════════════════════════════════════════════
#  RESULTADO
# ══════════════════════════════════════════════════════════════════
if submitted:
    inputs = {
        "Num_Cuentas_Bancarias": float(num_cuentas_bancarias),
        "Num_Tarjetas_Credito": float(num_tarjetas_credito),
        "Tasa_Interes": float(tasa_interes),
        "Num_Prestamos": float(num_prestamos),
        "Retraso_Desde_Vencimiento": float(retraso_desde_vencimiento),
        "Num_Pagos_Retrasados": float(num_pagos_retrasados),
        "Cambio_Limite_Credito": float(cambio_limite_credito),
        "Num_Consultas_Credito": float(num_consultas_credito),
        "Mezcla_Credito": mezcla_credito,
        "Deuda_Pendiente": float(deuda_pendiente),
        "Ratio_Utilizacion_Credito": float(ratio_utilizacion_credito),
        "Antiguedad_Historial_Crediticio": float(antiguedad_historial_crediticio),
        "Pago_Monto_Minimo": pago_monto_minimo,
        "Cuota_Mensual_Total": float(cuota_mensual_total),
        "Balance_Mensual": float(balance_mensual),
    }

    row = build_feature_row(inputs)
    row_encoded = encode_inputs(row, encoders)
    row_scaled = scaler.transform(row_encoded)
    row_pca = pca.transform(row_scaled)

    probs = model.predict(row_pca, verbose=0)[0]
    cls = int(np.argmax(probs))
    m = TARGET_META[cls]

    st.markdown(
        f"""
        <div class="cv-result {m['css']}">
          <p class="cv-result-eyebrow" style="color:{m['color']};">◆ Clasificación de riesgo crediticio</p>
          <p class="cv-result-verdict" style="color:{m['color']};">{m['label']}</p>
          <p class="cv-result-conf">
            Categoría: <strong>{m['sublabel']}</strong>
            &nbsp;·&nbsp;
            Confianza del modelo: <strong>{probs[cls]*100:.1f}%</strong>
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    for i, col in enumerate([c1, c2, c3]):
        m2 = TARGET_META[i]
        col.metric(f"{m2['label']} · {m2['sublabel']}", f"{probs[i]*100:.1f}%")

    st.markdown(
        '<p class="cv-probs-title">Distribución de probabilidad por categoría</p>',
        unsafe_allow_html=True,
    )
    for i in range(3):
        m2 = TARGET_META[i]
        pct = probs[i] * 100
        st.markdown(
            f"""
            <div class="cv-prob-row">
              <span class="cv-prob-name">{m2['label']} — {m2['sublabel']}</span>
              <div class="cv-prob-track">
                <div class="cv-prob-fill" style="width:{pct:.1f}%;background:{m2['bar']};"></div>
              </div>
              <span class="cv-prob-pct">{pct:.1f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Datos técnicos del análisis"):
        st.write("**Parámetros de entrada**")
        st.json(inputs)
        st.write("**Vector codificado y listo para el scaler**")
        st.dataframe(row_encoded, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown(
    """
<div class="cv-footer">
  <span class="cv-footer-brand">◆ CreditScore</span>
  <span class="cv-footer-note">
    Powered by ANN + PCA · Requiere
    <code>modelo_riesgo_credito.keras</code>
    <code>minmax_scaler.joblib</code>
    <code>pca_8_componentes.joblib</code>
    <code>label_encoders.joblib</code>
  </span>
</div>
""",
    unsafe_allow_html=True,
)
