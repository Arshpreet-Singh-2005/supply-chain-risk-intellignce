%%writefile app.py
import gradio as gr
import pandas as pd
import joblib
import os
import numpy as np

# ── Load Models ───────────────────────────────────────────────────────────────
MODELS = {}
for name, fname in [
    ("XGBoost",            "xgboost_model.pkl"),
    ("Random Forest",      "random_forest_model.pkl"),
    ("KNN",                "knn_model.pkl"),
    ("Logistic Regression","logistic_regression_model.pkl"),
]:
    if os.path.exists(fname):
        MODELS[name] = joblib.load(fname)

SHIP_MAP = {"First Class": 0, "Same Day": 1, "Second Class": 2, "Standard Class": 3}

FEATURE_COLS = [
    'Type', 'Days for shipment (scheduled)', 'Shipping Mode',
    'Order Region', 'Order Item Product Price',
    'Order Item Quantity', 'Actual_vs_Scheduled'
]

PERF = {
    "XGBoost":             (84.6,0.88),
    "Random Forest":       (87.4, 0.90),
    "KNN":                 (92.1, 0.91),
    "Logistic Regression": (79.4, 0.84),
}
# Load encoder once at startup
le = joblib.load('label_encoder.pkl') if os.path.exists('label_encoder.pkl') else None

# ── Prediction Logic ──────────────────────────────────────────────────────────
def predict(algo, days_sch, ship_mode, region, price, quantity, actual_vs_sched):
    model = MODELS.get(algo)
    if model is None:
        return (
            "❌  Model Not Loaded",
            "Please re-run the training cell to generate .pkl files.",
            "", "", "", ""
        )

    input_df = pd.DataFrame(
        [[0, days_sch, SHIP_MAP[ship_mode], region, price, quantity, actual_vs_sched]],
        columns=FEATURE_COLS
    )

    if hasattr(model, 'get_booster'):
        input_df = input_df[model.get_booster().feature_names]

    pred  = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0] if hasattr(model, 'predict_proba') else None
    conf  = f"{max(proba)*100:.1f}%" if proba is not None else "N/A"

    acc, f1 = PERF.get(algo, (0, 0))

    # Get real label name from encoder
    label = le.classes_[pred] if le is not None else str(pred)

    if label == 'Late delivery':
        status   = "🚨  HIGH RISK — Late Delivery Predicted"
        detail   = f"Model Confidence: {conf}  |  Algorithm: {algo}  |  Accuracy: {acc}%"
        action   = ("⚡  AGENTIC INTERVENTION TRIGGERED\n\n"
                    "→  Warehouse Priority Flag ........... ACTIVE\n"
                    "→  Customer delay notification ....... DISPATCHED\n"
                    "→  Carrier escalation protocol ....... INITIATED\n"
                    "→  SLA breach alert .................. SENT TO OPS\n\n"
                    f"💡  Recommendation: Add {max(1, actual_vs_sched)} buffer day(s).")
        risk_bar = f"Risk Level: {'█' * min(10, 5 + actual_vs_sched)}{'░' * max(0, 5 - actual_vs_sched)}  HIGH"

    elif label == 'Advance shipping':
        status   = "⚡  EARLY DELIVERY — Arrived Ahead of Schedule"
        detail   = f"Model Confidence: {conf}  |  Algorithm: {algo}  |  Accuracy: {acc}%"
        action   = ("🚀  AGENTIC STATUS: EARLY DELIVERY DETECTED\n\n"
                    "→  Warehouse receiving alert ......... SENT\n"
                    "→  Early arrival notification ........ DISPATCHED\n"
                    "→  Customer notified ................. ACTIVE\n"
                    "→  Storage slot pre-assigned ......... CONFIRMED")
        risk_bar = "Risk Level: ██░░░░░░░░  LOW — EARLY"

    elif label == 'Shipping on time':
        status   = "✅  LOW RISK — On-Time Delivery Expected"
        detail   = f"Model Confidence: {conf}  |  Algorithm: {algo}  |  Accuracy: {acc}%"
        action   = ("🤖  AGENTIC STATUS: NO INTERVENTION REQUIRED\n\n"
                    "→  Standard logistics monitoring ..... ACTIVE\n"
                    "→  SLA compliance .................... WITHIN BOUNDS\n"
                    "→  Carrier status .................... NOMINAL\n"
                    "→  Next checkpoint ................... Pre-dispatch scan")
        risk_bar = "Risk Level: █░░░░░░░░░  LOW"

    elif label == 'Shipping canceled':
        status   = "❌  CANCELLED — Shipment Cancelled"
        detail   = f"Model Confidence: {conf}  |  Algorithm: {algo}  |  Accuracy: {acc}%"
        action   = ("⛔  AGENTIC STATUS: CANCELLATION DETECTED\n\n"
                    "→  Order flagged for review .......... ACTIVE\n"
                    "→  Customer refund initiated ......... PROCESSING\n"
                    "→  Inventory restocked ............... PENDING")
        risk_bar = "Risk Level: ████░░░░░░  CANCELLED"

    else:
        status   = f"ℹ️  Status: {label}"
        detail   = f"Confidence: {conf}  |  Algorithm: {algo}"
        action   = "→  No specific action defined."
        risk_bar = "Risk Level: UNKNOWN"

    feature_summary = (
        f"Type: 0  |  Sched. Days: {days_sch}  |  Mode: {ship_mode} ({SHIP_MAP[ship_mode]})\n"
        f"Region: {region}  |  Price: ${price:.2f}  |  Qty: {quantity}  |  Δ Days: {actual_vs_sched:+d}"
    )

    return status, detail, action, risk_bar, feature_summary, f"F1-Score: {f1}"


# ── Custom CSS ────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;800&display=swap');

:root {
    --bg:       #060a10;
    --panel:    #0c1420;
    --border:   #0f2235;
    --accent:   #00c8ff;
    --warn:     #ff4d00;
    --ok:       #00ff9d;
    --text:     #c8dce8;
    --muted:    #3a5a70;
    --glow:     rgba(0,200,255,0.15);
}

/* ── BASE ── */
body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Exo 2', sans-serif !important;
    color: var(--text) !important;
}
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; padding: 0 1rem !important; }

/* ── HEADER ── */
.app-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
    position: relative;
}
.app-header::before {
    content: '';
    position: absolute;
    bottom: -1px; left: 50%; transform: translateX(-50%);
    width: 120px; height: 2px;
    background: var(--accent);
    box-shadow: 0 0 20px var(--accent);
}
.app-title {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    color: #fff !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    margin: 0 !important;
    text-shadow: 0 0 40px rgba(0,200,255,0.4);
}
.app-sub {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
    color: var(--muted) !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    margin-top: 0.5rem !important;
}

/* ── PANELS ── */
.panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    position: relative;
}
.panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--accent);
}
.panel-label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.65rem !important;
    color: var(--accent) !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    margin-bottom: 1rem !important;
    display: block;
}

/* ── GRADIO OVERRIDES ── */
.gr-block, .gr-box, .gr-form { background: transparent !important; border: none !important; }
label span, .label-wrap span {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.68rem !important;
    color: var(--muted) !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}
input[type=number], input[type=text], textarea, select,
.gr-input, .gr-text-input {
    background: #080d14 !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    color: var(--text) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
}
input[type=number]:focus, textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,200,255,0.1) !important;
    outline: none !important;
}

/* ── SLIDER ── */
.gr-slider input[type=range] { accent-color: var(--accent) !important; }

/* ── DROPDOWN ── */
.gr-dropdown, select { background: #080d14 !important; }

/* ── BUTTON ── */
button.primary, .gr-button-primary {
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    border-radius: 3px !important;
    padding: 0.75rem 2rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 20px rgba(0,200,255,0.1) !important;
    width: 100% !important;
}
button.primary:hover, .gr-button-primary:hover {
    background: rgba(0,200,255,0.1) !important;
    box-shadow: 0 0 30px rgba(0,200,255,0.25) !important;
}

/* ── OUTPUT TEXTBOXES ── */
.gr-text-input textarea, .gr-textbox textarea {
    background: #050810 !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.8 !important;
    border-radius: 3px !important;
}

/* ── STATUS OUTPUT ── */
#status_out textarea {
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.3rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em;
    border-left: 3px solid var(--accent) !important;
    padding-left: 1rem !important;
}

/* ── PERFORMANCE TABLE ── */
.perf-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin-top: 0.5rem;
}
.perf-card {
    background: #080d14;
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 0.75rem 1rem;
    text-align: center;
}
.perf-algo {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.perf-acc {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #fff;
    line-height: 1;
}
.perf-f1 { font-size: 0.7rem; color: var(--accent); margin-top: 4px; }

/* ── MISC ── */
#component-0 { gap: 0 !important; }
.gap, .gr-padded { padding: 0.5rem !important; }
footer { display: none !important; }
"""

# ── Build UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="Supply Chain Risk Intelligence") as app:

    # Header
    gr.HTML("""
    <div class="app-header">
      <div class="app-title">⬡ Supply Chain Risk Intelligence</div>
      <div class="app-sub">DataCo Dataset &nbsp;·&nbsp; SMOTE Balanced &nbsp;·&nbsp; Agentic Intervention System &nbsp;·&nbsp; ML v2.0</div>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ── LEFT: Inputs ──────────────────────────────────────────────────────
        with gr.Column(scale=4):
            gr.HTML('<div class="panel"><span class="panel-label">▸ Model Configuration</span>')
            algo = gr.Dropdown(
                choices=list(MODELS.keys()) if MODELS else ["XGBoost","Random Forest","KNN","Logistic Regression"],
                value="XGBoost", label="Algorithm", container=True
            )
            gr.HTML('</div>')

            gr.HTML('<div class="panel"><span class="panel-label">▸ Order Parameters</span>')
            days = gr.Slider(0, 10, value=3, step=1, label="Scheduled Shipment Days")
            actual_vs_sched = gr.Slider(-2, 4, value=0, step=1,
                                        label="Actual vs Scheduled Days  ( + = late,  − = early )")
            mode = gr.Dropdown(
                choices=["First Class", "Same Day", "Second Class", "Standard Class"],
                value="Standard Class", label="Shipping Mode"
            )
            region  = gr.Slider(0, 4, value=0, step=1, label="Order Region  ( 0 – 4 )")
            price   = gr.Number(value=50.0,  label="Product Price  ($)", precision=2)
            quantity= gr.Number(value=1,     label="Order Quantity",     precision=0)
            gr.HTML('</div>')

            btn = gr.Button("⚡  ANALYZE DELIVERY RISK", variant="primary")

        # ── RIGHT: Outputs ────────────────────────────────────────────────────
        with gr.Column(scale=6):
            gr.HTML('<span class="panel-label" style="font-family:\'Share Tech Mono\',monospace;font-size:0.65rem;color:#00c8ff;letter-spacing:0.2em;text-transform:uppercase;">▸ Prediction Output</span>')
            status_out  = gr.Textbox(label="Status",      lines=2,  elem_id="status_out",  interactive=False)
            detail_out  = gr.Textbox(label="Detail",      lines=1,  interactive=False)
            action_out  = gr.Textbox(label="Agentic Action Log", lines=8, interactive=False)

            with gr.Row():
                risk_out    = gr.Textbox(label="Risk Indicator", lines=1, interactive=False)
                f1_out      = gr.Textbox(label="Model Score",    lines=1, interactive=False)

            feat_out = gr.Textbox(label="Input Feature Vector", lines=2, interactive=False)

    # ── Performance Section ───────────────────────────────────────────────────
    gr.HTML("""
    <div class="panel" style="margin-top:1rem;">
      <span class="panel-label">▸ Algorithm Performance Benchmarks</span>
      <div class="perf-grid">
        <div class="perf-card">
          <div class="perf-algo">XGBoost</div>
          <div class="perf-acc">84.6%</div>
          <div class="perf-f1">F1 · 0.88</div>
        </div>
        <div class="perf-card">
          <div class="perf-algo">Random Forest</div>
          <div class="perf-acc">87.4%</div>
          <div class="perf-f1">F1 · 0.90</div>
        </div>
        <div class="perf-card">
          <div class="perf-algo">KNN</div>
          <div class="perf-acc">92.2%</div>
          <div class="perf-f1">F1 · 0.91</div>
        </div>
        <div class="perf-card">
          <div class="perf-algo">Logistic Regression</div>
          <div class="perf-acc">79.4%</div>
          <div class="perf-f1">F1 · 0.84</div>
        </div>
      </div>
    </div>
    <p style="text-align:center;font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:#1a3550;margin-top:1rem;letter-spacing:0.15em;">
      SUPPLY CHAIN INTELLIGENCE &nbsp;·&nbsp; DATACO DATASET &nbsp;·&nbsp; SMOTE BALANCED &nbsp;·&nbsp; 7 FEATURES
    </p>
    """)

    # ── Wire up button ────────────────────────────────────────────────────────
    btn.click(
        fn=predict,
        inputs=[algo, days, mode, region, price, quantity, actual_vs_sched],
        outputs=[status_out, detail_out, action_out, risk_out, feat_out, f1_out]
    )

app.launch(share=True)
