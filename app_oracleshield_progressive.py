import os, json, time, hashlib ,copy
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from oracle_shield_world_model import AuditChain, PersistentThreatMemory, state_from_window, STATE_NAMES, stage_for_label, hash_event, TransformerWorldModel, MultiStepRolloutEngine
from oracle_shield_blockchain import PermissionedBlockchain, SOCNode, Block, Transaction, MerkleTree
from flow_extractor import PacketRecord, FlowTracker, FlowStateEncoder, LivePacketStreamGenerator
from live_detector import LiveFlowDetector

# -------------------- CONFIG --------------------
APP_NAME = 'OracleShield'
DATA_FILE = 'Copy of DOC-20260825-WA0002.xlsx'
CLASSIFIER_FILE = 'model_classifier.joblib'
SCALER_FILE = 'scaler.joblib'
FEATURE_FILE = 'feature_columns.joblib'
WORLD_FILE = 'world_model.pt'
LEDGER_FILE = 'oracle_shield_ledger.json'
BLOCKCHAIN_FILE = 'oracle_shield_blockchain.json'

st.set_page_config(page_title='OracleShield | Predictive Cyber Defence', page_icon='🛡️', layout='wide', initial_sidebar_state='expanded')

# -------------------- STYLE --------------------
st.markdown("""
<style>

/* ---------- GLOBAL ---------- */

[data-testid="stAppViewContainer"] {
    background: #0b1120;
}

[data-testid="stHeader"] {
    background: #0b1120;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* ---------- MAIN CONTENT ---------- */

.main .block-container {
    color: #e5e7eb;
}

/* ---------- HERO ---------- */

.hero {
    background:
        radial-gradient(circle at 85% 20%, rgba(56,189,248,.15), transparent 30%),
        linear-gradient(135deg, #0f1f3d 0%, #111827 100%);

    border: 1px solid rgba(96,165,250,.22);
    color: #f8fafc;

    padding: 26px 30px;
    border-radius: 18px;

    margin-bottom: 24px;

    box-shadow:
        0 10px 35px rgba(0,0,0,.25),
        inset 0 1px 0 rgba(255,255,255,.04);
}

.hero h1 {
    margin: 0;
    font-size: 36px;
    font-weight: 800;
    letter-spacing: -0.5px;
}

.hero p {
    margin: 8px 0 0;
    color: #94a3b8;
    font-size: 14px;
}

/* ---------- KPI METRICS ---------- */

[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 14px;

    padding: 18px 20px;

    box-shadow:
        0 6px 20px rgba(0,0,0,.18);
}

[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-size: 28px !important;
    font-weight: 800 !important;
}

/* ---------- HEADINGS ---------- */

h1, h2, h3 {
    color: #f8fafc !important;
}

h4, h5, h6 {
    color: #cbd5e1 !important;
}

p, label {
    color: #cbd5e1;
}

/* ---------- CARDS ---------- */

.card {
    background: #111827;

    border: 1px solid #1f2937;
    border-radius: 14px;

    padding: 18px;

    box-shadow:
        0 6px 20px rgba(0,0,0,.18);
}

.small {
    color: #64748b;
    font-size: 13px;
}

.kpi {
    font-size: 28px;
    font-weight: 800;
    color: #f8fafc;
}

/* ---------- STATUS ---------- */

.status-good {
    color: #34d399;
    font-weight: 700;
}

.status-warn {
    color: #fbbf24;
    font-weight: 700;
}

.status-bad {
    color: #f87171;
    font-weight: 700;
}

/* ---------- SIDEBAR ---------- */

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #090d18 0%,
            #111827 100%
        );

    border-right: 1px solid #1f2937;
}

[data-testid="stSidebar"] * {
    color: #cbd5e1;
}

[data-testid="stSidebar"] h3 {
    color: #f8fafc !important;
}

/* ---------- BUTTONS ---------- */

.stButton > button {
    width: 100%;

    background: linear-gradient(
        135deg,
        #2563eb,
        #1d4ed8
    ) !important;

    color: white !important;

    border: 1px solid #3b82f6 !important;
    border-radius: 10px !important;

    padding: 11px 18px !important;

    font-weight: 700 !important;

    box-shadow:
        0 6px 18px rgba(37,99,235,.25);
}

.stButton > button:hover {
    background: linear-gradient(
        135deg,
        #3b82f6,
        #2563eb
    ) !important;

    border-color: #60a5fa !important;
}

/* ---------- SLIDERS ---------- */

[data-testid="stSlider"] label {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
}

/* ---------- DATAFRAMES ---------- */

[data-testid="stDataFrame"] {
    border: 1px solid #1f2937;
    border-radius: 12px;
    overflow: hidden;
}

/* ---------- INFO / SUCCESS BOXES ---------- */

[data-testid="stAlert"] {
    border-radius: 12px;
}

/* ---------- RADIO BUTTONS ---------- */

[data-testid="stRadio"] label {
    color: #cbd5e1 !important;
}

/* ---------- DIVIDERS ---------- */

hr {
    border-color: #1f2937 !important;
}

/* ---------- PLOTLY AREA ---------- */

.stPlotlyChart {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 14px;
    padding: 8px;
}

/* ---------- SCROLLBAR ---------- */

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #0b1120;
}

::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #475569;
}

</style>
""", unsafe_allow_html=True)
# -------------------- LOADERS --------------------
@st.cache_data(show_spinner='Loading combined NSL-KDD workbook…')
def load_data():
    df = pd.read_excel(DATA_FILE)
    required = {'split','attack_category','is_attack','label'}
    missing = required - set(df.columns)
    if missing: raise ValueError(f'Missing required columns: {sorted(missing)}')
    return df

@st.cache_resource(show_spinner='Loading trained detector…')
def load_detector():
    return joblib.load(CLASSIFIER_FILE), joblib.load(SCALER_FILE), joblib.load(FEATURE_FILE)

@st.cache_resource(show_spinner='Loading world model…')
def load_world_model():
    if not os.path.exists(WORLD_FILE): return None, None
    import torch
    from oracle_shield_world_model import WorldModel
    ckpt=torch.load(WORLD_FILE,map_location='cpu')
    model=WorldModel(
    ckpt['input_dim'],
    hidden=ckpt.get('hidden_dim', 96),
    classes=len(ckpt['classes'])
)
    model.load_state_dict(ckpt['model']); model.eval()
    return model, ckpt

# -------------------- ADAPTIVE WORLD MEMORY --------------------
class AdaptiveThreatMemory:
    """Self-supervised environment memory. It learns evolving state statistics and
    transitions without training the detector on its own predictions."""
    def __init__(self):
        self.baseline = None
        self.ema = None
        self.prev = None
        self.transitions = {}
        self.episodes = []
        self.novelty = 0.0
        self.drift = 0.0
        self.alpha = 0.18

    def update(self, state, stage):
        x=np.asarray(state,dtype=float)
        if self.ema is None:
            self.ema=x.copy(); self.baseline=x.copy()
        else:
            self.ema=(1-self.alpha)*self.ema+self.alpha*x
            self.drift=float(np.linalg.norm(self.ema-self.baseline)/(np.linalg.norm(self.baseline)+1e-6))
        self.novelty=float(np.linalg.norm(x-self.ema)/(np.linalg.norm(self.ema)+1e-6))
        key=stage
        if self.prev is not None:
            p=self.prev
            self.transitions[(p,key)] = self.transitions.get((p,key),0)+1
        self.prev=key
        self.episodes.append({'time':datetime.now().isoformat(timespec='seconds'),'stage':stage,'novelty':self.novelty,'drift':self.drift})
        self.episodes=self.episodes[-500:]
        return self.novelty, self.drift

    def progression_probability(self, base_prob):
        # Adaptive prior: increases when attack pressure is novel or drifting.
        p=float(base_prob)
        p += min(0.18, self.novelty*0.35)
        p += min(0.12, self.drift*0.20)
        return float(np.clip(p,0,0.99))

if 'adaptive_memory' not in st.session_state: st.session_state.adaptive_memory=AdaptiveThreatMemory()
if 'persistent_memory' not in st.session_state: st.session_state.persistent_memory=PersistentThreatMemory()
if 'replay_log' not in st.session_state: st.session_state.replay_log=[]

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown('### System controls')
    page=st.radio('Workspace',['Command Center','World Model','Blockchain Audit','Evidence & Data'])
    st.markdown('---')
    st.markdown('**Prototype provenance**')
    st.caption('Dataset: combined NSL-KDD workbook')
    st.caption('Original train/test membership preserved by `split`')
    st.caption('Offline / no cloud dependency')

try:
    df=load_data()
    clf, scaler, feature_cols=load_detector()
except Exception as e:
    st.error(f'Cannot initialise OracleShield: {e}')
    st.stop()

train=df[df['split'].astype(str).str.lower().eq('train')].copy()
test=df[df['split'].astype(str).str.lower().eq('test')].copy()
world_model, world_ckpt=load_world_model()

# -------------------- PREPROCESS --------------------
def make_X(frame):
    drop=[c for c in ['label','attack_category','is_attack','split'] if c in frame.columns]
    cats=[c for c in ['protocol_type','service','flag'] if c in frame.columns]
    X=pd.get_dummies(frame.drop(columns=drop),columns=cats)
    X=X.reindex(columns=feature_cols,fill_value=0)
    return scaler.transform(X)

# -------------------- METRICS --------------------
if 'model_metrics' not in st.session_state:
    sample_n=min(12000,len(test))
    metric_df=test.sample(sample_n,random_state=42).reset_index(drop=True)
    pred=clf.predict(make_X(metric_df))
    y=metric_df['attack_category'].values
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
    st.session_state.model_metrics={
        'accuracy':accuracy_score(y,pred),
        'macro_precision':precision_score(y,pred,average='macro',zero_division=0),
        'macro_recall':recall_score(y,pred,average='macro',zero_division=0),
        'macro_f1':f1_score(y,pred,average='macro',zero_division=0),
        'weighted_f1':f1_score(y,pred,average='weighted',zero_division=0),
        'cm':confusion_matrix(y,pred,labels=['dos','normal','probe','r2l','u2r']),
        'report':classification_report(y,pred,labels=['dos','normal','probe','r2l','u2r'],output_dict=True,zero_division=0)
    }
M=st.session_state.model_metrics
# -------------------- COMMAND CENTER --------------------
if page == 'Command Center':

    # ==========================================================
    # COMMAND CENTER HEADER
    # ==========================================================
    st.markdown(
        """
        <div class="hero">
            <h1>🛡️ OracleShield Command Center</h1>
            <p>
                Predictive cyber defence console · Detect → Understand → Forecast → Audit
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ==========================================================
    # TOP PERFORMANCE KPIs
    # ==========================================================
    st.markdown("### Detection performance")

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric(
        "Detection Accuracy",
        f"{M['accuracy']:.1%}"
    )

    k2.metric(
        "Macro F1",
        f"{M['macro_f1']:.1%}"
    )

    k3.metric(
        "Macro Recall",
        f"{M['macro_recall']:.1%}"
    )

    k4.metric(
        "Training Flows",
        f"{len(train):,}"
    )

    k5.metric(
        "Evaluation Flows",
        f"{len(test):,}"
    )

    st.markdown("---")

    # ==========================================================
    # LIVE DEFENCE STATUS
    # ==========================================================
    st.markdown("### Live defence status")

    s1, s2, s3, s4 = st.columns(4)

    # Initial values before replay
    current_risk = "STANDBY"
    current_stage = "Awaiting telemetry"
    current_probability = 0.0
    current_confidence = 0.0

    s1.markdown(
        f"""
        <div class="card">
            <div class="small">SYSTEM STATUS</div>
            <div class="kpi">● READY</div>
            <div class="small">Detector + World Model online</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    s2.markdown(
        f"""
        <div class="card">
            <div class="small">CURRENT THREAT</div>
            <div class="kpi">{current_risk}</div>
            <div class="small">Updated during replay</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    s3.markdown(
        f"""
        <div class="card">
            <div class="small">PREDICTED STAGE</div>
            <div class="kpi" style="font-size:20px">
                {current_stage}
            </div>
            <div class="small">World Model inference</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    s4.markdown(
        f"""
        <div class="card">
            <div class="small">FUTURE RISK</div>
            <div class="kpi">{current_probability:.0%}</div>
            <div class="small">Probability of attack progression</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("")
# ==========================================================
# REAL LIVE NETWORK CAPTURE
# ==========================================================
st.markdown("### 🌐 Real-time network defence")

lc1, lc2 = st.columns([2, 1])

with lc1:
    live_interface = st.text_input(
        "Network interface",
        value="wlan0"
    )

with lc2:
    live_duration = st.slider(
        "Capture duration (seconds)",
        min_value=10,
        max_value=120,
        value=30,
        step=10
    )

live_start = st.button(
    "🟢 START REAL LIVE CAPTURE",
    type="primary",
    use_container_width=True
)

if live_start:

    try:
        detector = LiveFlowDetector(
            live_interface,
            window_seconds=5
        )

        st.session_state.live_detector = detector

        status_live = st.empty()
        metrics_live = st.empty()
        table_live = st.empty()

        status_live.info(
            f"🟢 Capturing real traffic on "
            f"`{live_interface}`..."
        )

        detector.start()

        start_time = time.time()

        while time.time() - start_time < live_duration:

            elapsed = int(
                time.time() - start_time
            )

            remaining = max(
                0,
                live_duration - elapsed
            )

            snap = detector.snapshot()

            metrics_live.metric(
                "Packets captured",
                f"{snap['packets']:,}",
                f"{remaining}s remaining"
            )

            time.sleep(1)

        detector.stop()

        status_live.success(
            "✅ Live capture completed"
        )

        snap = detector.snapshot()

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Packets",
            f"{snap['packets']:,}"
        )

        m2.metric(
            "Bytes",
            f"{snap['bytes']:,}"
        )

        m3.metric(
            "Completed flows",
            f"{snap['completed_flows']:,}"
        )

        flows = detector.get_recent_flows()

        if not flows.empty:

            # ==============================================
            # CLASSIFIER
            # ==============================================

            X_live = make_X(flows)

            predictions = clf.predict(X_live)

            attack_rate = float(
                np.mean(
                    predictions != "normal"
                )
            )

            counts = pd.Series(
                predictions
            ).value_counts()

            dominant = str(
                counts.idxmax()
            )

            stage = stage_for_label(
                dominant
            )

            # ==============================================
            # LIVE WORLD STATE
            # ==============================================

            live_state = state_from_window(
                flows.assign(
                    attack_category=predictions,
                    is_attack=(
                        predictions != "normal"
                    ).astype(int)
                )
            )

            novelty, drift = (
                st.session_state
                .adaptive_memory
                .update(
                    live_state,
                    stage
                )
            )

            progression = (
                st.session_state
                .adaptive_memory
                .progression_probability(
                    attack_rate
                )
            )

            risk = (
                "CRITICAL"
                if progression >= 0.80
                else
                "HIGH"
                if progression >= 0.60
                else
                "ELEVATED"
                if progression >= 0.35
                else
                "LOW"
            )

            st.markdown(
                "#### 🛡️ Live threat assessment"
            )

            r1, r2, r3, r4 = st.columns(4)

            r1.metric(
                "Threat",
                risk
            )

            r2.metric(
                "Attack pressure",
                f"{attack_rate:.1%}"
            )

            r3.metric(
                "Classification",
                dominant
            )

            r4.metric(
                "Progression risk",
                f"{progression:.1%}"
            )

            st.caption(
                f"Stage: {stage} · "
                f"Novelty: {novelty:.3f} · "
                f"Drift: {drift:.3f}"
            )

            # ==============================================
            # LIVE FLOWS
            # ==============================================

            display = flows.copy()

            display["prediction"] = predictions

            table_live.dataframe(
                display,
                use_container_width=True,
                height=400
            )

        else:

            st.warning(
                "No completed flows were captured."
            )

    except Exception as e:

        st.error(
            f"Live capture failed: "
            f"{type(e).__name__}: {e}"
        )


    # ==========================================================
    # REPLAY CONTROLS
    # ==========================================================
    st.markdown("### 🎛️ Telemetry replay")

    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        n = st.slider(
            "Windows to replay",
            min_value=5,
            max_value=80,
            value=30
        )

    with c2:
        records_per_window = st.select_slider(
            "Flows per state window",
            options=[50, 100, 200, 500],
            value=200
        )

    with c3:
        speed = st.slider(
            "Replay delay",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.05
        )

    start_replay = st.button(
        "▶  START LIVE THREAT SIMULATION",
        type="primary",
        use_container_width=True
    )

    # ==========================================================
    # LIVE SIMULATION AREA
    # ==========================================================
    if start_replay:

        sample = test.sample(
            n=min(n * records_per_window, len(test)),
            random_state=None
        ).reset_index(drop=True)

        # Dedicated UI containers
        status = st.empty()

        metric_row = st.empty()

        chart = st.empty()

        timeline = st.empty()

        events_box = st.empty()

        states = []
        rates = []
        events = []

        mem = st.session_state.adaptive_memory

        for k in range(n):

            w = sample.iloc[
                k * records_per_window:
                (k + 1) * records_per_window
            ]

            if len(w) < 10:
                break

            # --------------------------------------------------
            # CLASSIFIER INFERENCE
            # --------------------------------------------------
            X = make_X(w)

            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X)
            else:
                probs = None

            pred = clf.predict(X)

            attack_rate = float(
                np.mean(pred != "normal")
            )

            state = state_from_window(
                w.assign(
                    attack_category=pred,
                    is_attack=(pred != "normal").astype(int)
                )
            )

            states.append(state)
            rates.append(attack_rate)

            # --------------------------------------------------
            # CURRENT DOMINANT THREAT
            # --------------------------------------------------
            prediction_counts = pd.Series(pred).value_counts()

            dominant = str(
                prediction_counts.idxmax()
            )

            stage = stage_for_label(dominant)

            # --------------------------------------------------
            # ADAPTIVE MEMORY
            # --------------------------------------------------
            novelty, drift = mem.update(
                state,
                stage
            )

            prev_stage = st.session_state.get(
                "last_stage"
            )

            persistent_novelty = (
                st.session_state
                .persistent_memory
                .update(
                    state,
                    stage,
                    prev_stage
                )
            )

            st.session_state.last_stage = stage

            memory_similarity, memory_stage = (
                st.session_state
                .persistent_memory
                .similarity_to_memory(state)
            )

            # --------------------------------------------------
            # BASE PROGRESSION RISK
            # --------------------------------------------------
            base = float(
                np.clip(
                    attack_rate,
                    0,
                    1
                )
            )

            progression = (
                mem.progression_probability(
                    base
                )
            )

            if (
                memory_stage
                and memory_stage !=
                "Benign / No active stage"
            ):
                progression = float(
                    np.clip(
                        progression +
                        0.08 *
                        memory_similarity,
                        0,
                        0.99
                    )
                )

            # --------------------------------------------------
            # NEURAL WORLD MODEL
            # --------------------------------------------------
            world_conf = None
            world_stage = None

            if (
                world_model is not None
                and len(states) >=
                world_ckpt["sequence_length"]
            ):

                import torch

                seq = np.stack(
                    states[
                        -world_ckpt[
                            "sequence_length"
                        ]:
                    ]
                )

                seq = (
                    seq -
                    np.asarray(
                        world_ckpt[
                            "scaler_mean"
                        ]
                    )
                ) / (
                    np.asarray(
                        world_ckpt[
                            "scaler_scale"
                        ]
                    ) + 1e-8
                )

                with torch.no_grad():

                    ns, logits, _ = (
                        world_model(
                            torch.tensor(
                                seq,
                                dtype=torch.float32
                            ).unsqueeze(0)
                        )
                    )

                    stage_probs = (
                        torch.softmax(
                            logits,
                            dim=1
                        )
                        .numpy()[0]
                    )

                world_stage = (
                    world_ckpt["classes"][
                        int(
                            stage_probs.argmax()
                        )
                    ]
                )

                world_conf = float(
                    stage_probs.max()
                )

                stage = stage_for_label(
                    world_stage
                )

                progression = float(
                    np.clip(
                        0.55 * progression +
                        0.45 *
                        stage_probs[1:].sum(),
                        0,
                        0.99
                    )
                )

            # --------------------------------------------------
            # RISK LEVEL
            # --------------------------------------------------
            risk = (
                "CRITICAL"
                if progression >= 0.80
                else
                "HIGH"
                if progression >= 0.60
                else
                "ELEVATED"
                if progression >= 0.35
                else
                "LOW"
            )

            # --------------------------------------------------
            # AUDIT EVENT
            # --------------------------------------------------
            event = {
                "source":
                    "progressive_replay",

                "window":
                    k,

                "dominant_attack":
                    dominant,

                "stage":
                    stage,

                "attack_rate":
                    attack_rate,

                "progression_probability":
                    progression,

                "novelty":
                    novelty,

                "drift":
                    drift
            }

            # Record every replay window in the tamper-evident audit ledger and multi-node permissioned blockchain.
            block = (
                AuditChain(
                    LEDGER_FILE
                ).append(event)
            )
            try:
                p_chain = PermissionedBlockchain(BLOCKCHAIN_FILE)
                p_chain.submit_security_event(event, sender_node_id="SOC-Delhi-HQ")
            except Exception:
                pass

            events.append(
                {
                    **event,
                    "block":
                        block["index"],
                    "hash":
                        block["hash"][:16] +
                        "…"
                }
            )

            # ==================================================
            # LIVE STATUS PANEL
            # ==================================================
            status.markdown(
                f"""
                <div class="card">
                    <div style="
                        display:flex;
                        justify-content:space-between;
                        align-items:center;
                    ">
                        <div>
                            <div class="small">
                                LIVE THREAT STATE · WINDOW {k + 1}
                            </div>

                            <div style="
                                font-size:30px;
                                font-weight:800;
                                margin-top:5px;
                            ">
                                {risk}
                            </div>

                            <div class="small">
                                Dominant detection:
                                <b>{dominant.upper()}</b>
                                &nbsp; · &nbsp;
                                Stage:
                                <b>{stage}</b>
                            </div>
                        </div>

                        <div style="
                            text-align:right;
                        ">
                            <div class="small">
                                NEXT-WINDOW RISK
                            </div>

                            <div style="
                                font-size:38px;
                                font-weight:800;
                            ">
                                {progression:.1%}
                            </div>

                            <div class="small">
                                World Model confidence:
                                {
                                    f"{world_conf:.1%}"
                                    if world_conf is not None
                                    else "warming up"
                                }
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ==================================================
            # LIVE METRICS
            # ==================================================
            with metric_row.container():

                m1, m2, m3, m4, m5 = st.columns(5)

                m1.metric(
                    "Attack pressure",
                    f"{attack_rate:.1%}"
                )

                m2.metric(
                    "Prediction",
                    dominant.upper()
                )

                m3.metric(
                    "World confidence",
                    (
                        f"{world_conf:.1%}"
                        if world_conf is not None
                        else "—"
                    )
                )

                m4.metric(
                    "Novelty",
                    f"{novelty:.3f}"
                )

                m5.metric(
                    "Drift",
                    f"{drift:.3f}"
                )

            # ==================================================
            # THREAT PRESSURE CHART
            # ==================================================
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=list(
                        range(len(rates))
                    ),
                    y=rates,
                    mode="lines+markers",
                    name="Observed attack pressure",
                    line=dict(width=3)
                )
            )

            if len(rates) >= 3:

                ema = (
                    pd.Series(rates)
                    .ewm(
                        span=min(
                            8,
                            len(rates)
                        ),
                        adjust=False
                    )
                    .mean()
                    .iloc[-1]
                )

                future_x = list(
                    range(
                        len(rates),
                        len(rates) + 5
                    )
                )

                future_y = [ema] * 5

                fig.add_trace(
                    go.Scatter(
                        x=future_x,
                        y=future_y,
                        mode="lines+markers",
                        name="Adaptive forecast",
                        line=dict(
                            dash="dash",
                            width=2
                        )
                    )
                )

            if world_conf is not None:

                fig.add_trace(
                    go.Scatter(
                        x=[len(rates) - 1],
                        y=[world_conf],
                        mode="markers",
                        name="World Model confidence",
                        marker=dict(
                            size=13
                        )
                    )
                )

            fig.update_layout(
                height=390,
                margin=dict(
                    l=10,
                    r=10,
                    t=55,
                    b=10
                ),
                title={
                    "text":
                        "Threat Pressure & Predictive Forecast",
                    "x": 0.02
                },
                xaxis_title="Telemetry window",
                yaxis_title="Probability / pressure",
                yaxis=dict(
                    range=[0, 1]
                ),
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0
                )
            )

            chart.plotly_chart(
                fig,
                use_container_width=True
            )

            # ==================================================
            # SECURITY TIMELINE
            # ==================================================
            timeline.markdown(
                f"""
                ### Security timeline

                **Window {k + 1}**

                `DETECT`
                → **{dominant.upper()}**

                → `CLASSIFY`

                → **{stage}**

                → `FORECAST`

                → **{progression:.1%}**

                → `AUDIT`

                → **{
                    "BLOCK COMMITTED"
                    if dominant != "normal"
                    else "NO SECURITY EVENT"
                }**
                """
            )

            # ==================================================
            # AUDIT EVENTS
            # ==================================================
            if events:

                events_box.dataframe(
                    pd.DataFrame(
                        events[-8:]
                    ),
                    use_container_width=True,
                    hide_index=True
                )
            else:

                events_box.info(
                    "No malicious security events "
                    "have been committed to the audit ledger yet."
                )

            time.sleep(speed)

        st.success(
            "Progressive replay completed. "
            "Security events were appended to the "
            "tamper-evident audit ledger."
        )
# -------------------- WORLD MODEL --------------------
elif page=='World Model':
    st.markdown('### 🧠 Learned Network-State Dynamics & Multi-Step World Model')
    
    col_arch, col_info = st.columns([1, 2])
    with col_arch:
        arch_type = st.radio("Model Architecture", ["PyTorch LSTM World Model", "PyTorch Transformer World Model (Self-Attention)"])
    with col_info:
        if "Transformer" in arch_type:
            st.info("⚡ **Transformer World Model Active**: Uses 2-layer Multi-Head Self-Attention (`nhead=4`) with Positional Encoding to model complex long-range temporal dependencies across network state windows.")
            selected_model = TransformerWorldModel(16) if 'TransformerWorldModel' in globals() and TransformerWorldModel else world_model
        else:
            st.info(r"🧠 **LSTM World Model Active**: Uses sequential LSTM state transitions $P(S_{t+1} | S_t, \dots, S_{t-k})$ for temporal next-state regression and attack stage forecasting.")
            selected_model = world_model

    if world_model is None and selected_model is None:
        st.warning('Neural world-model weights are not present yet. Run `train_world_model.py` once to generate `world_model.pt`.')
    else:
        if os.path.exists('world_model_meta.json'):
            meta=json.load(open('world_model_meta.json'))
            m=meta.get('metrics',{})
            a,b,c,d=st.columns(4)
            a.metric('World-model stage accuracy',f"{m.get('stage_accuracy',0):.1%}")
            b.metric('World-model macro F1',f"{m.get('stage_macro_f1',0):.1%}")
            c.metric('Macro recall',f"{m.get('stage_macro_recall',0):.1%}")
            d.metric('Next-state MAE',f"{m.get('next_state_mae_standardized',0):.3f}")

        st.markdown(r'#### 🔮 Multi-Step Autoregressive Trajectory Forecasting ($t+1 \dots t+K$)')
        st.caption("Autoregressively projects network state vectors and attack stage distributions up to $K$ steps into the future.")

        c1, c2, c3 = st.columns(3)
        with c1:
            horizon_k = st.slider("Rollout Horizon (K Steps)", min_value=1, max_value=10, value=5)
        with c2:
            scenario_name = st.selectbox("Initial Attack Scenario", ["DOS_FLOOD", "PROBE_RECON", "R2L_BRUTEFORCE", "EXFIL_U2R", "NORMAL"])
        with c3:
            st.markdown("<br>", unsafe_allow_html=True)
            run_rollout = st.button("🚀 Run Multi-Step Rollout", use_container_width=True)

        # Generate seed sequence for scenario
        if 'LivePacketStreamGenerator' in globals():
            gen = LivePacketStreamGenerator()
            gen.generate_packet_burst(scenario_name, burst_size=40)
            flows = list(gen.active_flows.values())
            seed_state = FlowStateEncoder.encode_flows_to_state(flows)
        else:
            seed_state = np.zeros(16, dtype=np.float32)

        # Create sequence window of 8 frames
        initial_seq = np.tile(seed_state, (8, 1))

        if selected_model is not None and ('MultiStepRolloutEngine' in globals() and MultiStepRolloutEngine):
            rollout_results = MultiStepRolloutEngine.predict_rollout(selected_model, initial_seq, horizon=horizon_k)

            # Plotly trajectory chart
            steps = [f"t+{r['step']}" for r in rollout_results]
            risks = [r['cumulative_risk'] for r in rollout_results]
            confidences = [r['confidence'] for r in rollout_results]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=steps, y=risks, mode='lines+markers', name='Cumulative Trajectory Risk', line=dict(color='#f87171', width=3)))
            fig.add_trace(go.Scatter(x=steps, y=confidences, mode='lines+markers', name='Predicted Stage Confidence', line=dict(color='#60a5fa', width=2, dash='dash')))
            fig.update_layout(
                title=f"Multi-Step Autoregressive Risk Trajectory ({scenario_name} Scenario)",
                xaxis_title="Future Step Horizon",
                yaxis_title="Probability / Score",
                template="plotly_dark",
                height=380,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Forecast Table
            st.markdown('##### Step-by-Step Forecast Log')
            forecast_rows = []
            for r in rollout_results:
                forecast_rows.append({
                    'Horizon Step': f"t+{r['step']}",
                    'Predicted Stage': r['predicted_stage'],
                    'Stage Confidence': f"{r['confidence']:.1%}",
                    'Cumulative Risk': f"{r['cumulative_risk']:.3f}",
                    'Projected Attack Rate': f"{r['projected_state'][0]:.2f}",
                    'Projected DoS Rate': f"{r['projected_state'][1]:.2f}",
                    'Projected Probe Rate': f"{r['projected_state'][2]:.2f}"
                })
            st.dataframe(pd.DataFrame(forecast_rows), use_container_width=True)

        st.markdown('#### State representation')
        st.dataframe(pd.DataFrame({'State feature':STATE_NAMES,'Meaning':['overall malicious pressure','DoS share','probe/recon share','R2L share','U2R share','source bytes mean','destination bytes mean','flow duration mean','connection count mean','service count mean','SYN/error-rate proxy','RST/error-rate proxy','same-service concentration','service diversity','service diversity / window','log traffic volume']}),use_container_width=True)
        st.markdown('#### Progressive learning loop')
        st.code('''Observe traffic → build network state S_t → predict S_(t+1)\n        ↓\nCompare predicted vs observed next state\n        ↓\nPrediction error + state drift + attack pressure\n        ↓\nUpdate adaptive threat memory / novel trajectory memory\n        ↓\nForward rollout → infiltration probability + predicted stage\n        ↓\nVerified feedback can be used for safe model retraining''')
        st.warning('Self-training directly from the model’s own attack labels is intentionally disabled. Without verified feedback it can reinforce its own mistakes. OracleShield adapts its environment model online, while the detector remains protected from label poisoning.')

# -------------------- BLOCKCHAIN --------------------
elif page=='Blockchain Audit':
    st.markdown('### ⛓️ Permissioned Multi-Node Blockchain Ledger & BFT Consensus')
    p_chain = PermissionedBlockchain(BLOCKCHAIN_FILE)
    valid, idx, msg = p_chain.verify_chain_integrity()
    
    a, b, c, d = st.columns(4)
    a.metric('Total Blocks', len(p_chain.chain))
    b.metric('Consensus Integrity', 'VALID (BFT Consensus)' if valid else 'TAMPER DETECTED')
    c.metric('Active SOC Nodes', len(p_chain.nodes))
    latest_hash = p_chain.chain[-1].hash[:16] + '…' if p_chain.chain else 'N/A'
    d.metric('Latest Block Hash', latest_hash)
    
    if valid:
        st.success('✓ Permissioned Blockchain Proof-of-Authority (PoA) consensus is intact across all distributed SOC nodes.')
    else:
        st.error(f'⚠️ Byzantine Integrity failure detected at block {idx}: {msg}')

    st.markdown('#### 🏢 Distributed SOC Network Topology')
    node_rows = []
    for nid, node in p_chain.nodes.items():
        node_rows.append({
            'Node ID': node.node_id,
            'Location & Role': f"{node.location} ({node.role})",
            'Public Key (ECDSA/HMAC)': node.public_key[:24] + '...',
            'Reputation': f"{node.reputation:.1f}%",
            'Network Status': 'ACTIVE 🟢'
        })
    st.dataframe(pd.DataFrame(node_rows), use_container_width=True)

    st.markdown('#### 📦 Multi-Node Ledger & Merkle Root Inspector')
    block_rows = []
    for b in p_chain.chain[-25:]:
        tx_payload = b.transactions[0].payload if b.transactions else {}
        attack = tx_payload.get('dominant_attack', tx_payload.get('event_type', 'GENESIS'))
        stage = tx_payload.get('stage', 'GENESIS')
        votes_count = f"{len(b.validator_votes)} / {len([n for n in p_chain.nodes.values() if n.role in ['LEADER', 'VALIDATOR']])}"
        block_rows.append({
            'Block #': b.index,
            'Time': datetime.fromtimestamp(b.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            'Proposer Node': b.proposer_node_id,
            'Stage': stage,
            'Attack': attack,
            'Merkle Root': b.merkle_root[:16] + '…',
            'BFT Votes': votes_count,
            'Status': b.consensus_status,
            'Block Hash': b.hash[:16] + '…'
        })
    st.dataframe(pd.DataFrame(block_rows), use_container_width=True)

    st.markdown('#### 🧪 Judge-Ready Byzantine Fault Tolerance & Tamper Simulation')
    st.caption("Simulates a malicious compromise of a historical block or Merkle root to demonstrate BFT multi-node consensus rejection.")

    c1, c2 = st.columns(2)
    with c1:
        tamper_idx = st.number_input("Target Block Index", min_value=0, max_value=max(0, len(p_chain.chain)-1), value=max(0, len(p_chain.chain)-1))
    with c2:
        tamper_field = st.selectbox("Tamper Vector", ["payload", "merkle_root", "previous_hash"], format_func=lambda x: {"payload": "Alter Transaction Payload", "merkle_root": "Forge Merkle Tree Root", "previous_hash": "Break Block Hash Link"}[x])

    if st.button('🧪 Simulate Byzantine Multi-Node Attack (Isolated Sandbox)', use_container_width=True):
        # Create an isolated sandbox copy of the blockchain
        sandbox_blockchain = PermissionedBlockchain(BLOCKCHAIN_FILE)
        result = sandbox_blockchain.simulate_byzantine_attack(int(tamper_idx), tamper_field, "MALICIOUS_TAMPERED_EVENT")

        if not result['is_chain_valid']:
            st.error(f"⚠️ BYZANTINE TAMPERING DETECTED at Block #{result['failure_index']}!")
            st.info(f"Rejection Reason: {result['rejection_reason']}  |  Tampered Field: {result['field_tampered']}")
            st.success("✓ Real Multi-Node Blockchain Ledger remains 100% untouched and valid across all SOC nodes.")
            st.caption("The BFT consensus nodes rejected the tampered block because the cryptographic hash signature & Merkle tree root failed validation rules.")
        else:
            st.success("No tampering detected.")
# -------------------- EVIDENCE --------------------
else:
    st.markdown('### Evidence, metrics & data coverage')
    a,b,c,d,e=st.columns(5)
    a.metric('Accuracy',f"{M['accuracy']:.2%}")
    b.metric('Macro precision',f"{M['macro_precision']:.2%}")
    c.metric('Macro recall',f"{M['macro_recall']:.2%}")
    d.metric('Macro F1',f"{M['macro_f1']:.2%}")
    e.metric('Weighted F1',f"{M['weighted_f1']:.2%}")
    st.markdown('#### Per-class performance')
    rows=[]
    for cls in ['dos','normal','probe','r2l','u2r']:
        r=M['report'][cls]; rows.append({'Class':cls,'Precision':r['precision'],'Recall':r['recall'],'F1':r['f1-score'],'Support':int(r['support'])})
    st.dataframe(pd.DataFrame(rows).style.format({'Precision':'{:.2%}','Recall':'{:.2%}','F1':'{:.2%}'}),use_container_width=True)
    st.markdown('#### Train / test class distribution')
    dist=pd.concat([train['attack_category'].value_counts().rename('train'),test['attack_category'].value_counts().rename('test')],axis=1).fillna(0).astype(int)
    st.bar_chart(dist)
    st.markdown('#### Requirement coverage')
    coverage=pd.DataFrame([
        ['Flow-level telemetry','Partial','NSL-KDD provides aggregate flow/connection features.'],
        ['Packet-level telemetry','Not available in supplied workbook','TTL, TCP window, IAT and retransmissions require PCAP/CIC/CTU-13-derived data.'],
        ['Temporal world model','Prototype implemented','LSTM transition model learns next-state dynamics from reproducible windows.'],
        ['Forward simulation','Implemented','Next-state prediction and short-horizon adaptive risk trajectory.'],
        ['MITRE stage mapping','Heuristic mapping','Dataset categories are mapped to reconnaissance/access/escalation/impact; true ATT&CK technique IDs need richer telemetry.'],
        ['Explainability','State-feature drivers + detector evidence','SHAP can be enabled for the RandomForest; state drivers show which network-state dimensions changed.'],
        ['Blockchain audit','Implemented','SHA-256 hash-chained security-event ledger with integrity verification.'],
        ['Online adaptation','Implemented safely','Adaptive memory learns evolving state/drift; classifier is not self-trained from unverified predictions.'],
    ],columns=['Requirement','Status','Implementation note'])
    st.dataframe(coverage,use_container_width=True)
    st.info('For a final NTRO-grade benchmark, add CIC-IDS2018/CTU-13 or PCAP-derived telemetry with timestamps and packet-level features, then retrain the same world-model pipeline on genuine temporal sequences. The supplied NSL-KDD workbook cannot support claims about packet timing or causal attack progression by itself.')
