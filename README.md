# OracleShield

### AI-Based Network Attack Forecasting with Progressive World Models

**SIH26153 · National Technical Research Organisation (NTRO)**

OracleShield is a prototype for proactive cyber defence that moves beyond
isolated intrusion classification. It represents network traffic as
evolving state windows, learns temporal state-transition dynamics, performs
forward reasoning about attacker progression, maintains adaptive threat
memory, and records security decisions in a tamper-evident audit ledger.

> **Traditional IDS:** What attack is happening now?
>
> **OracleShield:** Given the network state now, where is the threat likely
> to move next?

## Architecture

```text
Network Traffic
      |
      v
Network State S_t
      |
      +------------------+
      |                  |
      v                  v
RandomForest        LSTM World Model
Detection           P(S_t+1 | S_t...)
      |                  |
      +--------+---------+
               v
      Forward State Rollout
               |
               v
   Progression / Risk Estimate
               |
      +--------+---------+
      |                  |
      v                  v
MITRE ATT&CK stage   Explainability
      |                  |
      +--------+---------+
               v
      SHA-256 Audit Chain
               |
               v
       Streamlit SOC UI
```

## What the current prototype does

- Uses the supplied combined NSL-KDD workbook as the primary prototype data.
- Preserves train/test membership using the `split` column.
- Uses a RandomForest as the detection/baseline layer.
- Builds 16-dimensional network-state vectors over traffic windows.
- Trains an LSTM to predict the next state and attack-stage distribution.
- Performs forward risk estimation from observed state, drift and novelty.
- Maintains persistent threat-memory prototypes for recurring trajectories.
- Maps prototype attack categories to high-level ATT&CK stages.
- Stores non-benign security events in a SHA-256 hash-chained ledger.
- Includes a live tamper demonstration in the Streamlit interface.

## Dataset provenance

Current workbook:

`Copy of DOC-20260825-WA0002.xlsx`

Prototype size:

- 148,517 records
- 125,973 train
- 22,544 test

The original train/test membership is preserved in `split`.

## Current detector evidence

The supplied prototype evaluation reports approximately **73.67% accuracy**
on the difficult test setting. OracleShield exposes macro precision, macro
recall, macro F1 and per-class metrics so accuracy is not presented in
isolation on this imbalanced intrusion-detection problem.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the combined workbook and trained detector artifacts in the project
root:

```text
OracleShield/
├── app_oracleshield_progressive.py
├── oracle_shield_world_model.py
├── train_world_model.py
├── preprocess.py
├── requirements.txt
├── Copy of DOC-20260825-WA0002.xlsx
├── model_classifier.joblib
├── scaler.joblib
└── feature_columns.joblib
```

## Train the World Model

```bash
python train_world_model.py --data "Copy of DOC-20260825-WA0002.xlsx"
```

This generates:

- `world_model.pt`
- `world_model_meta.json`

Then launch:

```bash
streamlit run app_oracleshield_progressive.py
```

Open the local Streamlit URL shown in the terminal.

## Dashboard workspaces

### Command Center
Replays traffic windows and shows attack pressure, progression probability,
predicted stage, novelty, drift, memory match and recent security events.

### World Model
Shows the state representation, model status and progressive learning loop.

### Blockchain Audit
Displays the hash-chained ledger, integrity status and a judge-ready tamper
simulation.

### Evidence & Data
Shows detector metrics, class distributions and requirement coverage.

## Progressive learning: what it really means

OracleShield intentionally does **not** train the classifier on its own
predictions. Doing so would allow a wrong prediction to become its own
future training label.

Instead, adaptation occurs through:

```text
observation
   -> state update
   -> novelty / drift measurement
   -> persistent trajectory memory
   -> transition statistics
   -> future comparison
```

A production autonomous learning loop should add:

```text
new observation
   -> novelty detection
   -> shadow learner
   -> analyst / ground-truth verification
   -> replay buffer
   -> drift test
   -> retraining
   -> validation gate
   -> model promotion
```

## Important research limitation

NSL-KDD does not provide the timestamped packet/PCAP telemetry described
by the full SIH problem statement. Therefore this prototype must not claim
that it has learned real packet-level causal attacker progression from
NSL-KDD alone.

For the final NTRO-grade benchmark, use timestamped telemetry such as
CIC-IDS2018, CTU-13, CICIoT2023 or PCAP-derived data and add:

- inter-arrival timing
- TTL and variance
- TCP window statistics
- retransmissions
- sequential/randomised port-scan behaviour
- flow duration and bidirectional ratios
- real chronological attack timelines

## Blockchain positioning

The current ledger is a **permissioned-style local hash chain** for
tamper-evident security auditing. It is not a public blockchain, mining
system, cryptocurrency or multi-node consensus network.

A production deployment can extend this layer across trusted security nodes
with access control and consensus.

## MITRE ATT&CK positioning

The supplied NSL-KDD categories do not contain native MITRE technique IDs.
The prototype therefore uses a transparent high-level mapping:

| Prototype category | Stage |
|---|---|
| probe | Reconnaissance |
| r2l | Initial Access |
| u2r | Privilege Escalation |
| dos | Impact / Disruption |
| normal | No active stage |

The final system should map richer telemetry to actual ATT&CK techniques
with evidence rather than treating this heuristic mapping as ground truth.

## Repository hygiene

Dataset files and trained binary artifacts are excluded from Git by design.
This keeps the public source repository lightweight and prevents accidental
publication of large data/model files.

## License

Add the license required by your SIH/team submission policy before public
distribution.
