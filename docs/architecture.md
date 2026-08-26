# OracleShield Architecture

## Core pipeline

Network telemetry
-> network-state encoder
-> temporal World Model
-> next-state prediction
-> forward rollout
-> infiltration/progression probability
-> MITRE ATT&CK stage mapping
-> explainable decision support
-> tamper-evident audit ledger

## Detection layer

A RandomForest detector provides the baseline classification layer using
the combined NSL-KDD workbook.

## World Model

The prototype represents each traffic window as a state vector containing
attack pressure, category proportions, byte statistics, duration, counts,
service diversity and traffic volume.

An LSTM learns:

    P(S_(t+1) | S_t, S_(t-1), ...)

It predicts both the next network-state vector and an attack-stage
distribution.

## Progressive learning

OracleShield maintains adaptive state statistics and persistent episodic
threat memory. Novel trajectories can be remembered and compared against
future observations.

The detector is deliberately not retrained from its own predictions.
Autonomous self-labeling would create a self-reinforcing error loop.
Production online learning should use verified feedback, drift tests,
replay buffers and a validation gate before model promotion.

## Blockchain audit

Security events are stored in a SHA-256 hash-chained ledger. Each block
contains the previous block hash and an event hash. Changing an old event
breaks verification.

This is a permissioned-style local audit ledger, not a public cryptocurrency
network or multi-node consensus blockchain.

## SIH alignment

The prototype demonstrates the requested World Model direction while
explicitly documenting the limits of NSL-KDD. For the final benchmark,
replace row-order windows with genuine timestamped flow/packet telemetry
and add technique-level MITRE ATT&CK evidence.
