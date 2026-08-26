"""OracleShield World Model trainer.

Prototype temporal world model for NSL-KDD.

Important:
NSL-KDD has no genuine timestamps. Row order is therefore used only as a
reproducible pseudo-temporal ordering. For final benchmarking, use genuinely
timestamped telemetry such as CIC-IDS2018/CTU-13/PCAP-derived features.
"""

import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
)
from torch.utils.data import TensorDataset, DataLoader

from oracle_shield_world_model import (
    build_state_series,
    STATE_NAMES,
    WorldModel,
)


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# CPU is safer on the current machine because the installed PyTorch CUDA
# build is newer than the installed NVIDIA driver.
torch.set_num_threads(1)


p = argparse.ArgumentParser()

p.add_argument(
    "--data",
    default="Copy of DOC-20260825-WA0002.xlsx"
)

p.add_argument("--out", default="world_model.pt")
p.add_argument("--meta", default="world_model_meta.json")

# 50 gives substantially better local attack-state separation than 200.
p.add_argument("--window", type=int, default=50)

# Short temporal context.
p.add_argument("--seq", type=int, default=8)

p.add_argument("--epochs", type=int, default=12)
p.add_argument("--batch", type=int, default=128)

args = p.parse_args()


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

print("\n[1/6] Loading dataset...")

df = pd.read_excel(args.data)

train = (
    df[df["split"].astype(str).str.lower().eq("train")]
    .reset_index(drop=True)
)

test = (
    df[df["split"].astype(str).str.lower().eq("test")]
    .reset_index(drop=True)
)

print("Train rows:", len(train))
print("Test rows :", len(test))


# ---------------------------------------------------------------------
# BUILD STATE SERIES
# ---------------------------------------------------------------------

print("\n[2/6] Building temporal states...")
print("Window:", args.window)

train_states, train_labels = build_state_series(
    train,
    args.window
)

test_states, test_labels = build_state_series(
    test,
    args.window
)

print("Train states:", len(train_states))
print("Test states :", len(test_states))

print("\nTrain state distribution:")
print(pd.Series(train_labels).value_counts())

print("\nTest state distribution:")
print(pd.Series(test_labels).value_counts())


# ---------------------------------------------------------------------
# SCALE
# ---------------------------------------------------------------------

print("\n[3/6] Scaling state features...")

scaler = StandardScaler()

train_s = scaler.fit_transform(train_states).astype("float32")
test_s = scaler.transform(test_states).astype("float32")


# ---------------------------------------------------------------------
# SEQUENCES
# ---------------------------------------------------------------------

classes = ["normal", "probe", "r2l", "u2r", "dos"]
c2i = {c: i for i, c in enumerate(classes)}


def make_sequences(arr, labels):

    X = []
    Y = []
    L = []

    for i in range(args.seq, len(arr)):

        X.append(arr[i - args.seq:i])

        # Next state target
        Y.append(arr[i])

        # Current target class
        L.append(c2i.get(labels[i], 0))

    return (
        np.asarray(X, dtype="float32"),
        np.asarray(Y, dtype="float32"),
        np.asarray(L, dtype="int64"),
    )


Xtr, Ytr, Ltr = make_sequences(
    train_s,
    train_labels
)

Xte, Yte, Lte = make_sequences(
    test_s,
    test_labels
)

if len(Xtr) == 0 or len(Xte) == 0:
    raise SystemExit(
        "Not enough temporal states for the selected window/sequence."
    )

print("\nTrain sequences:", len(Xtr))
print("Test sequences :", len(Xte))


# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------

print("\n[4/6] Building LSTM world model...")

model = WorldModel(
    Xtr.shape[-1],
    hidden=96,
    classes=len(classes),
)


# ---------------------------------------------------------------------
# CLASS WEIGHTS
# ---------------------------------------------------------------------
#
# Do NOT use inverse-frequency weights here.
#
# U2R has only 52 examples and enormous inverse weighting destabilizes the
# neural model. Instead use a square-root weighting with a cap.
#
# This gives minority classes more importance without forcing the network
# to predict every rare class everywhere.

class_counts = np.bincount(
    Ltr,
    minlength=len(classes)
)

raw_weights = 1.0 / np.sqrt(
    np.maximum(class_counts, 1)
)

raw_weights = raw_weights / raw_weights.mean()

# Prevent extreme U2R weighting.
class_weights = np.clip(
    raw_weights,
    0.5,
    3.0
)

print("\nClass counts:")

print(
    dict(
        zip(
            classes,
            class_counts.tolist()
        )
    )
)

print("\nClass weights:")

print(
    dict(
        zip(
            classes,
            class_weights.round(3).tolist()
        )
    )
)


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------

opt = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

reg = nn.SmoothL1Loss()

ce = nn.CrossEntropyLoss(
    weight=torch.tensor(
        class_weights,
        dtype=torch.float32
    ),
    label_smoothing=0.05,
)


loader = DataLoader(
    TensorDataset(
        torch.tensor(Xtr),
        torch.tensor(Ytr),
        torch.tensor(Ltr),
    ),
    batch_size=args.batch,
    shuffle=True,
)


print("\n[5/6] Training...")

for epoch in range(args.epochs):

    model.train()

    total = 0.0

    for xb, yb, lb in loader:

        opt.zero_grad()

        next_state, logits, _ = model(xb)

        # State prediction is the main world-model objective.
        state_loss = reg(
            next_state,
            yb
        )

        # Classification is secondary.
        stage_loss = ce(
            logits,
            lb
        )

        loss = (
            state_loss
            + 0.25 * stage_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        opt.step()

        total += loss.item() * len(xb)

    print(
        f"epoch {epoch + 1}/{args.epochs} "
        f"loss={total / len(Xtr):.4f}"
    )


# ---------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------

print("\n[6/6] Evaluating...")

model.eval()

with torch.no_grad():

    yp, logits, _ = model(
        torch.tensor(Xte)
    )

    pred = logits.argmax(1).numpy()

    mae = float(
        torch.mean(
            torch.abs(
                yp -
                torch.tensor(Yte)
            )
        ).item()
    )


accuracy = accuracy_score(
    Lte,
    pred
)

macro_f1 = f1_score(
    Lte,
    pred,
    average="macro",
    zero_division=0
)

macro_precision = precision_score(
    Lte,
    pred,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    Lte,
    pred,
    average="macro",
    zero_division=0
)


metrics = {

    "stage_accuracy": float(accuracy),

    "stage_macro_f1": float(macro_f1),

    "stage_macro_precision": float(
        macro_precision
    ),

    "stage_macro_recall": float(
        macro_recall
    ),

    "next_state_mae_standardized": float(mae),

    "window_size": args.window,

    "sequence_length": args.seq,

    "train_sequences": len(Xtr),

    "test_sequences": len(Xte),

    "temporal_note":
        "NSL-KDD has no timestamps; row order is used only "
        "as a reproducible prototype sequence."
}


print("\n===== WORLD MODEL RESULTS =====")

print(
    classification_report(
        Lte,
        pred,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0
    )
)

print(
    json.dumps(
        metrics,
        indent=2
    )
)


# ---------------------------------------------------------------------
# SAVE CHECKPOINT
# ---------------------------------------------------------------------

ckpt = {

    "model": model.state_dict(),

    "input_dim": Xtr.shape[-1],

    "classes": classes,

    "window_size": args.window,

    "sequence_length": args.seq,

    "scaler_mean": scaler.mean_.tolist(),

    "scaler_scale": scaler.scale_.tolist(),

    "state_names": STATE_NAMES,
}


torch.save(
    ckpt,
    args.out
)


with open(
    args.meta,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "metrics": metrics,
            "classes": classes,
            "state_names": STATE_NAMES,
            "window_size": args.window,
            "sequence_length": args.seq,
            "data": args.data,
            "class_counts":
                dict(
                    zip(
                        classes,
                        class_counts.tolist()
                    )
                ),
            "class_weights":
                dict(
                    zip(
                        classes,
                        class_weights.tolist()
                    )
                ),
            "temporal_note":
                "NSL-KDD has no timestamps; row order is used only "
                "as a reproducible prototype sequence.",
        },
        f,
        indent=2
    )


print(
    f"\nSaved {args.out}"
)

print(
    f"Saved {args.meta}"
)
