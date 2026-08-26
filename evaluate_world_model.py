import json
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

from oracle_shield_world_model import (
    build_state_series,
    WorldModel
)

DATA = "Copy of DOC-20260825-WA0002.xlsx"
CKPT = "world_model.pt"

df = pd.read_excel(DATA)

train = df[df["split"].astype(str).str.lower().eq("train")].reset_index(drop=True)
test = df[df["split"].astype(str).str.lower().eq("test")].reset_index(drop=True)

ckpt = torch.load(CKPT, map_location="cpu")

window = ckpt["window_size"]
seq = ckpt["sequence_length"]
classes = ckpt["classes"]

train_states, train_labels = build_state_series(train, window)
test_states, test_labels = build_state_series(test, window)

scaler = StandardScaler()
scaler.fit(train_states)

test_s = scaler.transform(test_states).astype("float32")

c2i = {c: i for i, c in enumerate(classes)}

X = []
Y = []

for i in range(seq, len(test_s)):
    X.append(test_s[i-seq:i])
    Y.append(c2i.get(test_labels[i], 0))

X = np.asarray(X, dtype="float32")
Y = np.asarray(Y, dtype="int64")

model = WorldModel(
    X.shape[-1],
    hidden=96,
    classes=len(classes)
)

model.load_state_dict(ckpt["model"])
model.eval()

with torch.no_grad():
    _, logits, _ = model(torch.tensor(X))
    pred = logits.argmax(1).numpy()

print("\n===== WORLD MODEL PER-CLASS RESULTS =====\n")

print(
    classification_report(
        Y,
        pred,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0,
        digits=4
    )
)

print("===== CONFUSION MATRIX =====\n")

cm = confusion_matrix(
    Y,
    pred,
    labels=list(range(len(classes)))
)

print(pd.DataFrame(
    cm,
    index=[f"actual_{c}" for c in classes],
    columns=[f"pred_{c}" for c in classes]
))

print("\n===== TEST SEQUENCE DISTRIBUTION =====\n")

print(
    pd.Series(
        [classes[i] for i in Y]
    ).value_counts().reindex(classes, fill_value=0)
)

print("\n===== PREDICTION DISTRIBUTION =====\n")

print(
    pd.Series(
        [classes[i] for i in pred]
    ).value_counts().reindex(classes, fill_value=0)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

disp.plot()
plt.title("OracleShield World Model — Test Confusion Matrix")
plt.tight_layout()
plt.savefig("world_model_confusion_matrix.png", dpi=200)

print("\nSaved: world_model_confusion_matrix.png")
