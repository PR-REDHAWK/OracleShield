import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from preprocess import encode_features


DATA_FILE = "Copy of DOC-20260825-WA0002.xlsx"

CLASSIFIER_FILE = "model_classifier.joblib"
SCALER_FILE = "scaler.joblib"
FEATURE_FILE = "feature_columns.joblib"


print("[1/5] Loading dataset...")

df = pd.read_excel(DATA_FILE)

train_df = df[
    df["split"].astype(str).str.lower().eq("train")
].copy()

test_df = df[
    df["split"].astype(str).str.lower().eq("test")
].copy()

print("Train rows:", len(train_df))
print("Test rows :", len(test_df))


print("\n[2/5] Encoding features...")

X_train, X_test, scaler, feature_columns = encode_features(
    train_df,
    test_df
)

y_train = train_df["attack_category"].astype(str).values
y_test = test_df["attack_category"].astype(str).values

print("Features:", X_train.shape[1])

print("\nTrain distribution:")
print(pd.Series(y_train).value_counts())


print("\n[3/5] Training classifier...")

clf = RandomForestClassifier(
    n_estimators=250,
    max_depth=20,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_train)


print("\n[4/5] Evaluating...")

pred = clf.predict(X_test)

print("\n===== CLASSIFIER RESULTS =====")
print(classification_report(
    y_test,
    pred,
    labels=["normal", "probe", "r2l", "u2r", "dos"],
    zero_division=0
))

print("Accuracy:", accuracy_score(y_test, pred))


print("\n[5/5] Saving artifacts...")

joblib.dump(clf, CLASSIFIER_FILE)
joblib.dump(scaler, SCALER_FILE)
joblib.dump(feature_columns, FEATURE_FILE)

print("\nSaved:")
print(" ", CLASSIFIER_FILE)
print(" ", SCALER_FILE)
print(" ", FEATURE_FILE)
