"""
preprocess.py
Shared loading + preprocessing utilities for the NSL-KDD dataset.
Used by both train_classifier.py and train_forecaster.py so both
scripts see identical, consistent features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label",
]

# Map the ~23 fine-grained NSL-KDD attack labels down to the 4 standard
# attack categories used in most published NSL-KDD papers, plus 'normal'.
ATTACK_CATEGORY_MAP = {
    "normal": "normal",
    # DoS
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "mailbomb": "dos", "apache2": "dos",
    "processtable": "dos", "udpstorm": "dos",
    # Probe
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "satan": "probe", "mscan": "probe", "saint": "probe",
    # R2L (remote to local)
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l",
    "multihop": "r2l", "phf": "r2l", "spy": "r2l", "warezclient": "r2l",
    "warezmaster": "r2l", "sendmail": "r2l", "named": "r2l", "snmpgetattack": "r2l",
    "snmpguess": "r2l", "xlock": "r2l", "xsnoop": "r2l", "worm": "r2l",
    # U2R (user to root)
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r",
    "rootkit": "u2r", "httptunnel": "u2r", "ps": "u2r", "sqlattack": "u2r",
    "xterm": "u2r",
}


def load_dataset(path):
    """Load a raw NSL-KDD CSV (no header) and attach column names."""
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    # some copies of this dataset have a trailing 'difficulty' column - drop
    # anything beyond our named columns defensively
    df = df.iloc[:, : len(COLUMN_NAMES)]
    df.columns = COLUMN_NAMES
    df["label"] = df["label"].str.strip()
    df["attack_category"] = df["label"].map(ATTACK_CATEGORY_MAP).fillna("unknown")
    df["is_attack"] = (df["attack_category"] != "normal").astype(int)
    return df


def encode_features(train_df, test_df):
    """
    One-hot encode categorical columns, align train/test columns,
    scale numeric features. Returns X_train, X_test (numpy arrays),
    and the fitted scaler + column list for reuse at inference time.
    """
    cat_cols = ["protocol_type", "service", "flag"]
    drop_cols = ["label", "attack_category", "is_attack", "split"]

    train_X = pd.get_dummies(train_df.drop(columns=drop_cols), columns=cat_cols)
    test_X = pd.get_dummies(test_df.drop(columns=drop_cols), columns=cat_cols)

    # align columns (test set may be missing some service/flag categories)
    train_X, test_X = train_X.align(test_X, join="left", axis=1, fill_value=0)

    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    test_X_scaled = scaler.transform(test_X)

    return train_X_scaled, test_X_scaled, scaler, list(train_X.columns)
