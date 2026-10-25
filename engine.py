"""Simple, explainable demo engine for CyberShield AI.

Important: the built-in model is trained on synthetic teaching data. It is not
validated for real-world intrusion detection. Use a documented network dataset
and a leakage-aware train/test evaluation before making performance claims.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

REQUIRED_FEATURES = [
    "flow_duration", "total_fwd_packets", "total_backward_packets",
    "flow_bytes_per_sec", "flow_packets_per_sec", "syn_flag_count",
    "dst_port", "packet_length_mean"
]
LABELS = ["Benign", "DDoS", "PortScan", "BruteForce", "Botnet"]

def _synthetic_features(n, seed=42):
    rng = np.random.default_rng(seed)
    labels = rng.choice(LABELS, size=n, p=[.62,.14,.12,.07,.05])
    rows = []
    for lab in labels:
        if lab == "Benign":
            duration = rng.lognormal(7.0, .65)
            fwd = rng.integers(2, 80); bwd = rng.integers(1, 90)
            bps = rng.lognormal(8.0, .8); pps = rng.lognormal(2.8, .6)
            syn = rng.integers(0, 4); port = rng.choice([80,443,53,22,25,8080])
            plen = rng.normal(600, 180)
        elif lab == "DDoS":
            duration = rng.lognormal(4.2, .55)
            fwd = rng.integers(150, 1200); bwd = rng.integers(0, 20)
            bps = rng.lognormal(11.5, .7); pps = rng.lognormal(6.0, .55)
            syn = rng.integers(20, 180); port = rng.choice([80,443,8080])
            plen = rng.normal(350, 120)
        elif lab == "PortScan":
            duration = rng.lognormal(3.8, .7)
            fwd = rng.integers(1, 12); bwd = rng.integers(0, 3)
            bps = rng.lognormal(5.5, .8); pps = rng.lognormal(1.8, .6)
            syn = rng.integers(1, 12); port = rng.integers(1, 65535)
            plen = rng.normal(90, 35)
        elif lab == "BruteForce":
            duration = rng.lognormal(5.2, .5)
            fwd = rng.integers(20, 140); bwd = rng.integers(10, 100)
            bps = rng.lognormal(7.5, .7); pps = rng.lognormal(3.4, .5)
            syn = rng.integers(0, 12); port = rng.choice([21,22,23,3389,3306])
            plen = rng.normal(250, 90)
        else:
            duration = rng.lognormal(6.0, .9)
            fwd = rng.integers(30, 300); bwd = rng.integers(5, 200)
            bps = rng.lognormal(8.8, 1.0); pps = rng.lognormal(3.8, .8)
            syn = rng.integers(0, 25); port = rng.choice([80,443,53,8080,4444])
            plen = rng.normal(420, 200)
        rows.append([max(1,duration), int(fwd), int(bwd), max(0,bps), max(.1,pps), int(syn), int(port), max(20,plen), lab])
    return pd.DataFrame(rows, columns=REQUIRED_FEATURES + ["demo_label"])

def demo_traffic(n=240, seed=42):
    df = _synthetic_features(n, seed)
    rng = np.random.default_rng(seed+7)
    df.insert(0, "timestamp", pd.date_range(end=pd.Timestamp.now().floor("min"), periods=n, freq="min").astype(str))
    df.insert(1, "source_ip", [f"192.168.{rng.integers(1,10)}.{rng.integers(2,254)}" for _ in range(n)])
    df.insert(2, "destination_ip", [f"10.0.{rng.integers(1,6)}.{rng.integers(2,254)}" for _ in range(n)])
    df.insert(3, "protocol", rng.choice(["TCP","UDP","ICMP"], size=n, p=[.72,.23,.05]))
    severity_map = {"Benign":"Low","DDoS":"High","PortScan":"Medium","BruteForce":"High","Botnet":"High"}
    reason_map = {
        "Benign":"No strong demo-pattern indicators",
        "DDoS":"High packet rate and high traffic volume pattern",
        "PortScan":"Short flows and probing-like port pattern",
        "BruteForce":"Repeated connection pattern to a commonly targeted service",
        "Botnet":"Traffic features resemble the synthetic botnet class"
    }
    df["demo_severity"] = df["demo_label"].map(severity_map)
    df["demo_reason"] = df["demo_label"].map(reason_map)
    return df

def train_demo_model(seed=42):
    df = _synthetic_features(5000, seed)
    X = df[REQUIRED_FEATURES]
    y = df["demo_label"]
    model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=seed, class_weight="balanced")
    model.fit(X, y)
    return model

def _severity(prediction, confidence, row):
    if prediction == "Benign":
        return "Low"
    if prediction in ("DDoS", "BruteForce") or confidence >= .82:
        return "High"
    if prediction in ("PortScan", "Botnet") or confidence >= .62:
        return "Medium"
    return "Low"

def _reason(prediction, row):
    if prediction == "Benign":
        return "Model did not identify a strong match to the learned demo attack patterns"
    if prediction == "DDoS":
        return "Predicted DDoS-like pattern; review packet rate and traffic volume"
    if prediction == "PortScan":
        return "Predicted scan-like pattern; review destination ports and flow duration"
    if prediction == "BruteForce":
        return "Predicted brute-force-like pattern; review repeated connections to target service"
    return "Predicted botnet-like pattern; investigate related flows and endpoints"

def analyze_traffic(df, model):
    df = df.copy()
    missing = [c for c in REQUIRED_FEATURES if c not in df.columns]
    if missing:
        raise ValueError("This CSV is missing required feature columns: " + ", ".join(missing) +
                         ". The built-in sample data includes them. Real dataset columns may need mapping first.")
    for col in REQUIRED_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[REQUIRED_FEATURES] = df[REQUIRED_FEATURES].replace([np.inf,-np.inf], np.nan)
    df = df.dropna(subset=REQUIRED_FEATURES).copy()
    if df.empty:
        raise ValueError("No valid rows remain after checking the required numerical features.")
    X = df[REQUIRED_FEATURES]
    pred = model.predict(X)
    probs = model.predict_proba(X)
    confidence = probs.max(axis=1)
    df["prediction"] = pred
    df["confidence"] = confidence
    df["severity"] = [_severity(p, c, row) for p,c,(_,row) in zip(pred, confidence, df.iterrows())]
    df["reason"] = [_reason(p, row) for p,(_,row) in zip(pred, df.iterrows())]
    for c, default in [("timestamp","Unknown"),("source_ip","Unknown"),("destination_ip","Unknown"),("protocol","Unknown")]:
        if c not in df.columns:
            df[c] = default
    return df.reset_index(drop=True)

def evaluate_demo_model(seed=21):
    df = _synthetic_features(3000, seed)
    X_train, X_test, y_train, y_test = train_test_split(
        df[REQUIRED_FEATURES], df["demo_label"], test_size=.25, random_state=seed, stratify=df["demo_label"]
    )
    model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=seed, class_weight="balanced")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    labels = list(model.classes_)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, pred, average="macro", zero_division=0)
    }
    cm = confusion_matrix(y_test, pred, labels=labels)
    return metrics, cm, labels
