"""
The measurement contract — shared by training, the app, and the practice-signal
generator, so the three can never drift apart.

Every signal is described by 8 raw numbers from the telescope's software:
four about the shape of the averaged pulse ("_ip") and four about how the signal
smears across radio frequencies ("_dm"). Training adds 4 derived numbers, all
row-wise (each row computed from itself only), so they cannot leak information
between signals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# the 8 raw columns, in HTRU_2.csv order
RAW8 = ["mean_ip", "std_ip", "kurt_ip", "skew_ip",
        "mean_dm", "std_dm", "kurt_dm", "skew_dm"]

ENGINEERED = ["log_mean_dm", "slog_skew_dm", "kurt_ip_dm_diff", "skew_ip_dm_diff"]

# exact fit order: 8 raw + 4 derived
FEATURES = RAW8 + ENGINEERED

LABEL_COL = "label"
CSV_COLS = RAW8 + [LABEL_COL]

# ------------------------------------------------------------------ friendly names
# Shown to users; the raw column name stays visible as a secondary label.
FRIENDLY = {
    "mean_ip": "Pulse shape — average brightness",
    "std_ip": "Pulse shape — spread",
    "kurt_ip": "Pulse shape — peakedness",
    "skew_ip": "Pulse shape — lopsidedness",
    "mean_dm": "Frequency smearing — average",
    "std_dm": "Frequency smearing — spread",
    "kurt_dm": "Frequency smearing — peakedness",
    "skew_dm": "Frequency smearing — lopsidedness",
}

# one plain sentence per measurement, for Tab 2
MEANING = {
    "mean_ip": "How bright the averaged pulse is overall.",
    "std_ip": "How much the pulse's brightness varies around its average.",
    "kurt_ip": "Whether the pulse has one sharp spike (high) or is flat and mushy (low).",
    "skew_ip": "Whether the pulse leans to one side instead of being symmetric.",
    "mean_dm": "The average strength of the tell-tale smear across radio frequencies.",
    "std_dm": "How much that smear varies.",
    "kurt_dm": "Whether the smear is concentrated in a sharp peak.",
    "skew_dm": "Whether the smear leans heavily to one side — real pulsars often score high here.",
}

LABEL_NAMES = {0: "background noise / interference", 1: "pulsar"}


def signed_log1p(s):
    """Log-squash that also works for negatives.

    skew_dm dips below -1 (min ~ -1.98 in HTRU2), so a plain log1p would produce
    NaN there. sign(x) * log1p(|x|) squashes the huge tail while keeping order.
    """
    return np.sign(s) * np.log1p(np.abs(s))


def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """8 raw columns in -> the 12-column frame the model expects, in fit order.

    Row-wise only (no cross-row statistics), so it is identical whether it runs
    on the training table, a user upload, or a generated practice signal.
    """
    d = df_raw.copy()
    d["log_mean_dm"] = np.log1p(d["mean_dm"])              # mean_dm min ~0.21 -> safe
    d["slog_skew_dm"] = signed_log1p(d["skew_dm"])         # skew_dm min ~-1.98 -> signed log
    d["kurt_ip_dm_diff"] = d["kurt_ip"] - d["kurt_dm"]     # differences, not ratios:
    d["skew_ip_dm_diff"] = d["skew_ip"] - d["skew_dm"]     # the denominators cross zero
    return d[FEATURES]


def load_dataset(csv_path) -> pd.DataFrame:
    """Raw HTRU_2.csv (no header) -> named columns + integer label."""
    df = pd.read_csv(csv_path, header=None, names=CSV_COLS)
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df


def prepare_upload(raw: pd.DataFrame):
    """Validate + engineer an uploaded table of signals.

    Returns (frame_or_None, messages, n_dropped). The frame has signal_id +
    the 12 measurements; no truth column — uploaded signals are unknowns.
    """
    msgs = []
    missing = [c for c in RAW8 if c not in raw.columns]
    if missing:
        msgs.append(
            "The file is missing these required columns: **" + ", ".join(missing) + "**. "
            "It needs all 8 raw measurements: " + ", ".join(RAW8) + "."
        )
        return None, msgs, 0

    coerced = raw.copy()
    coerced[RAW8] = coerced[RAW8].apply(pd.to_numeric, errors="coerce")
    eng = build_features(coerced[RAW8])
    finite = np.isfinite(eng.to_numpy()).all(axis=1)
    n_dropped = int((~finite).sum())
    eng = eng.loc[finite].reset_index(drop=True)

    if "signal_id" in raw.columns:
        sid = raw.loc[finite, "signal_id"].reset_index(drop=True)
    else:
        sid = pd.Series(np.arange(len(eng)), name="signal_id")

    if len(eng) == 0:
        msgs.append("Every row in the file produced an unusable value after the "
                    "derived measurements were computed, so there is nothing to score.")
        return None, msgs, n_dropped

    out = eng.copy()
    out.insert(0, "signal_id", sid.values)
    if n_dropped:
        msgs.append(
            f"Skipped **{n_dropped}** row(s) whose numbers could not be used "
            "(missing/non-numeric values, or a physically impossible negative "
            "frequency-smearing average). They were never shown to the model."
        )
    return out, msgs, n_dropped
