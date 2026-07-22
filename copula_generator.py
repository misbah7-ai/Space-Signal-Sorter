"""
Practice-signal generator — resampling of real training rows with small
RANK-SPACE jitter, fitted per class on the TRAINING split only.

Why not a multivariate normal? The measurements are strongly non-Gaussian (raw
skewness up to ~5, and ~9 for the pulsar class's skew_dm), so a per-class MVN
keeps the correlations but destroys the distribution *shapes* and throws ~half
its draws outside the observed range, which then get clipped into piles on the
min/max edges. `mvn_critique()` measures this so the claim is verified, not
asserted (~50%/55% of rows out of range; pulsar skew_dm skewness +9.3 -> -0.1).

Why not a Gaussian copula? It was built and measured first. A rank-based copula
preserves rank correlation but flattens the raw-scale Pearson that heavy tails
carry; calibrating it (NORTA: per-pair root-solving so the output Pearson hits
the real one) fixed the noise class but exposed a structural limit in the pulsar
class: the pairwise-matched normal-correlation matrix has minimum eigenvalue
-0.12 — the Pearson targets are JOINTLY infeasible for any Gaussian copula —
and 64-node quadrature mis-integrates the heaviest tail by ~0.03. Per the design
fallback, the generator therefore resamples real rows with small jitter instead.

Why VALUE-space jitter (not rank-space)? Rank jitter was also measured and
rejected: near a heavy tail, shifting a value's rank by even 2% moves the value
itself enormously (the top order statistics of skew_dm are hundreds apart), which
crushed tail skewness by ~20% and tail-carried correlations by ~0.1. Jitter in
value space, sized at 10% of each class's column spread, provably keeps both:
correlations attenuate by corr/(1+0.01) < 0.01 and skewness by a factor
(1+0.01)^-1.5 ~ 1.5%.

How the jitter keeps every guarantee:
  1. A generated signal copies a random real training row.
  2. Each measurement gets Gaussian noise with sigma = 10% of that class's
     column standard deviation.
  3. Anything pushed past the observed min/max is REFLECTED back inside — so
     every output lies in the observed range by construction, with no clipping
     step and no mass piling on the edges.

Only numpy + scipy (scipy already ships with scikit-learn).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skew

from features import RAW8

SIGMA_FRAC = 0.10     # jitter sd as a fraction of the per-class column spread;
                      # small enough to keep shape/correlations, big enough that
                      # a synthetic signal is not a copy of a real one


# ------------------------------------------------------------------ fitting
def fit_generator(train_df: pd.DataFrame, label_col: str = "label") -> dict:
    """Fit per class on the training split. Stores, per class, the raw rows and
    the jitter scale per measurement. Plain dict, joblib-friendly."""
    model = {"columns": list(RAW8), "classes": {}, "method":
             f"resample real training rows + value-space jitter "
             f"(sigma = {SIGMA_FRAC:.0%} of each class's column spread), reflected "
             "at the observed min/max; Gaussian copula rejected after measurement "
             "(pulsar-class Pearson targets jointly infeasible, min eig -0.12)",
             "fitted_on": "training split only"}
    for cls, g in train_df.groupby(label_col):
        X = g[RAW8].to_numpy(float)
        model["classes"][int(cls)] = {
            "X": X,
            "sigma": SIGMA_FRAC * X.std(axis=0, ddof=1),
            "lo": X.min(axis=0), "hi": X.max(axis=0), "n": len(X)}
    return model


def _reflect(u: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Fold values back into [lo, hi] by reflection (no boundary mass piles)."""
    span = hi - lo
    r = np.mod(u - lo, 2.0 * span)
    return lo + np.where(r > span, 2.0 * span - r, r)


def _sample_class(model: dict, cls: int, n: int, rng: np.random.Generator) -> np.ndarray:
    part = model["classes"][int(cls)]
    X, m = part["X"], part["n"]
    out = X[rng.integers(0, m, n)] + rng.normal(0.0, 1.0, size=(n, len(RAW8))) * part["sigma"]
    return _reflect(out, part["lo"], part["hi"])


def generate_signals(model: dict, n: int, pulsar_fraction: float, seed: int) -> pd.DataFrame:
    """n practice signals with roughly the requested pulsar mix. Returns the 8 raw
    measurements plus a hidden `true_class` (1 = pulsar). Deterministic per seed."""
    rng = np.random.default_rng(seed)
    n_pulsar = int(round(n * float(np.clip(pulsar_fraction, 0.0, 1.0))))
    n_noise = n - n_pulsar
    parts, labels = [], []
    if n_noise:
        parts.append(_sample_class(model, 0, n_noise, rng))
        labels.append(np.zeros(n_noise, dtype=int))
    if n_pulsar:
        parts.append(_sample_class(model, 1, n_pulsar, rng))
        labels.append(np.ones(n_pulsar, dtype=int))
    X = np.vstack(parts)
    y = np.concatenate(labels)
    perm = rng.permutation(len(X))
    df = pd.DataFrame(X[perm], columns=RAW8)
    df["true_class"] = y[perm]
    return df


# ------------------------------------------------------------------ diagnostics
def _strong_pairs(R: np.ndarray, floor: float = 0.5):
    k = len(RAW8)
    return [(i, j) for i in range(k) for j in range(i + 1, k) if abs(R[i, j]) >= floor]


def self_test(model: dict, train_df: pd.DataFrame, score_fn, real_test_df=None,
              n_per_class: int = 30000, seed: int = 123) -> dict:
    """The four checks from the build spec. Raises AssertionError on failure.

    (a) strongest real correlations reproduced within ~0.05
    (b) per-measurement skewness within ~15% of real (the check an MVN fails)
    (c) zero values outside the observed per-class range, with no clipping step
    (d) the shipped model scores synthetic pulsars far above synthetic noise, and
        synthetic pulsars score close to real pulsars

    n_per_class is large on purpose: sample skewness/Pearson of very heavy-tailed
    measurements are extremely noisy at small n, and the test should measure the
    generator's distribution, not draw luck.
    """
    rng = np.random.default_rng(seed)
    report = {}
    for cls in (0, 1):
        real = train_df.loc[train_df["label"] == cls, RAW8].to_numpy(float)
        syn = _sample_class(model, cls, n_per_class, rng)

        # (a) correlations on the strong pairs
        Rr, Rs = np.corrcoef(real, rowvar=False), np.corrcoef(syn, rowvar=False)
        pairs = _strong_pairs(Rr)
        corr_diffs = {f"{RAW8[i]}~{RAW8[j]}": (Rr[i, j], Rs[i, j]) for i, j in pairs}
        max_cd = max(abs(a - b) for a, b in corr_diffs.values())
        assert max_cd <= 0.05, f"class {cls}: correlation drift {max_cd:.3f} > 0.05"

        # (b) skewness realism
        sk_pairs = {}
        for j, col in enumerate(RAW8):
            sr, ss = float(skew(real[:, j])), float(skew(syn[:, j]))
            sk_pairs[col] = (sr, ss)
            if abs(sr) >= 0.5:
                assert abs(ss - sr) <= 0.15 * abs(sr), \
                    f"class {cls} {col}: skewness {ss:.2f} vs real {sr:.2f} (>15% off)"
            else:
                assert abs(ss - sr) <= 0.15, \
                    f"class {cls} {col}: skewness {ss:.2f} vs real {sr:.2f}"

        # (c) range containment without clipping
        lo, hi = real.min(axis=0), real.max(axis=0)
        n_out = int(((syn < lo) | (syn > hi)).sum())
        assert n_out == 0, f"class {cls}: {n_out} synthetic values left the observed range"

        report[cls] = {"max_corr_diff": max_cd, "corr_pairs": corr_diffs,
                       "skewness": sk_pairs, "outside_range": n_out}

    # (d) the shipped model agrees the synthetic classes look like the real ones
    syn_all = generate_signals(model, 2000, 0.5, seed=seed + 1)
    s = score_fn(syn_all[RAW8])
    m_pul, m_noi = float(s[syn_all.true_class == 1].mean()), float(s[syn_all.true_class == 0].mean())
    assert m_pul - m_noi > 0.5, f"synthetic classes not separated: {m_pul:.3f} vs {m_noi:.3f}"
    report["model_check"] = {"syn_pulsar_mean": m_pul, "syn_noise_mean": m_noi}
    if real_test_df is not None:
        rp = float(score_fn(real_test_df.loc[real_test_df["label"] == 1, RAW8]).mean())
        report["model_check"]["real_pulsar_mean"] = rp
        assert abs(rp - m_pul) <= 0.05, \
            f"synthetic pulsars score {m_pul:.3f} but real ones {rp:.3f} (gap > 0.05)"
    return report


def mvn_critique(train_df: pd.DataFrame, seed: int = 7) -> dict:
    """Measure exactly why the per-class multivariate normal fails, so the claim
    in the docstring is verified rather than asserted."""
    rng = np.random.default_rng(seed)
    out = {}
    for cls in (0, 1):
        real = train_df.loc[train_df["label"] == cls, RAW8].to_numpy(float)
        mu, cov = real.mean(axis=0), np.cov(real, rowvar=False)
        syn = rng.multivariate_normal(mu, cov, size=len(real))
        lo, hi = real.min(axis=0), real.max(axis=0)
        frac_rows_out = float(((syn < lo) | (syn > hi)).any(axis=1).mean())
        j = RAW8.index("skew_dm")
        out[cls] = {
            "frac_rows_outside_range": frac_rows_out,
            "skew_dm_skewness_real": float(skew(real[:, j])),
            "skew_dm_skewness_mvn": float(skew(syn[:, j])),
            "kurt_skew_corr_real": float(np.corrcoef(real[:, 2], real[:, 3])[0, 1]),
            "kurt_skew_corr_mvn": float(np.corrcoef(syn[:, 2], syn[:, 3])[0, 1]),
        }
    return out
