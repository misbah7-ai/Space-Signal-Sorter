"""
Train + select the pulsar signal-sorter model, honestly.

The defect this rebuild corrects: the previous search compared SVM kernels only,
so logistic regression was never eligible to win — even though that project's own
conclusion was that the boundary is essentially linear. Here all four candidates
compete on equal footing, on cross-validation only, and the tie-break rule is
stated BEFORE any result is seen:

    TIE-BREAK RULE (fixed in advance): models whose mean CV average-precision is
    within one fold-standard-deviation of the best mean are considered tied with
    the best; among tied models, ship the simplest / most interpretable one.
    Complexity order: LogisticRegression < SVM-linear < SVM-poly < SVM-rbf.
    This is a parsimony rule applied on CV — never test-set selection.

The test set is touched exactly once, after the choice is made. Every reported
score carries a bootstrap confidence interval, and the chosen-vs-runner-up gap
uses a PAIRED bootstrap (overlapping unpaired CIs settle nothing).
"""
from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from copula_generator import fit_generator, generate_signals, mvn_critique, self_test
from features import (FEATURES, FRIENDLY, LABEL_NAMES, MEANING, RAW8,
                      build_features, load_dataset)

ROOT = Path(__file__).parent
RANDOM_STATE = 42
RNG_BOOT = 42
N_BOOT = 2000

# independent-replication reference values; mismatches > 0.01 are FLAGGED, not fatal
REF_TEST_AP = {"LogisticRegression": 0.935, "SVM-linear": 0.933,
               "SVM-poly": 0.903, "SVM-rbf": 0.859}
REF_LOGREG_ROC = 0.973
REF_PAIRED_LOGREG_MINUS_POLY = (0.032, 0.020, 0.046)

COMPLEXITY = ["LogisticRegression", "SVM-linear", "SVM-poly", "SVM-rbf"]  # simplest first


def make_candidates():
    """All four, equal footing: Pipeline(StandardScaler -> model), balanced weights."""
    return {
        "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=5000,
                                                 random_state=RANDOM_STATE),
        "SVM-linear": SVC(kernel="linear", class_weight="balanced", random_state=RANDOM_STATE),
        "SVM-poly": SVC(kernel="poly", degree=2, C=1.0, class_weight="balanced",
                        random_state=RANDOM_STATE),
        "SVM-rbf": SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE),
    }


def pipe(est):
    return Pipeline([("scaler", StandardScaler()), ("clf", est)])


def rank_scores(model, X):
    """Ranking score for metrics: probability if available, else decision_function."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def boot_ci(y, s, metric_fn, n_boot=N_BOOT, seed=RNG_BOOT):
    """Percentile bootstrap CI. Skips single-class resamples."""
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    n, vals = len(y), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y[idx]
        if yt.min() == yt.max():
            continue
        vals.append(metric_fn(yt, s[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(metric_fn(y, s)), float(lo), float(hi)


def paired_boot(y, s_a, s_b, n_boot=N_BOOT, seed=RNG_BOOT):
    """Paired bootstrap of AP(a) - AP(b): the same resampled rows score both models."""
    y, s_a, s_b = np.asarray(y), np.asarray(s_a), np.asarray(s_b)
    rng = np.random.default_rng(seed)
    n, diffs = len(y), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y[idx]
        if yt.min() == yt.max():
            continue
        diffs.append(average_precision_score(yt, s_a[idx])
                     - average_precision_score(yt, s_b[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    point = float(average_precision_score(y, s_a) - average_precision_score(y, s_b))
    return point, float(lo), float(hi)


def flag(name, got, exp, tol=0.01):
    d = abs(got - exp)
    mark = "OK" if d <= tol else "** MISMATCH **"
    print(f"    {name:34s} got {got:+.3f}  ref {exp:+.3f}  |d|={d:.3f}  {mark}")
    return d <= tol


def main():
    # ---------------- Phase 2: data ----------------
    df = load_dataset(ROOT / "data" / "HTRU_2.csv")
    n_pulsar = int(df["label"].sum())
    print(f"rows={len(df)}  pulsars={n_pulsar}  rate={df['label'].mean():.4f} "
          f"(expected 17898 / 1639 / 0.0916)")
    assert len(df) == 17898 and n_pulsar == 1639

    raw_skews = {c: float(skew(df[c])) for c in RAW8}
    print(f"raw skewness by measurement (max |.|={max(abs(v) for v in raw_skews.values()):.2f}):")
    print("  " + "  ".join(f"{c}={v:+.2f}" for c, v in raw_skews.items()))
    print(f"skew_dm min = {df.skew_dm.min():.3f} -> plain log1p would NaN; signed log REQUIRED")
    assert df.skew_dm.min() < -1

    X = build_features(df[RAW8])
    assert list(X.columns) == FEATURES and np.isfinite(X.to_numpy()).all()
    y = df["label"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    print(f"split: train={len(X_train)}  test={len(X_test)}  "
          f"test pulsars={int(y_test.sum())} (expected 14318 / 3580 / 328)")
    assert len(X_train) == 14318 and len(X_test) == 3580 and int(y_test.sum()) == 328

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ---------------- Phase 3: selection on CV only ----------------
    print("\n===== CV COMPARISON (StratifiedKFold-5, average precision) =====")
    cands = make_candidates()
    cv_table = {}
    for name in COMPLEXITY:
        scores = cross_val_score(pipe(cands[name]), X_train, y_train,
                                 scoring="average_precision", cv=cv, n_jobs=-1)
        cv_table[name] = {"mean": float(scores.mean()), "sd": float(scores.std(ddof=1)),
                          "folds": [float(s) for s in scores]}
        print(f"  {name:20s} CV AP = {scores.mean():.4f} +/- {scores.std(ddof=1):.4f} "
              f"(folds: {', '.join(f'{s:.4f}' for s in scores)})")

    best_name = max(cv_table, key=lambda k: cv_table[k]["mean"])
    best_mean, best_sd = cv_table[best_name]["mean"], cv_table[best_name]["sd"]
    tied = [n for n in COMPLEXITY if best_mean - cv_table[n]["mean"] <= best_sd]
    chosen = tied[0]                       # simplest among the tied, by the pre-stated rule
    print(f"\n  best mean: {best_name} ({best_mean:.4f}, fold SD {best_sd:.4f})")
    print(f"  tied with best (within one fold-SD): {tied}")
    print(f"  PARSIMONY RULE selects -> {chosen}")

    # fit everything on the full training split (test still untouched for selection)
    fitted = {n: pipe(cands[n]).fit(X_train, y_train) for n in COMPLEXITY}
    model = fitted[chosen]
    if not hasattr(model, "predict_proba"):    # contingency; LogisticRegression has it natively
        est = cands[chosen].set_params(probability=True)
        model = pipe(est).fit(X_train, y_train)
        fitted[chosen] = model

    # ---------------- Phase 3.4: the single test-set look ----------------
    print("\n===== TEST SET (looked at once, AFTER the choice; others reported "
          "after-the-fact for transparency) =====")
    test_scores = {n: rank_scores(fitted[n], X_test) for n in COMPLEXITY}
    test_ap_ci, ref_ok = {}, True
    for n in COMPLEXITY:
        ap, lo, hi = boot_ci(y_test, test_scores[n], average_precision_score)
        test_ap_ci[n] = {"ap": ap, "lo": lo, "hi": hi}
        tag = "  <- CHOSEN" if n == chosen else "  (after-the-fact)"
        print(f"  {n:20s} test AP = {ap:.4f}  95% CI [{lo:.4f}, {hi:.4f}]{tag}")
        ref_ok &= flag(f"ref check {n}", ap, REF_TEST_AP[n])
    roc, roc_lo, roc_hi = boot_ci(y_test, test_scores[chosen], roc_auc_score)
    print(f"  {chosen} test ROC-AUC = {roc:.4f}  95% CI [{roc_lo:.4f}, {roc_hi:.4f}]")
    if chosen == "LogisticRegression":
        ref_ok &= flag("ref check LogReg ROC-AUC", roc, REF_LOGREG_ROC)

    # paired bootstrap: chosen vs every other (the correct test for "is it really better")
    print("\n  paired bootstrap of AP differences (chosen minus other):")
    paired = {}
    for n in COMPLEXITY:
        if n == chosen:
            continue
        d, lo, hi = paired_boot(y_test, test_scores[chosen], test_scores[n])
        paired[n] = {"diff": d, "lo": lo, "hi": hi}
        verdict = "chosen clearly ahead" if lo > 0 else (
            "other clearly ahead" if hi < 0 else "within noise")
        print(f"    {chosen} - {n:14s} = {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  ({verdict})")
    if chosen == "LogisticRegression" and "SVM-poly" in paired:
        p = paired["SVM-poly"]
        rd, rl, rh = REF_PAIRED_LOGREG_MINUS_POLY
        flag("ref paired LogReg-poly diff", p["diff"], rd)
        flag("ref paired CI low", p["lo"], rl)
        flag("ref paired CI high", p["hi"], rh)

    # ---------------- Phase 5: two operating points, chosen on CV only ----------------
    print("\n===== OPERATING POINTS (chosen on cross-validated scores, never on test) =====")
    oof = cross_val_predict(pipe(make_candidates()[chosen]), X_train, y_train,
                            cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    prec_c, rec_c, thr_c = precision_recall_curve(y_train, oof)

    def pick(target_recall):
        ok = np.where(rec_c[:-1] >= target_recall)[0]
        i = ok[-1]                                   # highest threshold meeting the target
        return {"threshold": float(thr_c[i]), "target_recall": target_recall,
                "cv_recall": float(rec_c[i]), "cv_precision": float(prec_c[i])}

    ops = {"balanced": pick(0.85), "high_recall": pick(0.95)}
    test_p = test_scores[chosen]
    for key, op in ops.items():
        pred = (test_p >= op["threshold"]).astype(int)
        op["test_recall"] = float(recall_score(y_test, pred))
        op["test_precision"] = float(precision_score(y_test, pred, zero_division=0))
        op["test_flagged"] = int(pred.sum())
        op["test_missed_pulsars"] = int(((pred == 0) & (y_test == 1)).sum())
        print(f"  {key:12s} thr={op['threshold']:.3f}  CV rec/prec "
              f"{op['cv_recall']:.3f}/{op['cv_precision']:.3f}  ->  test rec/prec "
              f"{op['test_recall']:.3f}/{op['test_precision']:.3f}  "
              f"flags {op['test_flagged']} signals, misses {op['test_missed_pulsars']} pulsars")
    extra = ops["high_recall"]["test_flagged"] - ops["balanced"]["test_flagged"]
    print(f"  catch-almost-everything asks a person to read {extra} more signals "
          f"and misses {ops['balanced']['test_missed_pulsars'] - ops['high_recall']['test_missed_pulsars']} fewer pulsars")

    # precomputed strictness curve + top-slice story (the ONLY numbers the app displays)
    grid = np.round(np.linspace(0.0, 1.0, 101), 2)
    curve = {"threshold": grid.tolist(),
             "n_flagged": [int((test_p >= t).sum()) for t in grid],
             "recall": [float(recall_score(y_test, (test_p >= t).astype(int)))
                        for t in grid],
             "precision": [float(precision_score(y_test, (test_p >= t).astype(int),
                                                 zero_division=1)) for t in grid]}
    order = np.argsort(-test_p)
    cum = np.cumsum(y_test[order])
    total_pul = int(y_test.sum())
    K95 = int(np.searchsorted(cum, int(np.ceil(0.95 * total_pul))) + 1)
    top_slice = {"K": K95, "caught": int(cum[K95 - 1]), "total": total_pul,
                 "of_list": int(len(y_test))}
    print(f"\n  top-slice: reading the top {K95} of {len(y_test)} held-out signals surfaces "
          f"{top_slice['caught']} of the {total_pul} real pulsars")

    # ---------------- Phase 4: generator ----------------
    print("\n===== PRACTICE-SIGNAL GENERATOR =====")
    train_raw = df.loc[X_train.index, RAW8 + ["label"]]
    crit = mvn_critique(train_raw)
    print("  why the old multivariate-normal generator is wrong (measured):")
    for cls in (0, 1):
        c = crit[cls]
        print(f"    class {cls}: {c['frac_rows_outside_range']:.1%} of MVN rows leave the "
              f"observed range; skew_dm skewness real {c['skew_dm_skewness_real']:+.2f} vs "
              f"MVN {c['skew_dm_skewness_mvn']:+.2f}; kurt~skew corr kept "
              f"({c['kurt_skew_corr_real']:+.3f} -> {c['kurt_skew_corr_mvn']:+.3f})")

    cop = fit_generator(train_raw)
    print(f"  method: {cop['method']}")

    def score_raw(raw_frame):
        return model.predict_proba(build_features(raw_frame)[FEATURES])[:, 1]

    test_raw = df.loc[X_test.index, RAW8 + ["label"]]
    rep = self_test(cop, train_raw, score_raw, real_test_df=test_raw)
    mc = rep["model_check"]
    print("  generator self-test PASSED:")
    for cls in (0, 1):
        print(f"    class {cls}: max corr drift {rep[cls]['max_corr_diff']:.3f} (<=0.05), "
              f"skewness within 15%, {rep[cls]['outside_range']} values out of range")
    print(f"    scores: synthetic pulsars {mc['syn_pulsar_mean']:.3f} vs synthetic noise "
          f"{mc['syn_noise_mean']:.3f}; real pulsars {mc['real_pulsar_mean']:.3f} "
          f"(ref ~0.890 vs ~0.904)")
    joblib.dump(cop, ROOT / "models" / "signal_generator.joblib")

    # ---------------- Phase 7: persistence ----------------
    clf = model.named_steps["clf"]
    feature_ranges = {c: [float(X_train[c].min()), float(X_train[c].max())] for c in FEATURES}
    heldout = X_test.copy()
    heldout.insert(0, "signal_id", X_test.index)
    heldout["label"] = y_test
    heldout.to_csv(ROOT / "data" / "heldout_signals.csv", index=False)

    bundle = {
        "pipeline": model,
        "features": FEATURES,
        "raw_columns": RAW8,
        "operating_points": ops,
        "default_operating_point": "balanced",
        "feature_ranges": feature_ranges,
        "friendly_names": FRIENDLY,
        "meanings": MEANING,
        "label_names": LABEL_NAMES,
        "metadata": {
            "model_name": chosen,
            "hyperparameters": {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                                    else str(v)) for k, v in clf.get_params().items()},
            "python": platform.python_version(),
            "sklearn": __import__("sklearn").__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__,
            "random_state": RANDOM_STATE,
            "split": "stratified 80/20, random_state=42",
            "n_train": int(len(X_train)), "n_test": int(len(X_test)),
            "base_rate": float(df.label.mean()),
            "test_pulsars": int(y_test.sum()),
            "cv_table": cv_table,
            "tie_break_rule": "within one fold-SD of best mean -> tied; ship simplest "
                              "(LogReg < SVM-linear < SVM-poly < SVM-rbf). Applied on CV only.",
            "tied_with_best": tied,
            "test_ap_ci": test_ap_ci,
            "test_roc_auc_ci": {"roc": roc, "lo": roc_lo, "hi": roc_hi},
            "paired_vs_chosen": paired,
            "strictness_curve": curve,
            "top_slice": top_slice,
            "raw_skewness": raw_skews,
            "mvn_critique": crit,
            "generator_check": {"syn_pulsar_mean": mc["syn_pulsar_mean"],
                                "syn_noise_mean": mc["syn_noise_mean"],
                                "real_pulsar_mean": mc["real_pulsar_mean"]},
            "reference_checks_all_ok": bool(ref_ok),
        },
    }
    out = ROOT / "models" / "pulsar_model.joblib"
    joblib.dump(bundle, out)
    (ROOT / "models" / "metrics.json").write_text(
        json.dumps(bundle["metadata"], indent=2, default=str))
    print(f"\nsaved -> {out} ({out.stat().st_size / 1024:.1f} KB)  "
          f"+ signal_generator.joblib "
          f"({(ROOT / 'models' / 'signal_generator.joblib').stat().st_size / 1024:.1f} KB)  "
          f"+ data/heldout_signals.csv")

    # ---------------- verification gate ----------------
    rb = joblib.load(out)
    assert rb["features"] == FEATURES
    p1 = rb["pipeline"].predict_proba(X_test[FEATURES])[:, 1]
    assert np.allclose(p1, test_p, atol=1e-10), "round-trip scores drifted"
    hh = pd.read_csv(ROOT / "data" / "heldout_signals.csv")
    rebuilt = build_features(hh[RAW8])
    assert np.allclose(rebuilt.to_numpy(), hh[FEATURES].to_numpy(), atol=1e-10), \
        "build_features no longer reproduces the saved engineered columns"
    p2 = rb["pipeline"].predict_proba(rebuilt[FEATURES])[:, 1]
    assert np.allclose(p2, test_p, atol=1e-10), "upload-style path drifted from training path"
    print("verification: round-trip scores, build_features reproduction, and the "
          "raw->engineered->score path all reproduce to 1e-10  -> PASS")


if __name__ == "__main__":
    main()
