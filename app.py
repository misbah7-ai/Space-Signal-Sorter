"""
Space Signal Sorter — "Is It a Star or Just Noise?"

A machine-learning demo for the public, built on the HTRU2 survey data
(Lyon et al. 2016). Plain-English rules apply to every visible string:
signals not candidates, strictness not threshold, and every technical term
gets a one-line gloss the first time it appears.

Everything operational (measurement list, strictness presets, metrics, ranges)
is read from models/pulsar_model.joblib — nothing is hardcoded, so the app can
never drift from the model it ships. Scores are always computed on a DataFrame
in bundle["features"] order, never a bare array.
"""
from __future__ import annotations

from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from copula_generator import generate_signals
from features import ENGINEERED, RAW8, build_features, prepare_upload

ROOT = Path(__file__).parent
BUNDLE_PATH = ROOT / "models" / "pulsar_model.joblib"
GEN_PATH = ROOT / "models" / "signal_generator.joblib"
HELDOUT_PATH = ROOT / "data" / "heldout_signals.csv"

st.set_page_config(page_title="Space Signal Sorter", page_icon="✨", layout="wide")


# ------------------------------------------------------------------ loaders
@st.cache_resource
def load_bundle():
    return joblib.load(BUNDLE_PATH)


@st.cache_resource
def load_generator():
    return joblib.load(GEN_PATH)


@st.cache_data
def load_heldout():
    return pd.read_csv(HELDOUT_PATH)


try:
    bundle = load_bundle()
    generator = load_generator()
    heldout = load_heldout()
except FileNotFoundError:
    st.error("The model files aren't here yet. Run `python train_model.py` once, "
             "then reload this page.")
    st.stop()

FEATURES = list(bundle["features"])
FRIENDLY = bundle["friendly_names"]
MEANING = bundle["meanings"]
LABELS = {int(k): v for k, v in bundle["label_names"].items()}
OPS = bundle["operating_points"]
META = bundle["metadata"]
RANGES = bundle["feature_ranges"]
CURVE = META["strictness_curve"]
TOP = META["top_slice"]

ENG_LABEL = {
    "log_mean_dm": "Derived — squashed smearing average",
    "slog_skew_dm": "Derived — squashed smearing lopsidedness",
    "kurt_ip_dm_diff": "Derived — pulse minus smearing peakedness",
    "skew_ip_dm_diff": "Derived — pulse minus smearing lopsidedness",
}
NICE = {**FRIENDLY, **ENG_LABEL}


def score(feat_df: pd.DataFrame) -> np.ndarray:
    """The model's score: 0 = looks like noise, 1 = looks like a pulsar."""
    return bundle["pipeline"].predict_proba(feat_df[FEATURES])[:, 1]


def curve_at(thr: float):
    """Read the pre-computed held-out strictness curve (never recomputed live)."""
    i = int(np.clip(round(thr * 100), 0, 100))
    return CURVE["n_flagged"][i], CURVE["recall"][i], CURVE["precision"][i]


# ------------------------------------------------------------------ header
st.title("✨ Space Signal Sorter — Is It a Star or Just Noise?")
st.markdown(
    "Radio telescopes record huge numbers of blips, and almost all of them are junk — "
    "phones, radar, static. This app uses a model trained on thousands of hand-checked "
    "signals to **sort new ones so the most pulsar-like rise to the top**. On signals it "
    f"had never seen, reading only the top **{TOP['K']}** of {TOP['of_list']:,} caught "
    f"**{TOP['caught']} of the {TOP['total']} real pulsars**."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔭 Sort the signals", "🔎 Look at one signal", "🎮 Can you beat the model?",
     "📖 How it works & how good it is"])


# ------------------------------------------------------------------ shared: current list
def practice_list(n, mix_pct, seed):
    raw = generate_signals(generator, n, mix_pct / 100.0, seed)
    feats = build_features(raw[RAW8])
    out = feats.copy()
    out.insert(0, "signal_id", [f"practice-{i}" for i in range(len(out))])
    out["truth"] = raw["true_class"].values
    out["score"] = score(out)
    return out


def heldout_list():
    h = heldout.copy()
    out = h[["signal_id"] + FEATURES].copy()
    out["truth"] = h["label"].values
    out["score"] = score(out)
    return out


# =================================================================== TAB 1
with tab1:
    with st.expander("What is this, and why does it matter?"):
        st.markdown(
            "**What a pulsar is.** A pulsar is what's left after a massive star dies — its "
            "core collapses into a ball the size of a city block but heavier than the Sun. "
            "It spins incredibly fast, some hundreds of times a second, and sweeps a beam of "
            "radio waves past Earth like a lighthouse. Telescopes on the ground pick up that "
            "steady pulse.\n\n"
            "**The problem.** A sky survey records an enormous number of blips. Almost all are "
            "junk — mobile phones, microwaves, lightning, radar, plain static. Real pulsars "
            "are rare: about 9 in every 100 signals in this dataset, and far rarer in a live "
            "survey. Checking each one by eye takes expert time nobody has.\n\n"
            "**How the model works.** Each signal is boiled down to 8 numbers — four about the "
            "shape of the averaged pulse, four about how the signal smears across radio "
            "frequencies. The smearing is the crucial clue: a signal from deep space passes "
            "through clouds of thin electron gas that slow the low frequencies more than the "
            "high ones, in a very particular pattern local interference doesn't reproduce. "
            "The model learned from thousands of hand-sorted signals which combinations of "
            "those numbers belong to real pulsars, and it returns a score from 0 to 1. "
            "Because the shipped model is a simple linear one, you can also see **which "
            "measurements pushed a signal's score up or down** — no black box.\n\n"
            f"**Why it's useful.** The list arrives sorted most-promising-first: reading only "
            f"the top {TOP['K']} of {TOP['of_list']:,} unseen signals caught {TOP['caught']} "
            f"of the {TOP['total']} real pulsars. That turns weeks of eye strain into an "
            "afternoon.\n\n"
            "**What it does not do.** It doesn't discover pulsars on its own and doesn't "
            "replace a person — a human still confirms every real detection. It makes "
            "mistakes in both directions, and the strictness setting below lets you see "
            "exactly what each choice costs and buys."
        )

    src = st.radio("Where should the signals come from?",
                   ["Practice signals (generated)", "Upload a file",
                    "Real signals the model never saw"],
                   horizontal=True, key="src")

    work, truth_exists = None, False
    if src.startswith("Practice"):
        st.caption("Practice signals are built from the *statistics* of real training "
                   "signals — realistic, but not real observations.")
        g1, g2, g3 = st.columns(3)
        n_gen = g1.slider("How many signals?", 50, 2000, 300, 50, key="gen_n")
        mix = g2.slider("Roughly what share are pulsars? (%)", 1, 50, 10, 1, key="gen_mix")
        batch = g3.number_input("Batch number (change for a fresh batch)", 1, 9999, 1,
                                key="gen_seed")
        work = practice_list(n_gen, mix, int(batch))
        truth_exists = True
    elif src.startswith("Upload"):
        st.caption("The file needs the 8 raw measurement columns: " + ", ".join(RAW8) + ".")
        up = st.file_uploader("Choose a CSV file", type=["csv"], key="uploader")
        if up is not None:
            try:
                raw = pd.read_csv(up)
            except Exception:
                st.error("That file couldn't be read as a CSV. Please check it and try again.")
                raw = None
            if raw is not None:
                prepped, msgs, _ = prepare_upload(raw)
                for m in msgs:
                    (st.warning if prepped is not None else st.error)(m)
                if prepped is not None:
                    work = prepped.copy()
                    work["score"] = score(work)
                    truth_exists = False   # unknown signals: no truth column, ever
        else:
            st.info("Waiting for a file. Uploaded signals are unknowns — the app will "
                    "not pretend to know their true answer.")
    else:
        st.caption(f"These are the {len(heldout):,} signals kept aside while the model "
                   "was learning — it has never seen them.")
        work = heldout_list()
        truth_exists = True

    if work is not None:
        st.markdown("#### How strict should the filter be?")
        pc1, pc2, pc3 = st.columns([1, 1, 3])
        if pc1.button(f"Balanced ({OPS['balanced']['threshold']:.2f})", key="preset_bal",
                      help="Flags fewer signals; catches about "
                           f"{OPS['balanced']['test_recall']:.0%} of real pulsars."):
            st.session_state["strict"] = float(round(OPS["balanced"]["threshold"], 2))
        if pc2.button(f"Catch almost everything ({OPS['high_recall']['threshold']:.2f})",
                      key="preset_high",
                      help="Flags more signals to miss almost nothing "
                           f"(about {OPS['high_recall']['test_recall']:.0%} of pulsars caught)."):
            st.session_state["strict"] = float(round(OPS["high_recall"]["threshold"], 2))
        strict = st.slider("Flag a signal when its score is at least…", 0.0, 1.0,
                           float(round(OPS["balanced"]["threshold"], 2)), 0.01, key="strict")

        nf, rc, pr = curve_at(strict)
        st.info(f"At this setting, on the {TOP['of_list']:,} signals the model never saw, "
                f"you'd look at **{nf}** signals and catch about **{rc:.0%}** of the real "
                f"pulsars; roughly **{pr:.0%}** of what you flagged would really be pulsars. "
                "(Those figures come from the held-out check, computed once — they are not "
                "recalculated on whatever list is on screen.)")

        ranked = work.sort_values("score", ascending=False).reset_index(drop=True)
        ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
        ranked["flag"] = np.where(ranked["score"] >= strict,
                                  "⭐ worth a look", "· probably noise")
        show_cols = ["rank", "signal_id", "score", "flag"]
        if truth_exists:
            ranked["true answer"] = ranked["truth"].map(LABELS)
            show_cols.append("true answer")
        n_flag = int((ranked["score"] >= strict).sum())
        st.markdown(f"**{n_flag} of {len(ranked)} signals flagged on this list.**")

        st.markdown("#### How deep should I read?")
        topk = st.slider("Read only the top…", 1, int(len(ranked)),
                         int(min(len(ranked), max(10, n_flag))), key="topk")
        if truth_exists:
            got = int(ranked["truth"].head(topk).sum())
            tot = int(ranked["truth"].sum())
            share = f"{got / tot:.0%}" if tot else "—"
            st.success(f"Reading the top **{topk}** signals on this list would surface "
                       f"**{got} of its {tot}** real pulsars ({share}).")
        else:
            st.info("These are unknown signals, so nobody can say how many pulsars the top "
                    "slice contains — that's the honest answer for an upload.")

        st.dataframe(ranked[show_cols].round({"score": 3}), hide_index=True,
                     width="stretch", height=380)
        st.download_button("⬇️ Download this ranked list (CSV)",
                           ranked[show_cols].to_csv(index=False).encode(),
                           file_name="sorted_signals.csv", mime="text/csv", key="dl")
        st.session_state["current_list"] = ranked
        st.session_state["current_truth"] = truth_exists


# =================================================================== TAB 2
with tab2:
    st.subheader("Look at one signal")
    have = "current_list" in st.session_state
    pool = st.session_state.get("current_list", heldout_list()
                                .sort_values("score", ascending=False).reset_index(drop=True))
    truth_ok = st.session_state.get("current_truth", True)
    if not have:
        st.caption("Showing the real held-out signals (visit the first tab to sort a "
                   "different list).")

    ids = pool["signal_id"].astype(str).tolist()
    pick = st.selectbox("Pick a signal", ids, key="pick_a")
    row = pool.loc[pool["signal_id"].astype(str) == pick].iloc[0]

    strict2 = float(st.session_state.get("strict", round(OPS["balanced"]["threshold"], 2)))
    c1, c2, c3 = st.columns(3)
    c1.metric("Model's score (0 = noise, 1 = pulsar)", f"{row['score']:.3f}")
    c2.metric("Model's call at the current strictness",
              LABELS[int(row["score"] >= strict2)])
    c3.metric("True answer", LABELS[int(row["truth"])] if truth_ok and "truth" in row
              else "unknown (uploaded signal)")

    st.markdown("**The 8 measurements** (friendly name first; the raw column name in grey)")
    st.dataframe(pd.DataFrame({
        "measurement": [FRIENDLY[c] for c in RAW8],
        "raw name": RAW8,
        "value": [round(float(row[c]), 3) for c in RAW8],
        "what it means": [MEANING[c] for c in RAW8],
    }), hide_index=True, width="stretch")

    # ---- why the model said this (honest local explanation: linear model) ----
    st.markdown("#### Why the model said this")
    st.caption("The shipped model is linear, so each measurement genuinely pushes the score "
               "toward *pulsar* (right) or *noise* (left) by weight × how unusual the value "
               "is. This is the model's real arithmetic, not a guess after the fact.")
    scaler = bundle["pipeline"].named_steps["scaler"]
    clf = bundle["pipeline"].named_steps["clf"]
    x = row[FEATURES].to_numpy(float)
    z = (x - scaler.mean_) / scaler.scale_
    contrib = pd.DataFrame({"measurement": [NICE[c] for c in FEATURES],
                            "push": clf.coef_[0] * z})
    contrib["towards"] = np.where(contrib["push"] > 0, "pulsar", "noise")
    chart = alt.Chart(contrib).mark_bar().encode(
        x=alt.X("push:Q", title="← pushes toward noise   |   pushes toward pulsar →"),
        y=alt.Y("measurement:N", sort="-x", title=None),
        color=alt.Color("towards:N", scale=alt.Scale(domain=["pulsar", "noise"],
                                                     range=["#DD8452", "#4C72B0"]),
                        legend=None),
    ).properties(height=330)
    st.altair_chart(chart, width="stretch")
    tops = contrib.reindex(contrib["push"].abs().sort_values(ascending=False).index).head(3)
    st.markdown("Strongest drivers here: " + "; ".join(
        f"**{r.measurement}** (toward {r.towards})" for r in tops.itertuples()) + ".")

    # ---- where it sits ----
    st.markdown("#### Where it sits among the two populations")
    st.caption("Real held-out signals, split by their true answer; the red line is this "
               "signal. Shown for three of the most informative measurements.")
    coef_rank = pd.Series(np.abs(clf.coef_[0][:len(RAW8)]), index=RAW8)
    strongest = coef_rank.sort_values(ascending=False).index[:3].tolist()
    hh = heldout.copy()
    hh["class"] = hh["label"].map(LABELS)
    panels = []
    for feat in strongest:
        base = alt.Chart(hh).mark_bar(opacity=0.55).encode(
            x=alt.X(feat, bin=alt.Bin(maxbins=40), title=FRIENDLY[feat]),
            y=alt.Y("count()", title="how many signals"),
            color=alt.Color("class:N", scale=alt.Scale(domain=[LABELS[0], LABELS[1]],
                                                       range=["#4C72B0", "#DD8452"]),
                            legend=alt.Legend(title=None)),
        )
        rule = alt.Chart(pd.DataFrame({feat: [float(row[feat])]})).mark_rule(
            color="red", size=2).encode(x=feat)
        panels.append((base + rule).properties(height=170))
    st.altair_chart(alt.vconcat(*panels).resolve_scale(color="shared"),
                    width="stretch")

    # ---- what-if ----
    st.markdown("#### What-if sliders — *exploring, not a real reading*")
    st.caption("Nudge the measurements and watch the score move. Sliders are limited to the "
               "range seen in training; a warning appears if a derived value still leaves it.")
    if st.button("Reset sliders to this signal", key="wi_reset"):
        for c in RAW8:
            st.session_state.pop(f"wi_{c}", None)
    wcols = st.columns(4)
    wvals = {}
    for i, c in enumerate(RAW8):
        lo, hi = RANGES[c]
        wvals[c] = wcols[i % 4].slider(FRIENDLY[c], float(lo), float(hi),
                                       float(np.clip(row[c], lo, hi)), key=f"wi_{c}")
    wframe = build_features(pd.DataFrame([wvals]))
    oob = [NICE[c] for c in ENGINEERED
           if not (RANGES[c][0] <= float(wframe[c].iloc[0]) <= RANGES[c][1])]
    wscore = float(score(wframe)[0])
    st.metric("Score for this imagined signal", f"{wscore:.3f}",
              f"{wscore - float(row['score']):+.3f} vs the real one")
    if oob:
        st.warning("This combination pushes a derived value outside anything seen in "
                   "training (" + ", ".join(oob) + ") — the score here is a guess beyond "
                   "the model's experience, not a reading.")

    # ---- compare two ----
    st.markdown("#### Compare two signals")
    other = st.selectbox("…against", [i for i in ids if i != pick] or ids, key="pick_b")
    row_b = pool.loc[pool["signal_id"].astype(str) == other].iloc[0]
    comp = pd.DataFrame({
        "measurement": [FRIENDLY[c] for c in RAW8] + ["Model's score"],
        str(pick): [round(float(row[c]), 3) for c in RAW8] + [round(float(row["score"]), 3)],
        str(other): [round(float(row_b[c]), 3) for c in RAW8] + [round(float(row_b["score"]), 3)],
    })
    st.dataframe(comp, hide_index=True, width="stretch")


# =================================================================== TAB 3
with tab3:
    st.subheader("Can you beat the model?")
    st.markdown("You'll see a signal's 8 measurements. Decide: **pulsar or just noise?** "
                "Then the truth and the model's score are revealed. These are real held-out "
                "signals, so the truth genuinely exists.")

    if "game" not in st.session_state:
        st.session_state["game"] = {"n": 0, "user_ok": 0, "model_ok": 0,
                                    "idx": None, "revealed": False, "seed": 0}
    game = st.session_state["game"]
    hpool = heldout_list()

    def new_round():
        game["seed"] += 1
        rng = np.random.default_rng(1000 + game["seed"])
        # half pulsars so the game is interesting despite the 9% base rate
        pick_pulsar = rng.random() < 0.5
        cand = hpool[hpool["truth"] == int(pick_pulsar)]
        game["idx"] = int(cand.index[rng.integers(0, len(cand))])
        game["revealed"] = False

    if game["idx"] is None:
        new_round()
    grow = hpool.loc[game["idx"]]

    st.dataframe(pd.DataFrame({
        "measurement": [FRIENDLY[c] for c in RAW8],
        "value": [round(float(grow[c]), 3) for c in RAW8],
        "hint": [MEANING[c] for c in RAW8],
    }), hide_index=True, width="stretch")

    b1, b2, b3 = st.columns(3)
    guess = None
    if not game["revealed"]:
        if b1.button("It's a pulsar! ⭐", key="guess_pulsar"):
            guess = 1
        if b2.button("Just noise 📻", key="guess_noise"):
            guess = 0
    if guess is not None and not game["revealed"]:
        truth = int(grow["truth"])
        model_call = int(grow["score"] >= OPS["balanced"]["threshold"])
        game["n"] += 1
        game["user_ok"] += int(guess == truth)
        game["model_ok"] += int(model_call == truth)
        game["revealed"] = True
        game["last"] = {"truth": truth, "guess": guess, "model_call": model_call,
                        "score": float(grow["score"])}
    if game["revealed"] and game.get("last"):
        L = game["last"]
        st.markdown(f"**Truth: {LABELS[L['truth']]}.** You said *{LABELS[L['guess']]}* — "
                    f"{'✅ right!' if L['guess'] == L['truth'] else '❌ not this time.'} "
                    f"The model scored it **{L['score']:.3f}** and called "
                    f"*{LABELS[L['model_call']]}* — "
                    f"{'✅ right' if L['model_call'] == L['truth'] else '❌ wrong'}.")
        if b3.button("Next signal ➡️", key="next_signal"):
            new_round()
            st.rerun()
    if game["n"]:
        s1, s2, s3 = st.columns(3)
        s1.metric("Rounds played", game["n"])
        s2.metric("Your accuracy", f"{game['user_ok'] / game['n']:.0%}")
        s3.metric("Model's accuracy on the same signals",
                  f"{game['model_ok'] / game['n']:.0%}")


# =================================================================== TAB 4
with tab4:
    st.subheader("How it works and how good it is")
    with st.expander("What is this, and why does it matter? (full story)", expanded=False):
        st.markdown(
            "A pulsar is the collapsed core of a dead star — city-block sized, heavier than "
            "the Sun, spinning up to hundreds of times a second, sweeping a radio beam past "
            "Earth like a lighthouse. Surveys record far more blips than people can check, "
            "and almost all are interference. Each signal is summarised as 8 numbers; the "
            "frequency-smearing ones act as a fingerprint of a genuinely distant origin, "
            "because interstellar electron gas delays low frequencies in a precise pattern "
            "local junk can't fake. The model learned from hand-sorted examples and returns "
            "a 0-to-1 score so the list can be read best-first. It assists sorting; it does "
            "not confirm discoveries — a person always does that."
        )

    st.markdown(f"**The model:** {META['model_name']} (a linear method), inside a pipeline "
                "that first puts all measurements on a common scale. Real pulsars are rare — "
                f"about {META['base_rate']:.0%} of signals — so during learning the rare "
                "class is given extra weight rather than inventing fake examples.")

    st.markdown("#### How the model was chosen (openly)")
    st.markdown(f"> {META['tie_break_rule']}")
    cvt = META["cv_table"]
    st.dataframe(pd.DataFrame([
        {"model": m, "cross-validation score (mean)": f"{cvt[m]['mean']:.4f}",
         "spread across folds (SD)": f"{cvt[m]['sd']:.4f}"}
        for m in cvt]), hide_index=True, width="stretch")
    st.caption("Cross-validation = the model is trained and checked five times on different "
               "slices of the *learning* data, so the choice never peeks at the final exam. "
               f"Tied with the best under the rule: {', '.join(META['tied_with_best'])} → "
               f"the simplest, **{META['model_name']}**, ships.")

    st.markdown("#### The final exam (signals the model never saw while learning)")
    ap = META["test_ap_ci"]
    rows = []
    for m, d in ap.items():
        rows.append({"model": m + (" ← shipped" if m == META["model_name"] else ""),
                     "PR-AUC*": f"{d['ap']:.3f}",
                     "95% confidence interval": f"[{d['lo']:.3f}, {d['hi']:.3f}]",
                     "used for choosing?": "no — reported after the fact"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("*PR-AUC: across all strictness settings at once, how well the model keeps "
               "its flagged pile pure while still catching pulsars. 1.0 is perfect; the "
               f"'always guess pulsar' floor here is {META['base_rate']:.2f}.")
    rr = META["test_roc_auc_ci"]
    st.markdown(f"Shipped model ROC-AUC **{rr['roc']:.3f}** [{rr['lo']:.3f}, {rr['hi']:.3f}] "
                "(ROC-AUC: the chance a random real pulsar outscores a random noise signal).")
    pv = META["paired_vs_chosen"]
    st.markdown("**Is the shipped model really at least as good?** Paired comparison on the "
                "same resampled signals (the fair test):")
    st.dataframe(pd.DataFrame([
        {"comparison": f"{META['model_name']} − {m}", "score gap": f"{d['diff']:+.3f}",
         "95% CI": f"[{d['lo']:+.3f}, {d['hi']:+.3f}]",
         "reading": ("clearly ahead" if d["lo"] > 0 else
                     "clearly behind" if d["hi"] < 0 else "within noise")}
        for m, d in pv.items()]), hide_index=True, width="stretch")

    st.markdown("#### The two strictness presets")
    oprows = []
    for key, nice in [("balanced", "Balanced"), ("high_recall", "Catch almost everything")]:
        o = OPS[key]
        oprows.append({
            "preset": nice, "flag when score ≥": f"{o['threshold']:.3f}",
            "of real pulsars, caught": f"{o['test_recall']:.1%}",
            "of flagged, really pulsars": f"{o['test_precision']:.1%}",
            "signals a person reads": o["test_flagged"],
            "pulsars missed": o["test_missed_pulsars"]})
    st.dataframe(pd.DataFrame(oprows), hide_index=True, width="stretch")
    extra = OPS["high_recall"]["test_flagged"] - OPS["balanced"]["test_flagged"]
    fewer = OPS["balanced"]["test_missed_pulsars"] - OPS["high_recall"]["test_missed_pulsars"]
    st.markdown(f"The stricter preset protects the reader's time; the looser one asks them to "
                f"read **{extra} more signals** and in return misses **{fewer} fewer pulsars**. "
                "A missed pulsar is a lost discovery; a false alarm costs seconds — which is "
                "why both settings exist and you choose.")

    st.markdown("#### Honest notes")
    st.markdown(
        f"- **A simpler model ships on purpose.** A curvier model (a degree-2 SVM) was tried "
        "and not shipped: on cross-validation it wasn't better than the simple linear one, "
        "and the simple one explains itself. That is a *simplification*, not a performance "
        "claim — the paired comparison above is the evidence.\n"
        f"- **Accuracy would be a misleading number here.** With ~{META['base_rate']:.0%} "
        "pulsars, a model that says 'noise' every time is ~91% 'accurate' and totally "
        "useless. That's why the numbers above are catch-rates and purity instead.\n"
        "- **The labels came from people.** The true answers were assigned by human "
        "inspection during the original survey; people occasionally mislabel, so no score "
        "on this data can honestly claim perfection.\n"
        "- **Why there's no live feed.** A pulsar discovery isn't a daily event, and survey "
        "data flows through closed processing pipelines with no public stream of fresh "
        "signals. Practice signals and file uploads are the honest substitute.\n"
        "- **The headline numbers are locked.** Every figure on this tab was computed once "
        "on the held-out signals and stored with the model. The app never recomputes them "
        "on whatever subset is on screen — that would let anyone cherry-pick a flattering "
        "number."
    )

st.divider()
st.caption("Data: HTRU2 — R. J. Lyon et al. (2016), *Fifty Years of Pulsar Candidate "
           "Selection*, MNRAS 459:1104. Student project for learning purposes.")
