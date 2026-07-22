"""
Verification gate — run after train_model.py, before anything ships.

1. Upload round-trip: export the 8 raw columns of the held-out signals to a temp
   file, push it through the exact same prepare_upload() the app uses, and assert
   the scores match the browse-real-signals path to 1e-10.
2. Streamlit AppTest: all four tabs render with zero uncaught exceptions, and
   every interactive control works — strictness slider, both presets, top-K,
   source switching, signal picker, what-if sliders (incl. the out-of-range
   warning), compare, and the guessing game.

Plain asserts; exits non-zero on the first failure.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent


def get(seq, key):
    for w in seq:
        if w.key == key:
            return w
    raise AssertionError(f"widget with key={key!r} not found")


def upload_round_trip():
    from features import FEATURES, RAW8, prepare_upload

    bundle = joblib.load(ROOT / "models" / "pulsar_model.joblib")
    held = pd.read_csv(ROOT / "data" / "heldout_signals.csv")

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "raw_upload.csv"
        held[["signal_id"] + RAW8].to_csv(p, index=False)
        prepped, msgs, n_drop = prepare_upload(pd.read_csv(p))
    assert prepped is not None and n_drop == 0, f"upload prep failed: {msgs}"
    assert list(prepped["signal_id"]) == list(held["signal_id"])

    s_upload = bundle["pipeline"].predict_proba(prepped[FEATURES])[:, 1]
    s_browse = bundle["pipeline"].predict_proba(held[FEATURES])[:, 1]
    d = float(np.max(np.abs(s_upload - s_browse)))
    assert np.allclose(s_upload, s_browse, atol=1e-10), f"upload path drifted: {d:.2e}"
    print(f"[1] upload round-trip: {len(held)} signals, max score diff {d:.2e}  -> PASS")


def app_test():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, f"app raised on first render: {at.exception}"
    assert len(at.tabs) == 4, "expected four tabs"
    print("[2] first render: 4 tabs, no exceptions  -> PASS")

    # --- Tab 1: practice source is the default; exercise its controls
    get(at.slider, "gen_n").set_value(100)
    get(at.slider, "gen_mix").set_value(20)
    at.run(); assert not at.exception
    get(at.button, "preset_high").click(); at.run(); assert not at.exception
    hi = float(get(at.slider, "strict").value)
    get(at.button, "preset_bal").click(); at.run(); assert not at.exception
    bal = float(get(at.slider, "strict").value)
    assert hi < bal, f"presets did not move the strictness slider ({hi} vs {bal})"
    get(at.slider, "strict").set_value(0.30); at.run(); assert not at.exception
    get(at.slider, "topk").set_value(25); at.run(); assert not at.exception
    print(f"[3] tab 1 controls: generator sliders, presets (high {hi:.2f} < balanced "
          f"{bal:.2f}), strictness, top-K  -> PASS")

    # --- switch to the real held-out source
    get(at.radio, "src").set_value("Real signals the model never saw")
    at.run(); assert not at.exception
    print("[4] tab 1 source switch to held-out signals  -> PASS")

    # --- Tab 2: pick a signal, what-if sliders, out-of-range warning, compare
    pick = get(at.selectbox, "pick_a")
    pick.set_value(pick.options[1]); at.run(); assert not at.exception
    # a single extreme stays inside the derived ranges (the rows that carry the
    # extremes also define them) — the warning is for combinations never seen
    # together, so push paired sliders to OPPOSITE extremes
    for key, to_max in [("wi_kurt_ip", True), ("wi_kurt_dm", False),
                        ("wi_skew_ip", True), ("wi_skew_dm", False)]:
        w = get(at.slider, key)
        w.set_value(float(w.max if to_max else w.min))
    at.run(); assert not at.exception
    warned = any("beyond the model's experience" in str(w.value) for w in at.warning)
    assert warned, "opposite-extreme sliders never triggered the out-of-range warning"
    get(at.button, "wi_reset").click(); at.run(); assert not at.exception
    cmp_box = get(at.selectbox, "pick_b")
    cmp_box.set_value(cmp_box.options[-1]); at.run(); assert not at.exception
    print("[5] tab 2: picker, what-if sliders + out-of-range warning, reset, compare -> PASS")

    # --- Tab 3: play two rounds of the game
    get(at.button, "guess_pulsar").click(); at.run(); assert not at.exception
    assert any("Truth:" in str(m.value) for m in at.markdown), "reveal text missing"
    get(at.button, "next_signal").click(); at.run(); assert not at.exception
    get(at.button, "guess_noise").click(); at.run(); assert not at.exception
    metrics = [m for m in at.metric if m.label == "Rounds played"]
    assert metrics and metrics[0].value == "2", "game tally did not reach 2 rounds"
    print("[6] tab 3: two game rounds, reveal + tally  -> PASS")

    # --- Tab 4 renders its tables (it ran in every pass above; assert content once)
    assert any("How the model was chosen" in str(m.value) for m in at.markdown)
    assert not at.exception
    print("[7] tab 4: methods content present, still no exceptions  -> PASS")


if __name__ == "__main__":
    upload_round_trip()
    app_test()
    print("\nALL VERIFICATION CHECKS PASSED")
