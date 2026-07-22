# Space Signal Sorter — Is It a Star or Just Noise?

**What is this, and why does it matter?** A pulsar is what's left after a massive
star dies — its core collapses into a ball the size of a city block but heavier
than the Sun, spinning up to hundreds of times a second, sweeping a beam of radio
waves past Earth like a lighthouse. Radio telescopes surveying the sky record an
enormous number of blips, and almost all of them are junk: mobile phones,
microwave ovens, lightning, radar, plain static. Real pulsars are rare — about
**9 in every 100 signals** in this dataset, and far rarer in a live survey.
Checking each blip by eye takes expert time no research team has enough of.

This app sorts the pile. Each signal is boiled down to **8 summary numbers** —
four describing the shape of the averaged pulse, four describing how the signal
smears across radio frequencies. The smearing is the crucial clue: a signal from
deep space passes through clouds of thin electron gas that slow low frequencies
more than high ones in a very particular pattern local interference can't fake.
A model trained on thousands of hand-sorted signals scores each new one from 0
to 1, and the list arrives sorted most-promising-first: **reading only the top
839 of 3,580 signals the model had never seen surfaces 312 of the 328 real
pulsars.** That turns weeks of eye strain into an afternoon.

It does **not** discover pulsars on its own and does not replace a person — a
human confirms every real detection, and the model errs in both directions. The
app's strictness slider shows exactly what each setting costs and buys.

> **Student project — a machine-learning demo, not a real astronomy pipeline.**

**Live demo:** _URL to be added after deployment._

## The honest model choice (the point of this rebuild)

An earlier version of this project compared *SVM kernels only*, so logistic
regression was never eligible to win — even though that project's own conclusion
was that the boundary is essentially linear. A selection that cannot return the
simplest adequate model is broken. Here, all four candidates competed on equal
footing, on cross-validation only, under a tie-break rule stated **before** any
result was seen:

> Models whose mean CV average-precision is within one fold-standard-deviation of
> the best mean are tied; among tied models, ship the simplest. Complexity order:
> LogisticRegression < SVM-linear < SVM-poly < SVM-rbf. Applied on CV only —
> never test-set selection.

| model | CV average precision (mean ± fold SD) |
|---|---|
| Logistic regression | 0.9249 ± 0.0142 |
| SVM — linear | **0.9252** ± 0.0153 |
| SVM — poly (deg 2) | 0.8981 ± 0.0151 |
| SVM — rbf | 0.8772 ± 0.0154 |

Tied with the best: logistic regression and SVM-linear → **logistic regression
ships**. The test set was then scored once:

| model | test PR-AUC | 95% CI | role |
|---|---|---|---|
| **Logistic regression** | **0.935** | [0.913, 0.956] | **shipped** |
| SVM — linear | 0.933 | [0.911, 0.955] | after-the-fact |
| SVM — poly (deg 2) | 0.903 | [0.874, 0.931] | after-the-fact |
| SVM — rbf | 0.859 | [0.813, 0.900] | after-the-fact |

Shipped ROC-AUC 0.973 [0.960, 0.984]. **Paired** bootstrap (the correct test —
overlapping unpaired CIs settle nothing): LogReg − poly = **+0.032
[+0.019, +0.046]**; LogReg − linear = +0.002 [+0.000, +0.003]; LogReg − rbf =
+0.076 [+0.045, +0.109]. The previously shipped polynomial SVM was tested and
not shipped — a *simplification*, not a performance claim. Bonuses that fall out
of a linear model: the coefficients **are** the model (honest per-signal
explanations), native probabilities, and the saved model shrinks from ~240 KB of
support vectors to **8.5 KB**.

## Two strictness presets (chosen on cross-validation, never on test)

| preset | flag when score ≥ | pulsars caught | flagged really pulsars | signals to read | pulsars missed |
|---|---|---|---|---|---|
| Balanced | 0.848 | 87.8% | 92.3% | 312 | 40 |
| Catch almost everything | 0.176 | 94.5% | 38.3% | 809 | 18 |

The looser preset asks a person to read **497 more signals** and in return misses
**22 fewer pulsars**. A missed pulsar is a lost discovery; a false alarm costs
seconds — so both settings exist and the reader chooses.

## Practice-signal generator (and two rejected designs, with measurements)

Users don't have telescope data, so the app generates realistic practice signals
from the training split's statistics (never the test split).

- **Rejected: per-class multivariate normal.** Measured: ~50%/55% of its draws
  land outside the observed range and would need clipping onto the edges, and it
  destroys distribution shapes (real pulsar `skew_dm` skewness **+9.3** → MVN
  **−0.1**).
- **Rejected: Gaussian copula (built and calibrated first).** Rank copulas
  flatten the Pearson correlation that heavy tails carry; per-pair NORTA
  calibration fixed the noise class but exposed a structural limit — the pulsar
  class's Pearson targets are **jointly infeasible** for any Gaussian copula
  (pairwise-matched correlation matrix has minimum eigenvalue −0.12).
- **Shipped: resampling real training rows with small value-space jitter**
  (σ = 10% of each class's column spread, reflected at the observed min/max).
  Self-test, at 30,000 draws per class: strongest correlations within
  0.022/0.034 of real, every measurement's skewness within 15%, **zero** values
  out of range with no clipping step, and the model scores synthetic pulsars
  0.900 vs real pulsars 0.911.

## What's in the box

```
app.py                 the 4-tab app (sort / one signal / beat-the-model / methods)
features.py            the measurement contract: 8 raw + 4 derived, shared everywhere
copula_generator.py    practice-signal generator + self-tests + rejected-design audits
train_model.py         selection, operating points, persistence, verification gate
test_app.py            upload round-trip + full AppTest of every control
models/pulsar_model.joblib      pipeline, features, presets, ranges, all metrics + CIs
models/signal_generator.joblib  the fitted generator
data/HTRU_2.csv                 the dataset (no header; 17,898 rows)
data/heldout_signals.csv        the 3,580 held-out signals with ids and labels
```

Leakage discipline: the scaler lives inside the pipeline inside CV; the four
derived measurements are row-wise (`log1p(mean_dm)`, a **signed** log for
`skew_dm` — its minimum is −1.98, so a plain log would break — and two
pulse-minus-smearing differences, not ratios, because the denominators cross
zero); the split is stratified 80/20 with `random_state=42`; rare pulsars get
extra weight in the loss instead of synthetic oversampling.

## Run it

```bash
py -3.12 -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
venv\Scripts\python train_model.py     # retrain + verify (~90 s)
venv\Scripts\python test_app.py        # upload round-trip + AppTest gate
venv\Scripts\streamlit run app.py
```

Deploy on Streamlit Community Cloud with **Python 3.12** picked in the
Advanced-settings dropdown; `requirements.txt` carries exact pins.

---

Data: **HTRU2** — R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke,
J. D. Knowles (2016), *Fifty Years of Pulsar Candidate Selection*, MNRAS
459(1):1104–1123. Educational project; not for operational use.
