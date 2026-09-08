# Methodology

Maps each of the five project objectives onto the code that implements it and
the experiment that validates it.

---

## Objective 1 — Curate and preprocess encrypted-traffic datasets

**Code:** `src/encids/data/dataset_loaders.py`, `build_dataset.py`,
`features/schema.py`

Three heterogeneous sources are normalised onto one 67-column schema:

| Source | Native form | Flow stats | TLS metadata |
|---|---|---|---|
| ISCX VPN-nonVPN 2016 | pcap | full (extracted) | full (extracted) |
| CIC-Darknet2020 | CICFlowMeter CSV | full | none |
| CTU-13 | Argus netflow | partial | none |
| synthetic | generated | full | full |

Three decisions matter here.

**Identifier columns are excluded from the model.** `src_ip`, `dst_ip`,
`src_port`, `dst_port` and the timestamps are carried through for display and
forensics but never reach a model. Training on them is the standard leakage
failure in this field: the classifier memorises the capture lab's addressing
plan, reports near-perfect offline accuracy, and generalises to nothing.

**Missing is not zero.** CTU-13 carries no packet-size distribution and no
handshake metadata. Those columns come back as `NaN` and are median-imputed by
a transformer fitted on the training split only. Zero-filling them would assert
"this flow had zero packet-size variance", which is a false statement about the
traffic, and tree models would happily split on the fabricated constant.

**CICFlowMeter column names are normalised.** The names drift between
CICFlowMeter releases (`Total Fwd Packet` vs `Total Fwd Packets`, and so on), so
the loader canonicalises before mapping rather than matching literal strings.

---

## Objective 2 — Two-stage detection pipeline

**Code:** `models/supervised.py`, `models/anomaly.py`, `models/fusion.py`

### Stage 1 — supervised (known families)

XGBoost (or Random Forest) over the full feature matrix, trained multi-class on
the family label with balanced sample weights. Binary `P(malicious)` is derived
by summing probability mass over the malicious classes, which keeps the family
prediction and the binary verdict consistent with each other.

### Stage 2 — unsupervised (unseen threats)

Trained on **benign traffic only**. Two implementations:

| | ROC-AUC (held-out) | Recall @ 1% FPR |
|---|---|---|
| IsolationForest | ~0.73 | ~0.03 |
| Autoencoder (48-24-10) | ~0.85 | ~0.23 |

The autoencoder is the default. Reconstruction error captures feature
*interactions* — "regular timing **and** low volume **and** a rare fingerprint"
— that an axis-aligned isolation score cannot represent. The alert threshold is
the 99th percentile of the **benign training** scores; the detector never sees
an attack, not even for calibration.

Stage 2 also gets a restricted feature view: the JA3 hash buckets are excluded,
because bucket 200 is not "larger" than bucket 3 and a distance-based detector
would read that arbitrary numbering as real structure. Stage 1's trees can
carve buckets into arbitrary subsets and are unaffected, so the exclusion is
stage-local.

### Fusion

```
alert = (P_supervised > t₁)  OR  (anomaly_score > t₂)
```

OR, not AND. The two stages have complementary blind spots; requiring agreement
would discard the exact zero-day coverage Stage 2 was added for. The price is a
higher false-positive rate than either stage alone, which is why FPR is a
headline metric and why `models/metrics.py::threshold_sweep` reports the whole
operating-point curve rather than a single number.

Each alert carries a reason code — `known-attack`, `anomalous`, or both — so an
analyst can distinguish "this is Rbot C2" from "this resembles nothing normal".

---

## Objective 3 — TLS handshake metadata alongside flow statistics

**Code:** `features/tls_features.py`

The handshake is sent in the clear before the session goes encrypted, so
`ClientHello` and `ServerHello` are readable without decrypting anything.
The parser works directly on raw TCP payload bytes:

```
JA3  = MD5(TLSVersion, CipherSuites, Extensions, EllipticCurves, ECPointFormats)
JA3S = MD5(TLSVersion, CipherSuite, Extensions)
```

GREASE values (RFC 8701) are stripped before hashing, per the JA3 specification
— they are randomised per connection, so leaving them in would give every
connection a unique fingerprint. Their *presence* is kept as a separate feature,
because mainstream browsers emit them and most malware TLS stacks do not.

21 TLS features are derived in total: fingerprint buckets, fingerprint rarity,
cipher/extension/curve counts, SNI length, SNI Shannon entropy, SNI digit ratio,
ALPN, certificate-chain length, handshake duration.

**Fingerprint rarity is computed on the training corpus and frozen.**
Recomputing it on a live batch would make every fingerprint in a small window
look rare and destroy the signal entirely.

**Validation:** the ablation in `evaluate.py::run_ablation` trains the identical
pipeline with `use_tls=False` and compares. Results land in
`reports/metrics/ablation_flow_vs_flow_tls.csv`.

---

## Objective 4 — Live pipeline with SHAP explanations

**Code:** `live/capture.py`, `live/scorer.py`, `models/explain.py`,
`dashboard/app.py`

```
packets → FlowTable (idle/active timeout) → completed flows
        → same FeaturePipeline as training → both stages → fusion
        → SHAP explanation for alerts only → rolling store → dashboard
```

Three capture sources — live interface (scapy), pcap replay, and a synthetic
stream — all feed the *same* `FlowTable`, so a flow captured live produces a
byte-identical feature vector to the same flow read from a pcap. Training on
features computed one way and serving features computed another is a quiet and
common way to lose accuracy in deployment.

Explanations use SHAP `TreeExplainer` when Stage 1 fired, and per-feature
deviation from the benign profile when only Stage 2 fired — for a
reconstruction detector that deviation is literally *what the model got wrong
about this flow*, and unlike KernelSHAP it is fast enough to run inline.
Explanations are computed for alerts only; explaining every benign flow would
dominate the latency budget and nobody would read them.

Each is rendered to a sentence:

> *Does not match any known traffic profile (4.2σ from normal) — possible
> unseen threat. Driven by: unusually low packet size variability;
> unusually high regularity of packet timing; unusually high rarity of the TLS
> fingerprint.*

---

## Objective 5 — Evaluation against a flow-only baseline

**Code:** `evaluate.py`, `models/metrics.py`

Reported for the held-out split:

1. Binary metrics for Stage 1 alone, Stage 2 alone, and fused — accuracy,
   precision, recall, F1, MCC, **false-positive rate**, ROC-AUC, PR-AUC.
2. Multi-class family classification (accuracy, macro-F1, per-class report).
3. Per-family detection rate — so a strong headline cannot hide a family the
   system never catches. Benign rows in that table *are* the per-class FPR.
4. Ablation: flow-only vs. flow + TLS.
5. Zero-day: Stage 1 retrained without one family, then measure
   `stage2_only_rate` — the share of that family caught by Stage 2 where
   Stage 1 stayed silent.
6. Detection rate stratified by evasion level (synthetic source only).
7. Scoring latency at batch sizes 1 / 32 / 256, in ms per flow and flows/sec.

### On the zero-day result

Withholding a family from Stage 1 gives a *modest* Stage-2-only contribution
(roughly 4–8 percentage points, family-dependent). Stage 1 still catches much of
a withheld family because it has learned other malicious families and
`P(malicious)` aggregates across all of them — supervised models generalise
across attack families better than the "supervised models cannot see zero-days"
framing suggests. That is a finding, and it is reported as measured rather than
reframed into a larger win.

---

---

## A limitation worth stating plainly: flow timeouts vs. slow beacons

A flow is closed after `live.idle_timeout` seconds of silence (15 s by
default). A real C2 implant typically checks in every 30-60 seconds, sometimes
much less often. **Those check-ins therefore land in separate flows**, and the
signal that identifies beaconing - metronomic inter-arrival timing - is
destroyed before the model ever sees it, because each flow contains a single
exchange and has no meaningful IAT distribution at all.

This is not a bug in the implementation; it is a structural limit of
*per-flow* detection, and it applies to most of the flow-statistics literature
this project builds on. Three ways out:

1. **Raise `idle_timeout` above the beacon interval** (e.g. 300 s). Cheap, but
   inflates memory in the flow table and delays every alert by the timeout.
2. **Aggregate across flows per (host, destination) channel** and compute the
   timing statistics over check-in *arrivals* rather than packets. This is
   essentially what CBSeq (reference [4]) does, and it is the right fix.
3. Accept the limit and rely on Stage 2 plus the TLS-fingerprint signal, which
   do not depend on intra-flow timing.

The current implementation takes option 1 (configurable) and documents the
gap; option 2 is the natural next piece of work. The sample-capture generator
carries a comment marking exactly where this bites.

---

## Reproducibility

* Every hyper-parameter lives in `config/config.yaml`.
* `config.set_seed()` seeds Python, NumPy and PyTorch from `project.seed`.
* The preprocessing pipeline is fitted on the training split only and persisted
  alongside the models; evaluation and live scoring load the same artefact.
* The exact test split is written to `data/processed/test_split.parquet`.
* `models/training_metadata.json` records the feature list, class list, source
  filter, thresholds and dataset composition for every run.
* `python -m pytest -q` — 83 tests covering JA3 parsing against hand-built
  handshake bytes, flow aggregation arithmetic, leakage guards, fusion logic,
  and detector persistence.
