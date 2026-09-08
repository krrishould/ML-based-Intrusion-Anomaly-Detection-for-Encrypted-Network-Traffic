# ML-based Intrusion/Anomaly Detection for Encrypted Network Traffic

**BCSE497J — Project I · B.Tech CSE · VIT**

| | |
|---|---|
| **Team** | Prakhar Joshi (23BCB0051) · Rohit Chikhale (23BCE0154) · Krrish Kumar (23BCE0077) |
| **Guide** | Dr. Suresh A, Associate Professor Grade 1 |
| **SDG / TRL** | SDG 9 · TRL 2–3 |
| **Outcome** | Conference paper (Scopus) + open-source prototype |

Detects intrusions and anomalies in **encrypted** network traffic using only
flow-level statistics and TLS handshake metadata — **no payload is ever
decrypted or inspected**.

---

## The idea in one paragraph

Almost all traffic today is TLS-encrypted, so classical payload-inspection IDS
is effectively blind. But you do not need to read a conversation to notice it
behaves strangely. Packet sizes, the rhythm of inter-arrival times, how lopsided
the byte ratio is, and the *cleartext* TLS handshake that precedes every
encrypted session are all observable. This project learns from exactly those
signals: a supervised classifier recognises known attack families, an
unsupervised detector trained only on benign traffic flags things that have
never been labelled, and every alert comes with a SHAP-based reason.

---

## What is actually built

```
   pcap / live interface / netflow
                │
                ▼
   ┌────────────────────────────┐
   │  Feature extraction        │   NFStream or the built-in dpkt engine
   │  · flow statistics (46)    │   packet sizes, IAT, direction, TCP flags
   │  · TLS metadata     (21)   │   JA3 / JA3S, SNI shape, ALPN, GREASE
   └────────────┬───────────────┘
                │  one 67-column feature vector per flow
       ┌────────┴────────┐
       ▼                 ▼
┌──────────────┐   ┌──────────────────┐
│  STAGE 1     │   │  STAGE 2         │
│  supervised  │   │  unsupervised    │
│  RF / XGBoost│   │  AE / IsoForest  │
│  knows named │   │  trained on      │
│  families    │   │  BENIGN ONLY     │
└──────┬───────┘   └────────┬─────────┘
       │  P(malicious)      │  anomaly score
       └────────┬───────────┘
                ▼
        OR-fusion → alert + reason code
                ▼
        SHAP explanation ("why")
                ▼
        Streamlit live dashboard
```

**Why two stages.** Stage 1 is accurate on what it was taught and structurally
blind to everything else. Stage 2 cannot name what it finds but does not need a
label to find it. They fail in different directions, so an OR-fusion covers
both. The cost is a higher false-positive rate, which is why FPR is reported as
a headline metric rather than a footnote.

---

## Quick start

```bash
# 1. environment  (Python 3.10-3.14)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
pip install -e .

# 2. data  (works immediately - the synthetic generator needs no download)
python scripts/download_datasets.py --check
python scripts/prepare_data.py

# 3. train
python scripts/train_model.py

# 4. evaluate  -> reports/metrics/ + reports/figures/
python scripts/evaluate_model.py

# 5. live dashboard
python scripts/run_dashboard.py
```

Everything runs from a clean checkout with no downloads: the synthetic
generator supplies data until the real datasets are in place.

---

## Datasets

| Dataset | What it gives | Availability |
|---|---|---|
| **CTU-13** | 13 real botnet captures (Neris, Rbot, Virut, Menti, Sogou, Murlo, NSIS.ay) as netflow | **Auto-downloaded.** `python scripts/download_datasets.py --dataset ctu13` (~290 MB for 5 scenarios, `--full` for all 13) |
| **ISCX VPN-nonVPN 2016** | Encrypted + VPN-tunnelled traffic pcaps, labelled by application | Manual — behind UNB's registration form. Run `--list` for the URL and where to unpack |
| **CIC-Darknet2020** | Tor / VPN darknet flows, CICFlowMeter CSV | Manual — same portal |
| **Synthetic generator** | 11 traffic families with built-in evasion and class overlap | Always available, no download |

The two CIC datasets sit behind a terms-acceptance form, so the script prints
the exact URLs and target directories instead of bypassing it.

### About the synthetic data

It exists so the pipeline is testable and demonstrable before 28 GB of pcaps
finish downloading — **not** as evidence. It is built to be honest rather than
flattering:

* Flows are generated as **packet sequences**, then pushed through the same
  `FlowAccumulator` used for real pcaps, so every derived statistic is
  internally consistent (a mean that matches its own std).
* Each attack family is deliberately placed near a benign "twin" it is
  genuinely confusable with (`c2_beacon`↔`voip`, `data_exfiltration`↔
  `file_transfer`, `tor_darknet`↔`vpn_tunnel`).
* ~45% of malicious flows are **shaped to mimic benign traffic** (beacon jitter,
  padding removal, fingerprint spoofing) and keep their malicious label, so the
  benchmark has a genuinely hard tail. Detection rate is reported broken down by
  evasion level.

Headline claims in the report must come from CTU-13 and the CIC datasets.
Every artefact records which source produced it.

---

## Commands

```bash
# data
python scripts/download_datasets.py --list          # what is available
python scripts/download_datasets.py --check         # what is on disk
python scripts/prepare_data.py --force              # rebuild the feature table

# training
python scripts/train_model.py                       # dual-signal model
python scripts/train_model.py --no-tls              # flow-only baseline
python scripts/train_model.py --source ctu13        # real malware only
python scripts/train_model.py --hold-out c2_beacon  # zero-day setup
python scripts/train_model.py --stage1 random_forest --stage2 isolation_forest

# evaluation
python scripts/evaluate_model.py --source synthetic --zero-day tor_darknet
python scripts/evaluate_model.py --source ctu13 --skip-ablation

# live
python scripts/live_monitor.py --list-interfaces
python scripts/live_monitor.py --source synthetic --duration 30
python scripts/live_monitor.py --source pcap --pcap data/raw/sample.pcap
python scripts/live_monitor.py --source interface --duration 60    # needs admin
python scripts/run_dashboard.py

# tests
python -m pytest -q
```

**Important:** train the two sources separately. CTU-13 is netflow and carries
*no* TLS handshake metadata, so a model blended across it and a TLS-bearing
source produces an ablation that measures the source mix rather than the
signal. `--source` exists for exactly this reason.

---

## Live capture

Live sniffing needs a packet-capture driver and administrator/root rights:

* **Windows** — install Npcap. The installer is bundled at
  `tools/npcap-1.88.exe`; run it as Administrator and tick *"Install Npcap in
  WinPcap API-compatible Mode"*. Then run the terminal as Administrator.
* **Linux** — `sudo apt install libpcap0.8`, then run with `sudo` or grant
  `CAP_NET_RAW`.
* **macOS** — libpcap ships with the OS; run with `sudo`.

Without it, both `--source pcap` (replay a capture file) and
`--source synthetic` work fully and are enough for the demo.

---

## Layout

```
config/config.yaml            every hyper-parameter, one place
src/encids/
  config.py                   config loading, paths, seeding
  features/
    schema.py                 the canonical 67-feature schema
    tls_features.py           JA3 / JA3S parsing from raw handshake bytes
    flow_stats.py             packet -> bidirectional flow aggregation
    build_features.py         cleaning, imputation, scaling
  data/
    synthetic.py              the generator described above
    dataset_loaders.py        ISCX / CIC-Darknet / CTU-13 loaders
    pcap_to_flows.py          NFStream + native backends
    build_dataset.py          assembles the unified table
  models/
    supervised.py             Stage 1
    anomaly.py                Stage 2 (IsolationForest + PyTorch autoencoder)
    fusion.py                 two-stage decision fusion
    explain.py                SHAP + plain-English alert summaries
    metrics.py                accuracy, F1, FPR, latency, zero-day report
  live/
    capture.py                scapy live capture / pcap replay
    scorer.py                 real-time scoring + rolling alert store
  dashboard/app.py            Streamlit UI
  train.py  evaluate.py       pipelines
scripts/                      CLI entry points
tests/                        pytest suite
docs/                         methodology and design notes
```

---

## Design decisions worth defending

**IPs and ports are never model inputs.** They are kept for display only.
Training on them is the classic encrypted-traffic leakage trap — the model
memorises the capture lab's addressing plan, scores ~0.99 offline and collapses
on real traffic.

**JA3 hashes go through a hashing trick, not one-hot encoding.** A raw
fingerprint hash has unbounded cardinality and would not generalise to
fingerprints unseen at training time. `ja3_rarity` is computed on the training
corpus and applied *unchanged* at inference; recomputing it live would make
every fingerprint look rare and destroy the signal.

**Hash buckets are hidden from Stage 2.** Bucket 200 is not "larger" than
bucket 3, so a reconstruction-based detector would read the arbitrary numbering
as structure. Trees in Stage 1 are unaffected, so the exclusion is stage-local.

**Stage 2 never sees an attack.** Not during training, not for threshold
calibration — the cut-off is a percentile of *benign* training scores.

**Direction is set by whoever sent the first packet**, not by IP sort order.
Getting this wrong silently swaps upload-heavy and download-heavy flows, which
is precisely the signal data-exfiltration detection rests on. There is a
regression test for it.

**Unavailable features are NaN, not zero.** CTU-13 has no packet-size
distribution; a zero would be a claim about the traffic, NaN is an admission
that we do not know, and the imputer handles it explicitly.

---

## Known limitations

* TLS fingerprints can be spoofed or randomised. The synthetic data models this
  explicitly, and `ja3_is_known_browser` is treated as a hint, never a verdict.
* CTU-13 is netflow-only. Of the 65 model features it can populate just 31 —
  no inter-arrival distributions, no TCP flag counts, and **no TLS handshake
  metadata at all**. The dual-signal claim therefore cannot currently be tested
  on real data; doing so needs the ISCX VPN-nonVPN or CIC-Darknet pcaps, which
  is the single most valuable next step for the project.
* The zero-day experiment shows Stage 2 adding a **modest** 4–8 percentage
  points of unique coverage on withheld families. Stage 1 generalises across
  attack families better than expected, which shrinks the headroom. This is
  reported as measured rather than framed as a larger win.
* **OR-fusion is a net loss under severe class imbalance.** On CTU-13 (~1%
  malicious) Stage 1 alone scores F1 0.89; adding Stage 2 raises recall by less
  than a point but drops F1 to 0.60, because a ~1% false-positive rate applied
  to 250k benign flows produces more false alerts than there are true ones.
  The two-stage design earns its keep on balanced data and on unseen families,
  not on a raw imbalanced stream at the default threshold. Raise
  `anomaly.threshold_percentile` or switch `fusion.rule` to `weighted` for that
  case — the operating-point table in the evaluation output is what you tune
  from. Reported as measured rather than hidden.
* **Slow beacons out-live the flow timeout.** A flow closes after 15 s of
  silence, so a 60-second C2 check-in becomes one single-packet flow per
  check-in and its defining timing signal is lost before the model sees it.
  Raising `live.idle_timeout` works around it; the proper fix is per-channel
  cross-flow aggregation. See `docs/METHODOLOGY.md`.

---

## References

1. I. A. Alwhbi, C. C. Zou, R. N. Alharbi, "Encrypted Network Traffic Analysis and Classification Utilizing Machine Learning," *Sensors*, 24(11), art. 3509, 2024.
2. A. Sharma, A. H. Lashkari, "A Survey on Encrypted Network Traffic," *Computer Networks*, 257, art. 110984, 2024.
3. E. Polo-Peyres et al., "Detecting Malware in Encrypted Network Traffic Using Machine Learning and TLS Fingerprints," UCAmI, LNNS vol. 1819, Springer, 2026.
4. S. Cui et al., "CBSeq: A Channel-level Behavior Sequence for Encrypted Malware Traffic Detection," arXiv:2307.09002, 2023.
5. A. M. Elshewey, A. M. Osman, "Enhancing Encrypted HTTPS Traffic Classification Based on Stacked Deep Ensembles Models," *Scientific Reports*, 15, art. 35230, 2025.
6. S. García et al., "An empirical comparison of botnet detection methods," *Computers & Security*, 45, 2014. (CTU-13)
7. J. Althouse et al., "TLS Fingerprinting with JA3 and JA3S," Salesforce Engineering, 2019.
