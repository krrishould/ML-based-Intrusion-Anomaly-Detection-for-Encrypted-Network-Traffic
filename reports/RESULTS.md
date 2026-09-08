# Results

ML-based Intrusion/Anomaly Detection for Encrypted Network Traffic — BCSE497J

_Generated 2026-09-09 03:34 from evaluation_cic_darknet2020.json, evaluation_ctu13.json, evaluation_iscx_vpn2016.json, evaluation_synthetic.json._

> Results from the `synthetic` source are development figures and are labelled as such; the `ctu13` source is real captured botnet traffic. CTU-13 is netflow-only and carries no TLS handshake metadata, so the dual-signal ablation is reported for `synthetic` only.

---

## Source: `cic_darknet2020`

### Detection performance (held-out test split)

| Detector | Accuracy | Precision | Recall | F1-score | False-positive rate | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Stage 1 (supervised) | 0.9754 | 0.9166 | 0.9430 | 0.9296 | 0.0179 | 0.9962 | 0.9835 |
| Stage 2 (unsupervised) | 0.8309 | 0.5774 | 0.0708 | 0.1262 | 0.0108 | 0.6642 | 0.3191 |
| **Fused (both)** | 0.9670 | 0.8747 | 0.9439 | 0.9080 | 0.0282 | 0.9930 | 0.9645 |

> **OR-fusion costs more than it gains on this source.** Stage 1 alone scores F1 0.9296; fusing with Stage 2 drops it to 0.9080 (-0.0216) while adding +0.0009 recall.
>
> The classes are only about 5:1 here, so this is not an imbalance artefact: Stage 2 simply found nothing Stage 1 had missed (+0.0009 recall) while still contributing 477 false alerts. Stage 2 earns its place on *unseen* families - see the zero-day experiment - not on families Stage 1 already covers.
>
> Practical response: raise `anomaly.threshold_percentile` (see the operating-point table below), or switch `fusion.rule` to `weighted` so Stage 2 escalates rather than alerts on its own. Reported as measured.

**Attack-family classification (Stage 1):** accuracy 0.8380, macro-F1 0.7170, weighted-F1 0.8410

### Per-family alert rate

| Traffic family | Alert rate | Reading |
|---|---|---|
| browsing | 0.002 | detection rate |
| p2p | 0.004 | detection rate |
| file_transfer | 0.029 | false-positive rate |
| audio_streaming | 0.057 | detection rate |
| voip | 0.060 | false-positive rate |
| video_streaming | 0.090 | detection rate |
| chat | 0.094 | detection rate |
| email | 0.097 | detection rate |
| vpn_voip | 0.653 | detection rate |
| tor_video_streaming | 0.725 | detection rate |
| tor_voip | 0.898 | detection rate |
| tor_chat | 0.923 | detection rate |
| vpn_video_streaming | 0.924 | detection rate |
| tor_browsing | 0.925 | detection rate |
| vpn_email | 0.936 | detection rate |
| vpn_audio_streaming | 0.945 | detection rate |
| vpn_file_transfer | 0.982 | detection rate |
| vpn_chat | 0.984 | detection rate |
| tor_audio_streaming | 1.000 | detection rate |
| tor_email | 1.000 | detection rate |
| tor_file_transfer | 1.000 | detection rate |
| tor_p2p | 1.000 | detection rate |

_Benign families in this table are the per-class false-positive rate._

### Stage-2 operating points

| Percentile | Threshold | Recall | Precision | FPR |
|---|---|---|---|---|
| p90 | 0.0262 | 0.288 | 0.375 | 0.1000 |
| p95 | 0.0499 | 0.199 | 0.454 | 0.0500 |
| p97.5 | 0.0920 | 0.100 | 0.453 | 0.0250 |
| p99 | 0.1552 | 0.067 | 0.583 | 0.0100 |
| p99.5 | 0.2524 | 0.046 | 0.654 | 0.0050 |
| p99.9 | 0.6688 | 0.023 | 0.825 | 0.0010 |

_The threshold is a deployment choice; this is the trade-off an operator picks from._

### Scoring latency

| Batch size | Total (ms) | Per flow (ms) | Flows/sec |
|---|---|---|---|
| 1 | 51.96 | 51.9613 | 19 |
| 32 | 72.82 | 2.2757 | 439 |
| 256 | 86.81 | 0.3391 | 2,949 |

_Batch size 1 is dominated by fixed per-call overhead; the batched figures reflect real throughput._

### Example alert explanations

* **vpn_audio_streaming** — Matches the known family 'vpn_audio_streaming' (p=0.98). Driven by: unusually low fwd init win bytes (229.0); unusually low bwd init win bytes (29.0); unusually high bwd packets per s (12.4).
* **vpn_chat** — Matches the known family 'vpn_chat' (p=1.00). Driven by: unusually low fwd init win bytes (0.0); unusually low bwd init win bytes (0.0); unusually high flow iat min (124,029.0).
* **vpn_chat** — Matches the known family 'vpn_chat' (p=1.00). Driven by: unusually low fwd init win bytes (0.0); unusually low bwd init win bytes (0.0); unusually high flow iat min (134,025.0).
* **vpn_chat** — Matches the known family 'vpn_chat' (p=1.00). Driven by: unusually low fwd init win bytes (0.0); unusually low bwd init win bytes (0.0); unusually high flow iat min (129,889.0).

---

## Source: `ctu13`

### Detection performance (held-out test split)

| Detector | Accuracy | Precision | Recall | F1-score | False-positive rate | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Stage 1 (supervised) | 0.9977 | 0.8600 | 0.9211 | 0.8895 | 0.0015 | 0.9972 | 0.9534 |
| Stage 2 (unsupervised) | 0.9810 | 0.1159 | 0.1321 | 0.1235 | 0.0103 | 0.7700 | 0.0895 |
| **Fused (both)** | 0.9876 | 0.4459 | 0.9270 | 0.6022 | 0.0118 | 0.9817 | 0.8163 |

> **OR-fusion costs more than it gains on this source.** Stage 1 alone scores F1 0.8895; fusing with Stage 2 drops it to 0.6022 (-0.2873) while adding +0.0059 recall.
>
> The cause is class imbalance: benign flows outnumber malicious ones roughly 98:1 here, so Stage 2's 1.18% false-positive rate produces 584 false alerts against only 470 true ones. Under that ratio a detector must be far more specific than one calibrated on a balanced set.
>
> Practical response: raise `anomaly.threshold_percentile` (see the operating-point table below), or switch `fusion.rule` to `weighted` so Stage 2 escalates rather than alerts on its own. Reported as measured.

**Attack-family classification (Stage 1):** accuracy 0.9970, macro-F1 0.7620, weighted-F1 0.9971

### Zero-day experiment — `botnet_menti` withheld from Stage 1

| Measure | Value |
|---|---|
| Flows of the withheld family in the test split | 202 |
| Detected overall | 0.901 |
| Stage 1 fired | 0.901 |
| Stage 2 fired | 0.000 |
| **Stage 2 only (Stage 1 silent)** | **0.000** |
| Missed entirely | 20 |

_Stage 1 still catches part of a withheld family because it has learned other malicious families and `P(malicious)` aggregates over all of them. The last row is the coverage that exists only because Stage 2 is there._

### Per-family alert rate

| Traffic family | Alert rate | Reading |
|---|---|---|
| benign | 0.012 | false-positive rate |
| botnet_sogou | 0.400 | detection rate |
| botnet_nsis_ay | 0.855 | detection rate |
| botnet_virut | 0.919 | detection rate |
| botnet_rbot | 0.984 | detection rate |
| botnet_menti | 0.985 | detection rate |

_Benign families in this table are the per-class false-positive rate._

### Stage-2 operating points

| Percentile | Threshold | Recall | Precision | FPR |
|---|---|---|---|---|
| p90 | 0.0003 | 0.483 | 0.047 | 0.1000 |
| p95 | 0.0006 | 0.154 | 0.031 | 0.0500 |
| p97.5 | 0.0010 | 0.144 | 0.056 | 0.0250 |
| p99 | 0.0019 | 0.132 | 0.119 | 0.0100 |
| p99.5 | 0.0033 | 0.128 | 0.208 | 0.0050 |
| p99.9 | 0.0104 | 0.118 | 0.545 | 0.0010 |

_The threshold is a deployment choice; this is the trade-off an operator picks from._

### Scoring latency

| Batch size | Total (ms) | Per flow (ms) | Flows/sec |
|---|---|---|---|
| 1 | 28.00 | 27.9984 | 36 |
| 32 | 34.52 | 1.0787 | 927 |
| 256 | 39.27 | 0.1534 | 6,518 |

_Batch size 1 is dominated by fixed per-call overhead; the batched figures reflect real throughput._

### Example alert explanations

* **botnet_nsis_ay** — Matches the known family 'botnet_nsis_ay' (p=1.00). Driven by: unusually low throughput (45.9); unusually low fwd packets per s (0.3); unusually low bytes transferred (280.0).
* **botnet_rbot** — Matches 'botnet_rbot' and is 10.2 sigma from normal. Driven by: unusually high average packet size (1,066.0); unusually low throughput (213.3); unusually high average client packet size (1,066.0).
* **benign** — Does not match any known traffic profile (1.9 sigma from normal) - possible unseen threat. Driven by: unusually low bytes sent by client (0.0); unusually low download/upload packet ratio (0.0); unusually high bwd packets per s (35,087.7).
* **benign** — Does not match any known traffic profile (1.6 sigma from normal) - possible unseen threat. Driven by: unusually high fwd packets (470.0); unusually high packet count (470.0); unusually high bytes sent by client (57,800.0).

---

## Source: `iscx_vpn2016`

### Detection performance (held-out test split)

| Detector | Accuracy | Precision | Recall | F1-score | False-positive rate | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Stage 1 (supervised) | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | nan | nan |
| Stage 2 (unsupervised) | 0.9881 | 0.0000 | 0.0000 | 0.0000 | 0.0119 | nan | nan |
| **Fused (both)** | 0.9881 | 0.0000 | 0.0000 | 0.0000 | 0.0119 | nan | nan |

OR-fusion improves F1 by +0.0000 over Stage 1 alone (recall +0.0000), at a false-positive rate of 0.0119.

**Attack-family classification (Stage 1):** accuracy 0.8900, macro-F1 0.8543, weighted-F1 0.8906

### Ablation — does TLS metadata help?

| Metric | Flow statistics only | Flow + TLS metadata | Δ | Relative |
|---|---|---|---|---|
| Accuracy | 0.8890 | 0.8900 | +0.0010 | +0.11% |
| macro_f1 | 0.8548 | 0.8543 | -0.0005 | -0.06% |
| weighted_f1 | 0.8901 | 0.8906 | +0.0005 | +0.06% |

_The dual-signal claim of the project stands or falls on this table._

### Per-family alert rate

| Traffic family | Alert rate | Reading |
|---|---|---|
| streaming | 0.009 | false-positive rate |
| email | 0.009 | detection rate |
| voip | 0.011 | false-positive rate |
| chat | 0.023 | detection rate |

_Benign families in this table are the per-class false-positive rate._

### Stage-2 operating points

| Percentile | Threshold | Recall | Precision | FPR |
|---|---|---|---|---|
| p90 | 0.0904 | 0.000 | 0.000 | 0.1001 |
| p95 | 0.1491 | 0.000 | 0.000 | 0.0505 |
| p97.5 | 0.3065 | 0.000 | 0.000 | 0.0258 |
| p99 | 0.6424 | 0.000 | 0.000 | 0.0109 |
| p99.5 | 1.3576 | 0.000 | 0.000 | 0.0059 |
| p99.9 | 2.4001 | 0.000 | 0.000 | 0.0020 |

_The threshold is a deployment choice; this is the trade-off an operator picks from._

### Scoring latency

| Batch size | Total (ms) | Per flow (ms) | Flows/sec |
|---|---|---|---|
| 1 | 51.90 | 51.8993 | 19 |
| 32 | 38.75 | 1.2109 | 826 |
| 256 | 44.75 | 0.1748 | 5,721 |

_Batch size 1 is dominated by fixed per-call overhead; the batched figures reflect real throughput._

### Example alert explanations

* **voip** — Does not match any known traffic profile (13.3 sigma from normal) - possible unseen threat. Driven by: unusually high bwd pkt len max (17,374.0); unusually high ack count (385.0); unusually high PSH flag count (187.0).
* **voip** — Does not match any known traffic profile (3.4 sigma from normal) - possible unseen threat. Driven by: unusually high fwd pkt len max (4,420.0); unusually high fwd pkt len std (1,018.3); unusually high TLS handshake duration (99.7).
* **chat** — Does not match any known traffic profile (3.8 sigma from normal) - possible unseen threat. Driven by: unusually high average server packet size (1,360.3); unusually high bwd segment size avg (1,360.3); unusually high download/upload byte ratio (44.4).
* **chat** — Does not match any known traffic profile (3.3 sigma from normal) - possible unseen threat. Driven by: unusually high has alpn (1.0); unusually high HTTP/2 negotiated (1.0); unusually high hostname length (27.0).

---

## Source: `synthetic`

### Detection performance (held-out test split)

| Detector | Accuracy | Precision | Recall | F1-score | False-positive rate | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Stage 1 (supervised) | 0.9894 | 0.9859 | 0.9762 | 0.9810 | 0.0054 | 0.9992 | 0.9981 |
| Stage 2 (unsupervised) | 0.8393 | 0.9425 | 0.4536 | 0.6124 | 0.0108 | 0.8206 | 0.7584 |
| **Fused (both)** | 0.9817 | 0.9591 | 0.9762 | 0.9676 | 0.0162 | 0.9911 | 0.9835 |

> **OR-fusion costs more than it gains on this source.** Stage 1 alone scores F1 0.9810; fusing with Stage 2 drops it to 0.9676 (-0.0135) while adding +0.0000 recall.
>
> The classes are only about 3:1 here, so this is not an imbalance artefact: Stage 2 simply found nothing Stage 1 had missed (+0.0000 recall) while still contributing 140 false alerts. Stage 2 earns its place on *unseen* families - see the zero-day experiment - not on families Stage 1 already covers.
>
> Practical response: raise `anomaly.threshold_percentile` (see the operating-point table below), or switch `fusion.rule` to `weighted` so Stage 2 escalates rather than alerts on its own. Reported as measured.

**Attack-family classification (Stage 1):** accuracy 0.9496, macro-F1 0.9185, weighted-F1 0.9492

### Ablation — does TLS metadata help?

| Metric | Flow statistics only | Flow + TLS metadata | Δ | Relative |
|---|---|---|---|---|
| Accuracy | 0.9516 | 0.9817 | +0.0301 | +3.16% |
| F1-score | 0.9122 | 0.9676 | +0.0553 | +6.06% |
| Recall | 0.8988 | 0.9762 | +0.0774 | +8.61% |
| Precision | 0.9261 | 0.9591 | +0.0330 | +3.56% |
| False-positive rate | 0.0279 | 0.0162 | -0.0117 | -41.91% |
| ROC-AUC | 0.9764 | 0.9911 | +0.0147 | +1.50% |

_The dual-signal claim of the project stands or falls on this table._

### Zero-day experiment — `tor_darknet` withheld from Stage 1

| Measure | Value |
|---|---|
| Flows of the withheld family in the test split | 607 |
| Detected overall | 0.649 |
| Stage 1 fired | 0.639 |
| Stage 2 fired | 0.488 |
| **Stage 2 only (Stage 1 silent)** | **0.010** |
| Missed entirely | 213 |

_Stage 1 still catches part of a withheld family because it has learned other malicious families and `P(malicious)` aggregates over all of them. The last row is the coverage that exists only because Stage 2 is there._

### Per-family alert rate

| Traffic family | Alert rate | Reading |
|---|---|---|
| web_browsing | 0.013 | false-positive rate |
| vpn_tunnel | 0.014 | false-positive rate |
| streaming | 0.015 | false-positive rate |
| voip | 0.015 | false-positive rate |
| file_transfer | 0.034 | false-positive rate |
| data_exfiltration | 0.936 | detection rate |
| tor_darknet | 0.975 | detection rate |
| crypto_mining | 0.976 | detection rate |
| port_scan | 0.982 | detection rate |
| ddos_flood | 0.987 | detection rate |
| c2_beacon | 0.989 | detection rate |

_Benign families in this table are the per-class false-positive rate._

### Detection vs. evasion effort

| Evasion level | Flows | Detection rate |
|---|---|---|
| none | 2,174 | 0.990 |
| light (0-0.3) | 223 | 0.991 |
| moderate (0.3-0.6) | 556 | 0.968 |
| heavy (0.6-1.0) | 407 | 0.904 |

_Malicious flows only, split by how far they were shaped toward looking benign._

### Stage-2 operating points

| Percentile | Threshold | Recall | Precision | FPR |
|---|---|---|---|---|
| p90 | 0.0993 | 0.613 | 0.704 | 0.1000 |
| p95 | 0.1350 | 0.540 | 0.808 | 0.0500 |
| p97.5 | 0.1686 | 0.485 | 0.883 | 0.0250 |
| p99 | 0.2064 | 0.452 | 0.946 | 0.0101 |
| p99.5 | 0.2270 | 0.430 | 0.970 | 0.0051 |
| p99.9 | 0.5910 | 0.272 | 0.990 | 0.0010 |

_The threshold is a deployment choice; this is the trade-off an operator picks from._

### Scoring latency

| Batch size | Total (ms) | Per flow (ms) | Flows/sec |
|---|---|---|---|
| 1 | 55.10 | 55.1008 | 18 |
| 32 | 64.61 | 2.0190 | 495 |
| 256 | 72.00 | 0.2813 | 3,556 |

_Batch size 1 is dominated by fixed per-call overhead; the batched figures reflect real throughput._

### Example alert explanations

* **crypto_mining** — Matches the known family 'crypto_mining' (p=1.00). Driven by: unusually low throughput (1,248.2); unusually high average server send interval (647.6); unusually high average gap between packets (388.2).
* **voip** — Does not match any known traffic profile (4.3 sigma from normal) - possible unseen threat. Driven by: unusually high PSH flag count (6,482.0); unusually high ack count (16,543.0); unusually high urg count (27.0).
* **c2_beacon** — Matches the known family 'c2_beacon' (p=1.00). Driven by: unusually low throughput (3,496.9); unusually high average server send interval (203.2); unusually low packet rate (7.7).
* **data_exfiltration** — Matches 'data_exfiltration' and is 9.8 sigma from normal. Driven by: unusually low average gap between packets (28.2); unusually high average server packet size (922.9); unusually high average packet size (827.6).

---
