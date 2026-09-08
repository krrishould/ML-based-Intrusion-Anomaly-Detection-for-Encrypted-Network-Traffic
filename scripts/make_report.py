"""Turn the evaluation JSON into a report-ready Markdown summary.

    python scripts/make_report.py
    python scripts/make_report.py --out reports/RESULTS.md

Reads every ``reports/metrics/evaluation*.json`` produced by
``evaluate_model.py`` and writes tables that can be pasted straight into the
project report or the review slides.

Nothing is computed here - this only formats what evaluation already measured,
so the report cannot drift away from the numbers in the JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from encids.config import Paths, load_config              # noqa: E402
from encids.utils.logging_utils import get_logger         # noqa: E402

log = get_logger("scripts.report")

HEADLINE = ["accuracy", "precision", "recall", "f1", "false_positive_rate",
            "roc_auc", "pr_auc"]
PRETTY = {
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
    "f1": "F1-score", "false_positive_rate": "False-positive rate",
    "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "balanced_accuracy":
    "Balanced accuracy", "mcc": "MCC",
}


def fmt(value: Any, places: int = 4) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{places}f}" if isinstance(value, float) else f"{value:,}"
    return str(value)


def table(rows: list[list[str]], headers: list[str]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def stage_comparison(block: dict[str, Any]) -> str:
    """Stage 1 vs Stage 2 vs fused."""
    stages = [("Stage 1 (supervised)", "stage1_only"),
              ("Stage 2 (unsupervised)", "stage2_only"),
              ("**Fused (both)**", "fused")]
    rows = []
    for label, key in stages:
        m = block.get(key)
        if not m:
            continue
        rows.append([label] + [fmt(m.get(k, float("nan"))) for k in HEADLINE])
    return table(rows, ["Detector"] + [PRETTY[k] for k in HEADLINE])


def fusion_note(block: dict[str, Any]) -> str:
    """State plainly whether OR-fusion actually helped on this source.

    Under heavy class imbalance it often does not: Stage 2 contributes a
    roughly fixed false-positive rate, and when benign flows outnumber
    malicious ones 100:1 that costs far more precision than the extra recall is
    worth. Reporting the fused number without this comparison would hide it.
    """
    fused, stage1 = block.get("fused"), block.get("stage1_only")
    if not fused or not stage1:
        return ""

    delta_f1 = fused["f1"] - stage1["f1"]
    delta_recall = fused["recall"] - stage1["recall"]
    imbalance = ((fused["true_negatives"] + fused["false_positives"])
                 / max(fused["true_positives"] + fused["false_negatives"], 1))

    if delta_f1 >= 0:
        return (f"OR-fusion improves F1 by {delta_f1:+.4f} over Stage 1 alone "
                f"(recall {delta_recall:+.4f}), at a false-positive rate of "
                f"{fused['false_positive_rate']:.4f}.")

    head = (f"> **OR-fusion costs more than it gains on this source.** Stage 1 "
            f"alone scores F1 {stage1['f1']:.4f}; fusing with Stage 2 drops it "
            f"to {fused['f1']:.4f} ({delta_f1:+.4f}) while adding "
            f"{delta_recall:+.4f} recall.\n>\n")

    # Only blame imbalance when the ratio actually is severe - at 3:1 the cause
    # is simply that Stage 2 found nothing new, and saying otherwise would be a
    # misdiagnosis printed as a finding.
    if imbalance >= 20:
        cause = (f"> The cause is class imbalance: benign flows outnumber "
                 f"malicious ones roughly {imbalance:.0f}:1 here, so Stage 2's "
                 f"{fused['false_positive_rate']:.2%} false-positive rate "
                 f"produces {fused['false_positives']:,} false alerts against "
                 f"only {fused['true_positives']:,} true ones. Under that ratio "
                 f"a detector must be far more specific than one calibrated on "
                 f"a balanced set.\n>\n")
    else:
        cause = (f"> The classes are only about {imbalance:.0f}:1 here, so this "
                 f"is not an imbalance artefact: Stage 2 simply found nothing "
                 f"Stage 1 had missed ({delta_recall:+.4f} recall) while still "
                 f"contributing {fused['false_positives']:,} false alerts. "
                 f"Stage 2 earns its place on *unseen* families - see the "
                 f"zero-day experiment - not on families Stage 1 already "
                 f"covers.\n>\n")

    return head + cause + (
        "> Practical response: raise `anomaly.threshold_percentile` "
        "(see the operating-point table below), or switch `fusion.rule` to "
        "`weighted` so Stage 2 escalates rather than alerts on its own. "
        "Reported as measured.")


def ablation_table(results: dict[str, Any]) -> str:
    comparison = results.get("comparison")
    if not comparison:
        return "_Ablation not run for this source._"
    rows = [[PRETTY.get(r["metric"], r["metric"]),
             fmt(r["flow_only_baseline"]), fmt(r["flow_plus_tls"]),
             f"{r['delta']:+.4f}",
             f"{r['relative_%']:+.2f}%" if r.get("relative_%") is not None else "-"]
            for r in comparison]
    return table(rows, ["Metric", "Flow statistics only", "Flow + TLS metadata",
                        "Δ", "Relative"])


def family_table(block: dict[str, Any]) -> str:
    rates = block.get("per_family_detection_rate", {})
    if not rates:
        return ""
    rows = [[family, fmt(rate, 3),
             "false-positive rate" if rate < 0.5 and "benign" in family
             or family in ("benign", "web_browsing", "streaming", "voip",
                           "file_transfer", "vpn_tunnel") else "detection rate"]
            for family, rate in sorted(rates.items(), key=lambda kv: kv[1])]
    return table(rows, ["Traffic family", "Alert rate", "Reading"])


def operating_points(block: dict[str, Any]) -> str:
    points = block.get("stage2_operating_points")
    if not points:
        return ""
    rows = [[f"p{p['percentile']:g}", fmt(p["threshold"], 4), fmt(p["recall"], 3),
             fmt(p["precision"], 3), fmt(p["false_positive_rate"], 4)]
            for p in points]
    return table(rows, ["Percentile", "Threshold", "Recall", "Precision", "FPR"])


def evasion_table(block: dict[str, Any]) -> str:
    bands = block.get("detection_by_evasion_level")
    if not bands:
        return ""
    rows = [[str(b["band"]), fmt(int(b["n_flows"])), fmt(b["detection_rate"], 3)]
            for b in bands]
    return table(rows, ["Evasion level", "Flows", "Detection rate"])


def latency_table(results: dict[str, Any]) -> str:
    latency = results.get("latency")
    if not latency:
        return ""
    rows = [[key.replace("batch_", ""), fmt(v["total_ms"], 2),
             fmt(v["per_flow_ms"], 4), f"{v['flows_per_second']:,.0f}"]
            for key, v in latency.items()]
    return table(rows, ["Batch size", "Total (ms)", "Per flow (ms)", "Flows/sec"])


def render(path: Path) -> str:
    results = json.loads(path.read_text(encoding="utf-8"))
    source = results.get("source_filter") or "all sources"
    block = results.get("flow_plus_tls") or results.get("fused") or results

    parts = [f"## Source: `{source}`", ""]

    parts += ["### Detection performance (held-out test split)", "",
              stage_comparison(block), "", fusion_note(block), ""]

    if "family_classification" in block:
        fc = block["family_classification"]
        parts += ["**Attack-family classification (Stage 1):** "
                  f"accuracy {fmt(fc['accuracy'])}, "
                  f"macro-F1 {fmt(fc['macro_f1'])}, "
                  f"weighted-F1 {fmt(fc['weighted_f1'])}", ""]

    if results.get("comparison"):
        parts += ["### Ablation — does TLS metadata help?", "",
                  ablation_table(results), "",
                  "_The dual-signal claim of the project stands or falls on "
                  "this table._", ""]

    if "zero_day" in results and results["zero_day"].get("n_flows"):
        z = results["zero_day"]
        parts += [f"### Zero-day experiment — `{z['held_out']}` withheld from "
                  "Stage 1", "",
                  table([
                      ["Flows of the withheld family in the test split",
                       fmt(int(z["n_flows"]))],
                      ["Detected overall", fmt(z["overall_detection_rate"], 3)],
                      ["Stage 1 fired", fmt(z["stage1_rate"], 3)],
                      ["Stage 2 fired", fmt(z["stage2_rate"], 3)],
                      ["**Stage 2 only (Stage 1 silent)**",
                       f"**{fmt(z['stage2_only_rate'], 3)}**"],
                      ["Missed entirely", fmt(int(z["missed"]))],
                  ], ["Measure", "Value"]),
                  "",
                  "_Stage 1 still catches part of a withheld family because it "
                  "has learned other malicious families and `P(malicious)` "
                  "aggregates over all of them. The last row is the coverage "
                  "that exists only because Stage 2 is there._", ""]

    fam = family_table(block)
    if fam:
        parts += ["### Per-family alert rate", "", fam, "",
                  "_Benign families in this table are the per-class "
                  "false-positive rate._", ""]

    evasion = evasion_table(block)
    if evasion:
        parts += ["### Detection vs. evasion effort", "", evasion, "",
                  "_Malicious flows only, split by how far they were shaped "
                  "toward looking benign._", ""]

    ops = operating_points(block)
    if ops:
        parts += ["### Stage-2 operating points", "", ops, "",
                  "_The threshold is a deployment choice; this is the "
                  "trade-off an operator picks from._", ""]

    lat = latency_table(results)
    if lat:
        parts += ["### Scoring latency", "", lat, "",
                  "_Batch size 1 is dominated by fixed per-call overhead; "
                  "the batched figures reflect real throughput._", ""]

    if "per_source" in block:
        rows = [[name, fmt(int(m["n_flows"])), fmt(m["malicious_rate"], 3),
                 fmt(m["f1"]), fmt(m["recall"]), fmt(m["false_positive_rate"])]
                for name, m in block["per_source"].items()]
        parts += ["### Per-dataset breakdown", "",
                  table(rows, ["Source", "Flows", "Malicious rate", "F1",
                               "Recall", "FPR"]), ""]

    if results.get("example_explanations"):
        parts += ["### Example alert explanations", ""]
        for item in results["example_explanations"][:4]:
            parts.append(f"* **{item['true_label']}** — {item['summary']}")
        parts.append("")

    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="reports/RESULTS.md")
    args = parser.parse_args()

    paths = Paths.from_config(load_config())
    files = sorted(paths.metrics.glob("evaluation*.json"))
    if not files:
        log.error("No evaluation JSON in %s - run scripts/evaluate_model.py first",
                  paths.metrics)
        return 1

    sections = [
        "# Results",
        "",
        "ML-based Intrusion/Anomaly Detection for Encrypted Network Traffic — "
        "BCSE497J",
        "",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M} from "
        f"{', '.join(f.name for f in files)}._",
        "",
        "> Results from the `synthetic` source are development figures and are "
        "labelled as such; the `ctu13` source is real captured botnet traffic. "
        "CTU-13 is netflow-only and carries no TLS handshake metadata, so the "
        "dual-signal ablation is reported for `synthetic` only.",
        "",
        "---",
        "",
    ]
    for path in files:
        sections.append(render(path))
        sections.append("---\n")

    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[1] / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sections), encoding="utf-8")
    log.info("Results summary -> %s", out)

    figures = sorted(paths.figures.glob("*.png"))
    if figures:
        log.info("Figures available for the report:")
        for figure in figures:
            log.info("   %s", figure.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
