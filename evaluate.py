"""
evaluate.py — Evaluation & Failure Analysis
=============================================
Reads the output CSVs from classify_llm.py and prints a full breakdown of
model performance. Also saves a failure analysis CSV for your report.

Usage:
  python evaluate.py                          # evaluates both result files
  python evaluate.py --file llm_results_zero_shot.csv   # one file only

What this produces:
  1. Overall accuracy (zero-shot vs. few-shot comparison)
  2. Accuracy by party (Democrat vs. Republican)
  3. Accuracy by era (trump-era-early / trump-era-late / biden-era)
  4. Accuracy: normal cases vs. edge cases
  5. Accuracy by individual edge-case senator
  6. Confidence calibration — is high confidence actually more accurate?
  7. failure_analysis.csv — all misclassified tweets for your report
"""

import argparse
import pandas as pd


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def accuracy(df: pd.DataFrame) -> float:
    """Overall accuracy, ignoring parse errors."""
    valid = df[~df["parse_error"]]
    if len(valid) == 0:
        return 0.0
    correct = (valid["pred_leaning"] == valid["party"]).sum()
    return correct / len(valid)


def accuracy_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Returns a DataFrame with accuracy and sample size per group."""
    valid = df[~df["parse_error"]]
    rows = []
    for group, subset in valid.groupby(group_col):
        n = len(subset)
        n_correct = (subset["pred_leaning"] == subset["party"]).sum()
        rows.append({
            group_col:   group,
            "n_tweets":  n,
            "n_correct": n_correct,
            "accuracy":  n_correct / n if n > 0 else 0.0,
        })
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False)


def print_section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


CALIBRATION_BINS = [(0, 50), (50, 70), (70, 85), (85, 95), (95, 101)]


def fit_calibration_threshold(df: pd.DataFrame, bins=CALIBRATION_BINS) -> dict:
    """
    Fits a simple confidence-calibration model: buckets predictions by
    confidence, measures accuracy per bucket, then scans the boundaries
    between adjacent buckets and picks the one with the largest accuracy
    jump. That boundary is the triage threshold — confidence at or above
    it is accurate often enough to auto-accept; confidence below it gets
    routed to human review.
    """
    valid = df[~df["parse_error"]]
    bucket_stats = []
    for lo, hi in bins:
        bucket = valid[(valid["confidence"] >= lo) & (valid["confidence"] < hi)]
        if len(bucket) == 0:
            continue
        n = len(bucket)
        c = int((bucket["pred_leaning"] == bucket["party"]).sum())
        bucket_stats.append({"lo": lo, "hi": hi, "n": n, "accuracy": c / n})

    best = None
    for prev, cur in zip(bucket_stats, bucket_stats[1:]):
        jump = (cur["accuracy"] - prev["accuracy"]) * 100
        if best is None or jump > best["jump_points"]:
            best = {
                "threshold": cur["lo"],
                "jump_points": jump,
                "below_accuracy": prev["accuracy"],
                "at_or_above_accuracy": cur["accuracy"],
            }

    return {"bucket_stats": bucket_stats, "best_jump": best}


def apply_triage(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Automated triage rule built on the fitted threshold: predictions with
    confidence >= threshold are auto-accepted; predictions below it are
    flagged for human review before being trusted.
    """
    out = df.copy()
    out["triage_action"] = out["confidence"].apply(
        lambda c: "auto_accept" if c >= threshold else "flag_for_review"
    )
    return out


# ---------------------------------------------------------------------------
# MAIN ANALYSIS
# ---------------------------------------------------------------------------

def analyze(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Runs the full evaluation suite on one results DataFrame.
    Returns a DataFrame of misclassified rows for failure export.
    """
    print(f"\n{'='*60}")
    print(f"  RESULTS: {label}")
    print(f"{'='*60}")

    parse_errors = df["parse_error"].sum()
    if parse_errors > 0:
        print(f"  ⚠ Parse errors (excluded from accuracy): {parse_errors}")

    valid = df[~df["parse_error"]]

    # ------------------------------------------------------------------
    # 1. Overall accuracy
    # ------------------------------------------------------------------
    print_section("1. Overall Accuracy")
    overall = accuracy(df)
    print(f"  {overall:.1%}  ({(valid['pred_leaning'] == valid['party']).sum()} / {len(valid)} correct)")

    # ------------------------------------------------------------------
    # 2. By party
    # ------------------------------------------------------------------
    print_section("2. Accuracy by Party")
    party_acc = accuracy_by_group(valid, "party")
    print(party_acc.to_string(index=False))

    # ------------------------------------------------------------------
    # 3. By era
    # ------------------------------------------------------------------
    print_section("3. Accuracy by Political Era")
    era_acc = accuracy_by_group(valid, "era")
    print(era_acc.to_string(index=False))
    print("\n  NOTE: biden-era dominates (~99% of data). Cross-era numbers")
    print("  have low statistical power — flag this in the report.")

    # ------------------------------------------------------------------
    # 4. Normal vs. edge cases
    # ------------------------------------------------------------------
    print_section("4. Normal Cases vs. Edge Cases")
    normal = valid[~valid["is_edge_case"]]
    edge   = valid[ valid["is_edge_case"]]

    def _acc_line(subset, name):
        n = len(subset)
        c = (subset["pred_leaning"] == subset["party"]).sum()
        pct = c/n if n > 0 else 0.0
        return f"  {name:<20} {pct:.1%}  ({c}/{n})"

    print(_acc_line(normal, "Normal cases"))
    print(_acc_line(edge,   "Edge cases"))

    # ------------------------------------------------------------------
    # 5. Per edge-case senator
    # ------------------------------------------------------------------
    if "username" in valid.columns:
        print_section("5. Accuracy per Edge-Case Senator")
        edge_senators = valid[valid["is_edge_case"]]
        if len(edge_senators) > 0:
            senator_acc = accuracy_by_group(edge_senators, "username")
            print(senator_acc.to_string(index=False))
        else:
            print("  No edge-case rows found in this file.")

    # ------------------------------------------------------------------
    # 6. Confidence calibration
    # ------------------------------------------------------------------
    print_section("6. Confidence Calibration")
    print("  (Are high-confidence predictions actually more accurate?)\n")

    bins = [(0, 50), (50, 70), (70, 85), (85, 95), (95, 101)]
    for lo, hi in bins:
        bucket = valid[(valid["confidence"] >= lo) & (valid["confidence"] < hi)]
        if len(bucket) == 0:
            continue
        n = len(bucket)
        c = (bucket["pred_leaning"] == bucket["party"]).sum()
        print(f"  Confidence {lo:>3}–{hi-1:<3}: {c/n:.1%}  ({c}/{n})")

    # ------------------------------------------------------------------
    # 7. Failures
    # ------------------------------------------------------------------
    print_section("7. Sample Failures (first 5)")
    failures = valid[valid["pred_leaning"] != valid["party"]]
    print(f"  Total misclassified: {len(failures)} / {len(valid)}\n")

    for _, row in failures.head(5).iterrows():
        print(f"  USERNAME:   {row.get('username', 'N/A')}")
        print(f"  TWEET:      {row['text'][:120]}...")
        print(f"  TRUE LABEL: {row['party']}")
        print(f"  PREDICTED:  {row['pred_leaning']}  (confidence: {row['confidence']})")
        print(f"  REASONING:  {row['reasoning']}")
        print()

    # ------------------------------------------------------------------
    # 8. Confidence-calibration model & automated triage
    # ------------------------------------------------------------------
    print_section("8. Confidence-Calibration Model & Automated Triage")
    calib = fit_calibration_threshold(df)
    for b in calib["bucket_stats"]:
        print(f"  [{b['lo']:>3},{b['hi']:<3}): n={b['n']:>3}  accuracy={b['accuracy']:.1%}")

    triage_flagged = valid.iloc[0:0]
    best = calib["best_jump"]
    if best:
        print(f"\n  Fitted triage threshold: confidence >= {best['threshold']}")
        print(f"    Below threshold accuracy:    {best['below_accuracy']:.1%}")
        print(f"    At/above threshold accuracy: {best['at_or_above_accuracy']:.1%}")
        print(f"    Accuracy jump at this threshold: {best['jump_points']:.1f} points")

        triaged = apply_triage(valid, best["threshold"])
        auto = triaged[triaged["triage_action"] == "auto_accept"]
        review = triaged[triaged["triage_action"] == "flag_for_review"]
        auto_acc = (auto["pred_leaning"] == auto["party"]).mean() if len(auto) else 0.0
        review_acc = (review["pred_leaning"] == review["party"]).mean() if len(review) else 0.0
        print(f"    Auto-accept:  {len(auto):>3}/{len(triaged)} ({len(auto)/len(triaged):.1%}) rows, accuracy {auto_acc:.1%}")
        print(f"    Flag-review:  {len(review):>3}/{len(triaged)} ({len(review)/len(triaged):.1%}) rows, accuracy {review_acc:.1%}")
        triage_flagged = review

    return failures, triage_flagged


# ---------------------------------------------------------------------------
# COMPARISON TABLE (zero-shot vs. few-shot)
# ---------------------------------------------------------------------------

def comparison_table(results: dict):
    """Prints a side-by-side summary if multiple result files are loaded."""
    print(f"\n{'='*60}")
    print("  SUMMARY COMPARISON")
    print(f"{'='*60}")

    rows = []
    for label, df in results.items():
        valid = df[~df["parse_error"]]
        n = len(valid)
        edge = valid[valid["is_edge_case"]]
        normal = valid[~valid["is_edge_case"]]

        rows.append({
            "Mode":            label,
            "Overall":         f"{accuracy(df):.1%}",
            "Normal cases":    f"{(normal['pred_leaning'] == normal['party']).sum() / len(normal):.1%}" if len(normal) else "—",
            "Edge cases":      f"{(edge['pred_leaning'] == edge['party']).sum() / len(edge):.1%}" if len(edge) else "—",
            "Parse errors":    df["parse_error"].sum(),
        })

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM classification results")
    parser.add_argument("--file", type=str, default=None,
                        help="Evaluate a single result file instead of both defaults")
    args = parser.parse_args()

    if args.file:
        files = {args.file: args.file}
    else:
        files = {
            "zero-shot": "llm_results_zero_shot.csv",
            "few-shot":  "llm_results_few_shot.csv",
        }

    loaded = {}
    all_failures = []
    all_triage_flagged = []

    for label, path in files.items():
        try:
            df = pd.read_csv(path)
            # Ensure parse_error column exists (older runs may not have it)
            if "parse_error" not in df.columns:
                df["parse_error"] = False
            df["parse_error"] = df["parse_error"].fillna(False).astype(bool)
            loaded[label] = df
        except FileNotFoundError:
            print(f"  ⚠ File not found: {path} — skipping")

    if not loaded:
        print("No result files found. Run classify_llm.py first.")
        return

    for label, df in loaded.items():
        failures, triage_flagged = analyze(df, label)
        failures["mode"] = label
        all_failures.append(failures)
        if len(triage_flagged) > 0:
            triage_flagged = triage_flagged.copy()
            triage_flagged["mode"] = label
            all_triage_flagged.append(triage_flagged)

    if len(loaded) > 1:
        comparison_table(loaded)

    # Save all failures to one CSV for the report
    if all_failures:
        failure_df = pd.concat(all_failures, ignore_index=True)
        failure_df.to_csv("failure_analysis.csv", index=False)
        print(f"\n  Failure analysis saved to failure_analysis.csv ({len(failure_df)} rows)")

    # Save the automated-triage review queue (rows the calibration model
    # flagged as below its fitted confidence threshold) for the report
    if all_triage_flagged:
        triage_df = pd.concat(all_triage_flagged, ignore_index=True)
        triage_df.to_csv("triage_review_queue.csv", index=False)
        print(f"  Triage review queue saved to triage_review_queue.csv ({len(triage_df)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
