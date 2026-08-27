from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

PREDICTION_LOG = BASE_DIR / "reports" / "prediction_log.csv"
DRIFT_REPORT = BASE_DIR / "reports" / "drift_report.txt"
OUTPUT_REPORT = BASE_DIR / "reports" / "monitoring_report.txt"


# Operational thresholds
MAE_THRESHOLD_SECONDS = 600
SIGNIFICANT_DRIFT_PSI = 0.25


def main():

    predictions = pd.read_csv(PREDICTION_LOG)

    mae = predictions["absolute_error"].mean()

    rmse = (
        (
            predictions["prediction_error"] ** 2
        ).mean()
        ** 0.5
    )

    drift_text = DRIFT_REPORT.read_text(
        encoding="utf-8"
    )

    significant_drift = (
        "SIGNIFICANT DRIFT" in drift_text
    )

    performance_issue = (
        mae > MAE_THRESHOLD_SECONDS
    )

    retraining_required = (
        significant_drift
        or performance_issue
    )

    lines = []

    lines.append(
        "NYC TAXI ETA - MODEL MONITORING REPORT"
    )
    lines.append("=" * 60)
    lines.append("")

    lines.append("PREDICTION PERFORMANCE")
    lines.append("-" * 60)
    lines.append(
        f"Predictions monitored: {len(predictions):,}"
    )
    lines.append(
        f"MAE: {mae:.2f} seconds"
    )
    lines.append(
        f"RMSE: {rmse:.2f} seconds"
    )
    lines.append(
        f"MAE threshold: {MAE_THRESHOLD_SECONDS:.2f} seconds"
    )

    lines.append("")

    if performance_issue:
        lines.append(
            "Performance status: BELOW EXPECTED PERFORMANCE"
        )
    else:
        lines.append(
            "Performance status: ACCEPTABLE"
        )

    lines.append("")
    lines.append("DRIFT MONITORING")
    lines.append("-" * 60)

    if significant_drift:
        lines.append(
            "Drift status: SIGNIFICANT DRIFT DETECTED"
        )
    else:
        lines.append(
            "Drift status: NO SIGNIFICANT DRIFT"
        )

    lines.append("")
    lines.append("RETRAINING TRIGGER")
    lines.append("-" * 60)

    lines.append(
        "Trigger conditions:"
    )
    lines.append(
        f"1. MAE > {MAE_THRESHOLD_SECONDS} seconds"
    )
    lines.append(
        f"2. Any monitored feature PSI > {SIGNIFICANT_DRIFT_PSI}"
    )

    lines.append("")

    if retraining_required:
        lines.append(
            "DECISION: RETRAINING REQUIRED"
        )
        lines.append(
            "Reason: Model performance or feature drift "
            "exceeded the operational threshold."
        )
    else:
        lines.append(
            "DECISION: NO RETRAINING REQUIRED"
        )

    lines.append("")
    lines.append("RETRAINING DESIGN")
    lines.append("-" * 60)
    lines.append(
        "When the retraining trigger fires, the DVC pipeline "
        "should be executed to regenerate features and retrain "
        "the model."
    )
    lines.append(
        "The resulting model should then be evaluated against "
        "the existing model before deployment."
    )

    OUTPUT_REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\n".join(lines))
    print("")
    print(
        f"Saved monitoring report to: {OUTPUT_REPORT}"
    )


if __name__ == "__main__":
    main()