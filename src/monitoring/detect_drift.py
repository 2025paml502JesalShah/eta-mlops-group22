from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "features" / "nyc_features.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = REPORT_DIR / "drift_report.txt"


# Features we want to monitor
MONITOR_FEATURES = [
    "haversine_km",
    "passenger_count",
    "hour",
    "day_of_week",
]


def calculate_psi(expected, actual, bins=10):
    """
    Calculate Population Stability Index (PSI).

    PSI interpretation:
        < 0.10       No significant drift
        0.10 - 0.25  Moderate drift
        > 0.25       Significant drift
    """

    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # Remove invalid values
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]

    # Create bins using the baseline distribution
    breakpoints = np.percentile(
        expected,
        np.linspace(0, 100, bins + 1),
    )

    # Remove duplicate breakpoints
    breakpoints = np.unique(breakpoints)

    # Need at least two boundaries
    if len(breakpoints) < 2:
        return 0.0

    expected_counts, _ = np.histogram(
        expected,
        bins=breakpoints,
    )

    actual_counts, _ = np.histogram(
        actual,
        bins=breakpoints,
    )

    # Convert counts to percentages
    expected_pct = expected_counts / max(len(expected), 1)
    actual_pct = actual_counts / max(len(actual), 1)

    # Avoid division by zero
    expected_pct = np.clip(expected_pct, 0.0001, None)
    actual_pct = np.clip(actual_pct, 0.0001, None)

    psi = np.sum(
        (actual_pct - expected_pct)
        * np.log(actual_pct / expected_pct)
    )

    return float(psi)


def simulate_drift(df):
    """
    Create a simulated current-production dataset.

    The simulation represents a realistic traffic-pattern change:
    - More rush-hour trips
    - Longer trip distances
    - Slightly higher passenger counts
    """

    current = df.sample(
        n=min(5000, len(df)),
        random_state=42,
    ).copy()

    rng = np.random.default_rng(42)

    # Simulate more rush-hour traffic
    rush_mask = rng.random(len(current)) < 0.40

    current.loc[rush_mask, "hour"] = rng.choice(
        [7, 8, 9, 16, 17, 18, 19],
        size=rush_mask.sum(),
    )

    # Simulate increased trip distances
    current["haversine_km"] = (
        current["haversine_km"] * 1.20
    )

    # Slightly increase passenger counts
    current["passenger_count"] = np.clip(
        current["passenger_count"] + rng.binomial(
            1,
            0.15,
            size=len(current),
        ),
        1,
        6,
    )

    return current


def main():

    print("Loading feature data...")

    df = pd.read_csv(DATA_PATH)

    print(f"Loaded {len(df):,} rows.")

    # Baseline sample
    baseline = df.sample(
        n=min(5000, len(df)),
        random_state=1,
    ).copy()

    # Simulated current production data
    current = simulate_drift(df)

    results = []

    for feature in MONITOR_FEATURES:

        psi = calculate_psi(
            baseline[feature],
            current[feature],
        )

        if psi < 0.10:
            status = "NO DRIFT"
        elif psi <= 0.25:
            status = "MODERATE DRIFT"
        else:
            status = "SIGNIFICANT DRIFT"

        results.append(
            {
                "feature": feature,
                "psi": psi,
                "status": status,
            }
        )

    results_df = pd.DataFrame(results)

    # Trigger retraining if any feature has significant drift
    retraining_required = bool(
        (results_df["psi"] > 0.25).any()
    )

    # --------------------------------------------------------
    # Generate report
    # --------------------------------------------------------

    lines = []

    lines.append("NYC TAXI ETA - DRIFT MONITORING REPORT")
    lines.append("=" * 55)
    lines.append("")

    lines.append(
        f"Baseline rows: {len(baseline):,}"
    )

    lines.append(
        f"Simulated current rows: {len(current):,}"
    )

    lines.append("")

    lines.append("PSI THRESHOLDS")
    lines.append("-" * 55)
    lines.append(
        "< 0.10       : No significant drift"
    )
    lines.append(
        "0.10 - 0.25  : Moderate drift"
    )
    lines.append(
        "> 0.25       : Significant drift"
    )

    lines.append("")
    lines.append("FEATURE RESULTS")
    lines.append("-" * 55)

    for _, row in results_df.iterrows():

        lines.append(
            f"{row['feature']:20s} "
            f"PSI={row['psi']:.4f} "
            f"{row['status']}"
        )

    lines.append("")
    lines.append("RETRAINING DECISION")
    lines.append("-" * 55)

    if retraining_required:
        lines.append(
            "RETRAINING REQUIRED: Significant feature drift detected."
        )
    else:
        lines.append(
            "NO RETRAINING REQUIRED: No significant drift detected."
        )

    lines.append("")
    lines.append("DRIFT SIMULATION")
    lines.append("-" * 55)
    lines.append(
        "The simulated production data represents a change "
        "in NYC traffic conditions."
    )
    lines.append(
        "The simulation increases trip distances and "
        "rush-hour activity."
    )

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("")
    print("\n".join(lines))
    print("")
    print(f"Saved drift report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()