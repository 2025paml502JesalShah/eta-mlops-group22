"""
Data Validation Module

This module validates the raw NYC Taxi dataset and generates
a validation report under reports/validation_report.txt.
"""

from pathlib import Path

from src.data.ingest import ingest_data
from src.utils.config import load_params
from src.utils.logger import logger


def validate_data():
    """Validate the raw dataset and generate a report."""

    logger.info("========== DATA VALIDATION STARTED ==========")

    # Load dataset
    df = ingest_data()

    # Load configuration
    params = load_params()

    # Validation report location
    report_path = Path(params["validation"]["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Basic statistics
    rows = df.shape[0]
    cols = df.shape[1]

    # Missing values
    missing_values = df.isnull().sum()

    # Duplicate rows
    duplicate_rows = df.duplicated().sum()

    # Duplicate IDs
    duplicate_ids = df["id"].duplicated().sum()

    # Passenger Count Validation
    invalid_passenger_count = (
        (df["passenger_count"] <= 0) |
        (df["passenger_count"] > 6)
    ).sum()

    # Latitude Validation
    invalid_pickup_latitude = (
        (df["pickup_latitude"] < -90) |
        (df["pickup_latitude"] > 90)
    ).sum()

    invalid_dropoff_latitude = (
        (df["dropoff_latitude"] < -90) |
        (df["dropoff_latitude"] > 90)
    ).sum()

    # Longitude Validation
    invalid_pickup_longitude = (
        (df["pickup_longitude"] < -180) |
        (df["pickup_longitude"] > 180)
    ).sum()

    invalid_dropoff_longitude = (
        (df["dropoff_longitude"] < -180) |
        (df["dropoff_longitude"] > 180)
    ).sum()

    # Generate Report
    with open(report_path, "w", encoding="utf-8") as report:

        report.write("=" * 50 + "\n")
        report.write("NYC TAXI DATA VALIDATION REPORT\n")
        report.write("=" * 50 + "\n\n")

        report.write(f"Rows                : {rows}\n")
        report.write(f"Columns             : {cols}\n\n")

        report.write("Missing Values\n")
        report.write("-" * 50 + "\n")
        report.write(missing_values.to_string())
        report.write("\n\n")

        report.write(f"Duplicate Rows      : {duplicate_rows}\n")
        report.write(f"Duplicate IDs       : {duplicate_ids}\n\n")

        report.write("Validation Checks\n")
        report.write("-" * 50 + "\n")
        report.write(f"Invalid Passenger Count : {invalid_passenger_count}\n")
        report.write(f"Invalid Pickup Latitude : {invalid_pickup_latitude}\n")
        report.write(f"Invalid Dropoff Latitude: {invalid_dropoff_latitude}\n")
        report.write(f"Invalid Pickup Longitude: {invalid_pickup_longitude}\n")
        report.write(f"Invalid Dropoff Longitude: {invalid_dropoff_longitude}\n")

    logger.info("Validation report generated successfully.")
    logger.info(f"Validation report saved at: {report_path}")
    logger.info("========== DATA VALIDATION COMPLETED ==========")

    print("\nData validation completed successfully.")
    print(f"Validation report saved at: {report_path}")


if __name__ == "__main__":
    validate_data()