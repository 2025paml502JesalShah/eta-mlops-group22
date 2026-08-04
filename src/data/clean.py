"""
Data Cleaning Module

This module cleans the NYC Taxi dataset and saves the cleaned
dataset to data/processed/NYC_clean.csv.
"""
import pandas as pd
from pathlib import Path

from src.data.ingest import ingest_data
from src.utils.config import load_params
from src.utils.logger import logger


def clean_data():
    """Clean the raw dataset and save the processed dataset."""

    logger.info("=" * 60)
    logger.info("DATA CLEANING STARTED")
    logger.info("=" * 60)

    # -----------------------------
    # Load dataset
    # -----------------------------
    df = ingest_data()

    initial_rows = len(df)

    # -----------------------------
    # Convert datetime columns
    # -----------------------------
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])

    # -----------------------------
    # Remove duplicate rows
    # -----------------------------
    duplicate_rows = df.duplicated().sum()
    df = df.drop_duplicates()

    # -----------------------------
    # Remove rows with missing values
    # -----------------------------
    rows_before = len(df)
    df = df.dropna()
    missing_removed = rows_before - len(df)

    # -----------------------------
    # Remove invalid passenger counts
    # -----------------------------
    rows_before = len(df)
    df = df[
        (df["passenger_count"] >= 1) &
        (df["passenger_count"] <= 6)
    ]
    passenger_removed = rows_before - len(df)

    # -----------------------------
    # Remove invalid latitude values
    # -----------------------------
    rows_before = len(df)
    df = df[
        df["pickup_latitude"].between(-90, 90) &
        df["dropoff_latitude"].between(-90, 90)
    ]
    latitude_removed = rows_before - len(df)

    # -----------------------------
    # Remove invalid longitude values
    # -----------------------------
    rows_before = len(df)
    df = df[
        df["pickup_longitude"].between(-180, 180) &
        df["dropoff_longitude"].between(-180, 180)
    ]
    longitude_removed = rows_before - len(df)

    # -----------------------------
    # Final statistics
    # -----------------------------
    final_rows = len(df)
    total_removed = initial_rows - final_rows

    # -----------------------------
    # Save cleaned dataset
    # -----------------------------
    params = load_params()

    output_path = Path(params["dataset"]["processed_data_path"])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    # -----------------------------
    # Logging Summary
    # -----------------------------
    logger.info("=" * 60)
    logger.info("DATA CLEANING SUMMARY")
    logger.info("=" * 60)

    logger.info("Transformations Applied:")
    logger.info("-" * 60)
    logger.info("✓ Converted pickup_datetime to datetime")
    logger.info("✓ Converted dropoff_datetime to datetime")
    logger.info("✓ Removed duplicate rows")
    logger.info("✓ Removed rows with missing values")
    logger.info("✓ Filtered passenger_count between 1 and 6")
    logger.info("✓ Validated latitude range (-90 to 90)")
    logger.info("✓ Validated longitude range (-180 to 180)")

    logger.info("-" * 60)
    logger.info("Cleaning Statistics")
    logger.info("-" * 60)

    logger.info(f"Initial Rows               : {initial_rows}")
    logger.info(f"Duplicate Rows Removed     : {duplicate_rows}")
    logger.info(f"Rows Removed (Missing)     : {missing_removed}")
    logger.info(f"Rows Removed (Passenger)   : {passenger_removed}")
    logger.info(f"Rows Removed (Latitude)    : {latitude_removed}")
    logger.info(f"Rows Removed (Longitude)   : {longitude_removed}")
    logger.info(f"Final Rows                 : {final_rows}")
    logger.info(f"Total Rows Removed         : {total_removed}")

    logger.info(f"Cleaned Dataset Saved To   : {output_path}")

    logger.info("=" * 60)
    logger.info("DATA CLEANING COMPLETED")
    logger.info("=" * 60)

    # -----------------------------
    # Cleaning Report
    # -----------------------------
    report_path = Path(params["cleaning"]["report_path"])

    with open(report_path, "w", encoding="utf-8") as report:

        report.write("=" * 60 + "\n")
        report.write("NYC TAXI DATA CLEANING REPORT\n")
        report.write("=" * 60 + "\n\n")

        report.write("Transformations Applied\n")
        report.write("-" * 60 + "\n")
        report.write("✓ Converted pickup_datetime to datetime\n")
        report.write("✓ Converted dropoff_datetime to datetime\n")
        report.write("✓ Removed duplicate rows\n")
        report.write("✓ Removed rows with missing values\n")
        report.write("✓ Filtered passenger_count between 1 and 6\n")
        report.write("✓ Validated latitude range (-90 to 90)\n")
        report.write("✓ Validated longitude range (-180 to 180)\n\n")

        report.write("Cleaning Statistics\n")
        report.write("-" * 60 + "\n")

        report.write("Dataset Information\n")
        report.write("-" * 60 + "\n")
        report.write("Dataset : NYC Taxi Trip Duration\n")
        report.write("Source  : Kaggle\n\n")

        report.write(f"Initial Rows               : {initial_rows}\n")
        report.write(f"Duplicate Rows Removed     : {duplicate_rows}\n")
        report.write(f"Rows Removed (Missing)     : {missing_removed}\n")
        report.write(f"Rows Removed (Passenger)   : {passenger_removed}\n")
        report.write(f"Rows Removed (Latitude)    : {latitude_removed}\n")
        report.write(f"Rows Removed (Longitude)   : {longitude_removed}\n")
        report.write(f"\nFinal Rows                 : {final_rows}\n")
        report.write(f"Total Rows Removed         : {total_removed}\n\n")

        report.write("Processed Dataset\n")
        report.write("-" * 60 + "\n")
        report.write(f"{output_path}\n\n")

        report.write("\n")
        report.write("=" * 60 + "\n")
        report.write("Cleaning Completed Successfully.\n")
        report.write("=" * 60 + "\n")

    logger.info(f"Cleaning report saved to: {report_path}")

    print("\nData cleaning completed successfully.")
    print(f"Clean dataset saved at: {output_path}")
    print(f"Cleaning report saved at: {report_path}")


if __name__ == "__main__":
    clean_data()