
from pathlib import Path

import pandas as pd

# Find the project directory and input files.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "raw" / "DataCoSupplyChainDataset.csv"
DICTIONARY_FILE = PROJECT_DIR / "data" / "raw" / "DescriptionDataCoSupplyChain.csv"


def main():
    print("=" * 60)
    print("SUPPLY CHAIN DATA PROFILING")
    print("=" * 60)

    # Check that the main dataset exists.
    if not DATA_FILE.exists():
        print(f"ERROR: Dataset not found at {DATA_FILE}")
        return

    print("\n1. Loading dataset...")

    # latin1 helps read this dataset if it contains non-UTF-8 characters.
    df = pd.read_csv(DATA_FILE, encoding="latin1", low_memory=False)

    print("\n2. Dataset dimensions")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")

    print("\n3. Column names")
    for column in df.columns:
        print(f"- {column}")

    print("\n4. First five rows")
    print(df.head().to_string())

    print("\n5. Data types")
    print(df.dtypes.to_string())

    print("\n6. Missing values")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("No missing values detected.")
    else:
        print(missing.to_string())

    print("\n7. Duplicate rows")
    print(f"Exact duplicate rows: {df.duplicated().sum():,}")

    print("\n8. Data dictionary")
    if DICTIONARY_FILE.exists():
        dictionary = pd.read_csv(
            DICTIONARY_FILE,
            encoding="latin1",
            low_memory=False,
        )
        print(f"Dictionary dimensions: {dictionary.shape}")
        print(dictionary.head(10).to_string(index=False))
    else:
        print("Dictionary file not found.")

    # Save a concise profiling report.
    report_dir = PROJECT_DIR / "docs"
    report_dir.mkdir(exist_ok=True)

    report_file = report_dir / "data_profile.txt"

    with report_file.open("w", encoding="utf-8") as report:
        report.write("SUPPLY CHAIN DATA PROFILING REPORT\n")
        report.write("=" * 40 + "\n")
        report.write(f"Rows: {df.shape[0]:,}\n")
        report.write(f"Columns: {df.shape[1]:,}\n\n")

        report.write("COLUMN DATA TYPES\n")
        report.write(df.dtypes.to_string())
        report.write("\n\nMISSING VALUES\n")

        if missing.empty:
            report.write("No missing values detected.\n")
        else:
            report.write(missing.to_string())

        report.write(f"\n\nExact duplicate rows: {df.duplicated().sum():,}\n")

    print(f"\nReport saved to: {report_file}")
    print("\nData profiling completed!")


if __name__ == "__main__":
    main()
