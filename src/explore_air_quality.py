from pathlib import Path
import pandas as pd


# STEP 1: Locate the project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# STEP 2: Locate the Aotizhongxin station CSV file
CSV_PATH = PROJECT_ROOT / "data" / "raw" / (
    "PRSA_Data_Aotizhongxin_20130301-20170228.csv"
)

# STEP 3: Check that the dataset exists before attempting to load it
if not CSV_PATH.is_file():
    raise FileNotFoundError(
        f"Dataset not found: {CSV_PATH}\n"
        "Place PRSA_Data_Aotizhongxin_20130301-20170228.csv "
        "inside the data/raw folder."
    )

# STEP 4: Load the CSV file into a Pandas DataFrame
df = pd.read_csv(CSV_PATH)

# STEP 5: Display the dataset structure and a preview
print("DATASET LOADED")
print("Rows and columns:", df.shape)
print("Column names:", df.columns.tolist())
print("First five rows:")
print(df.head())

# STEP 6: Inspect the data types
print("\nDATA TYPES")
print(df.dtypes)

# STEP 7: Count missing values in each column
print("\nMISSING VALUES PER COLUMN")
missing_counts = df.isna().sum()
print(missing_counts)

# STEP 8: Count all missing cells
print("\nTOTAL MISSING CELLS")
print(missing_counts.sum())

# STEP 9: Summarize the PM2.5 column
if "PM2.5" not in df.columns:
    raise KeyError(
        "The column 'PM2.5' was not found. "
        f"Available columns: {df.columns.tolist()}"
    )

print("\nPM2.5 SUMMARY")
pm25_summary = df["PM2.5"].describe()
print(pm25_summary)

# STEP 10: Count exact duplicate rows
print("\nNUMBER OF DUPLICATE ROWS")
duplicate_count = df.duplicated().sum()
print(duplicate_count)

# STEP 11: Create a timestamp from the date/time columns
required_time_columns = ["year", "month", "day", "hour"]
missing_time_columns = [
    column for column in required_time_columns if column not in df.columns
]
if missing_time_columns:
    raise KeyError(
        "Required date/time columns are missing: "
        f"{missing_time_columns}"
    )

df["timestamp"] = pd.to_datetime(
    df[required_time_columns],
    errors="raise"
)

# STEP 12: Inspect the time range and row ordering
print("\nTIME RANGE")
print("First timestamp:", df["timestamp"].min())
print("Last timestamp:", df["timestamp"].max())
print(
    "Timestamps sorted:",
    df["timestamp"].is_monotonic_increasing
)

# STEP 13: Save the inspection report
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
report_path = REPORT_DIR / "inspection_report.txt"

with report_path.open("w", encoding="utf-8") as report:
    report.write(f"Dataset: {CSV_PATH.name}\n")
    report.write(f"Shape: {df.shape}\n")
    report.write("\nMissing values by column:\n")
    report.write(missing_counts.to_string())
    report.write(f"\n\nTotal missing cells: {missing_counts.sum()}\n")
    report.write(f"\nDuplicate entire rows: {duplicate_count}\n")
    report.write("\nPM2.5 summary:\n")
    report.write(pm25_summary.to_string())
    report.write("\n\nTime range:\n")
    report.write(f"First timestamp: {df['timestamp'].min()}\n")
    report.write(f"Last timestamp: {df['timestamp'].max()}\n")
    report.write(
        "Timestamps sorted: "
        f"{df['timestamp'].is_monotonic_increasing}\n"
    )

# STEP 14: Confirm where the report was saved
print("\nReport saved to:", report_path)
