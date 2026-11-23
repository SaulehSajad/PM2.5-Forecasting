from pathlib import Path
import pandas as pd


# STEP 1: Import libraries and locate project folders
ROOT = Path(__file__).resolve().parent.parent

RAW = ROOT / "data" / "raw" / (
    "PRSA_Data_Aotizhongxin_20130301-20170228.csv"
)

OUT = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


# STEP 2: Load the original dataset without changing it
if not RAW.is_file():
    raise FileNotFoundError(f"CSV not found: {RAW}")

df = pd.read_csv(RAW)

print("Rows, columns:", df.shape)
print("Column names:", df.columns.tolist())


# STEP 3: Build timestamps and inspect time order
time_parts = df[["year", "month", "day", "hour"]]
df["timestamp"] = pd.to_datetime(time_parts, errors="coerce")

print("Invalid timestamps:", df["timestamp"].isna().sum())
print("Repeated timestamps:", df["timestamp"].duplicated().sum())
print(
    "Already chronological:",
    df["timestamp"].is_monotonic_increasing
)


# STEP 4: Measure missing values and exact duplicate rows
missing_count = df.isna().sum().sort_values(ascending=False)
missing_percent = (df.isna().mean() * 100).round(2)
exact_duplicates = df.duplicated().sum()

print("\nMissing values by column:")
print(missing_count)

print("\nMissing percentages:")
print(missing_percent)

print("\nExact duplicate rows:", exact_duplicates)


# STEP 5: Flag negative pollutant measurements
pollutants = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
negative_counts = {}

for name in pollutants:
    if name in df.columns:
        negative_counts[name] = int((df[name] < 0).sum())

print("\nNegative pollutant values:", negative_counts)


# STEP 6: Apply conservative cleaning rules
# 1. Remove exact duplicate rows.
# 2. Remove rows with invalid timestamps.
# 3. Sort rows chronologically.
# 4. Convert negative pollutant values to missing values.
clean = df.drop_duplicates().copy()
clean = clean.dropna(subset=["timestamp"]).copy()
clean = clean.sort_values("timestamp").reset_index(drop=True)

for name in pollutants:
    if name in clean.columns:
        clean.loc[clean[name] < 0, name] = float("nan")

remaining_repeated_times = int(
    clean["timestamp"].duplicated().sum()
)

print(
    "Repeated timestamps after cleaning:",
    remaining_repeated_times
)


# STEP 7: Check for missing hourly timestamps
unique_times = pd.DatetimeIndex(
    clean["timestamp"].dropna().unique()
)

if len(unique_times) > 0:
    expected_hours = pd.date_range(
        start=unique_times.min(),
        end=unique_times.max(),
        freq="h"
    )
    missing_hours = expected_hours.difference(unique_times)
else:
    missing_hours = pd.DatetimeIndex([])

print("Missing hourly timestamps:", len(missing_hours))


# STEP 8: Save cleaned data and the quality report
clean_path = OUT / "aotizhongxin_quality_checked.csv"
report_path = REPORTS / "practical03_quality_report.txt"

clean.to_csv(clean_path, index=False)

lines = [
    "Practical 03: Aotizhongxin Data Quality Report",
    f"Original rows: {len(df)}",
    f"Cleaned rows: {len(clean)}",
    f"Exact duplicate rows in original: {exact_duplicates}",
    f"Invalid original timestamps: {df['timestamp'].isna().sum()}",
    f"Repeated timestamps remaining: {remaining_repeated_times}",
    f"Missing hourly timestamps: {len(missing_hours)}",
    "",
    "Original missing counts:",
    missing_count.to_string(),
    "",
    "Negative pollutant counts:",
    str(negative_counts),
    "",
    "Cleaning policy: no missing-value imputation performed.",
]

report_path.write_text("\n".join(lines), encoding="utf-8")

print("\nSaved:", clean_path)
print("Saved:", report_path)


# OPTIONAL: Exercise 3 from the manual
print(
    "Missing PM2.5 after cleaning:",
    clean["PM2.5"].isna().sum()
)
