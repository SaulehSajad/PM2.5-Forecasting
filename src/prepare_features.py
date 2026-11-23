from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error


# Project paths
ROOT = Path(__file__).resolve().parent.parent
# Practical 03 in this project creates this file.
# If your cleaned CSV has a different name, change this path.
INPUT = ROOT / "data" / "processed" / "aotizhongxin_quality_checked.csv"
OUTPUT = ROOT / "data" / "processed"
OUTPUT.mkdir(parents=True, exist_ok=True)


# Stage A: Load the quality-checked dataset
if not INPUT.is_file():
    raise FileNotFoundError(
        f"Cleaned input not found: {INPUT}\n"
        "Copy your Practical 03 cleaned CSV into data/processed, "
        "or edit INPUT in this script to match its exact filename."
    )

df = pd.read_csv(INPUT)
print("Loaded rows:", len(df))
print("Columns:", df.columns.tolist())


# Stage B: Construct and validate timestamps
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
else:
    required = ["year", "month", "day", "hour"]
    if not all(col in df.columns for col in required):
        raise ValueError("Need timestamp or year/month/day/hour columns")
    df["timestamp"] = pd.to_datetime(df[required], errors="coerce")

if df["timestamp"].isna().any():
    raise ValueError("Some timestamps cannot be interpreted")

df = df.sort_values("timestamp").reset_index(drop=True)

if df["timestamp"].duplicated().any():
    raise ValueError("Duplicate timestamps: review Practical 03 cleaning")

intervals = df["timestamp"].diff().dropna()
if not intervals.eq(pd.Timedelta(hours=1)).all():
    raise ValueError(
        "Hourly gaps detected; inspect missing timestamps and repair or "
        "document them before shifting. Do not blindly fill missing hours."
    )

print("Hourly timestamp sequence verified")


# Stage C: Create features and the one-hour-ahead target
if "PM2.5" not in df.columns:
    raise KeyError("Required column 'PM2.5' was not found")

df["PM2.5"] = pd.to_numeric(df["PM2.5"], errors="coerce")
df["pm25_lag1"] = df["PM2.5"].shift(1)
df["pm25_lag2"] = df["PM2.5"].shift(2)
df["hour_of_day"] = df["timestamp"].dt.hour
df["target_pm25_next_hour"] = df["PM2.5"].shift(-1)

feature_cols = ["PM2.5", "pm25_lag1", "pm25_lag2", "hour_of_day"]
target_col = "target_pm25_next_hour"

model_df = df[["timestamp"] + feature_cols + [target_col]].dropna().copy()
print("Usable supervised rows:", len(model_df))
print(model_df.head(3).to_string(index=False))

if len(model_df) < 4:
    raise ValueError("Not enough usable rows to create non-empty chronological splits")


# Stage D: Split chronologically (70% train, 15% validation, 15% test)
n = len(model_df)
train_end = int(n * 0.70)
valid_end = int(n * 0.85)

train = model_df.iloc[:train_end].copy()
valid = model_df.iloc[train_end:valid_end].copy()
test = model_df.iloc[valid_end:].copy()

assert len(train) > 0 and len(valid) > 0 and len(test) > 0
assert train["timestamp"].max() < valid["timestamp"].min()
assert valid["timestamp"].max() < test["timestamp"].min()


# Stage E: Remove rows whose next-hour target crosses a split boundary
train = train.iloc[:-1].copy()
valid = valid.iloc[:-1].copy()

assert len(train) > 0 and len(valid) > 0 and len(test) > 0

for name, part in [
    ("train", train),
    ("validation", valid),
    ("test", test),
]:
    part.to_csv(OUTPUT / f"{name}_features.csv", index=False)
    print(
        name,
        "rows:", len(part),
        "start:", part["timestamp"].min(),
        "end:", part["timestamp"].max(),
    )


# Stage F: Evaluate the persistence baseline on the test set
baseline_predictions = test["PM2.5"]
actual_values = test[target_col]
mae = mean_absolute_error(actual_values, baseline_predictions)
print(f"Test persistence MAE: {mae:.2f}")

summary_path = OUTPUT / "split_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(f"Training rows: {len(train)}\n")
    f.write(f"Validation rows: {len(valid)}\n")
    f.write(f"Test rows: {len(test)}\n")
    f.write(f"Training start: {train['timestamp'].min()}\n")
    f.write(f"Training end: {train['timestamp'].max()}\n")
    f.write(f"Validation start: {valid['timestamp'].min()}\n")
    f.write(f"Validation end: {valid['timestamp'].max()}\n")
    f.write(f"Test start: {test['timestamp'].min()}\n")
    f.write(f"Test end: {test['timestamp'].max()}\n")
    f.write(f"Persistence test MAE: {mae:.2f}\n")

print("Saved split files and summary in:", OUTPUT)
