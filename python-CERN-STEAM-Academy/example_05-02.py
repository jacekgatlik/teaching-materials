from pathlib import Path
import json
import numpy as np
import pandas as pd

project_root = Path("lecture05_pipeline_demo")

raw_dir = project_root / "data" / "raw"
processed_dir = project_root / "data" / "processed"
table_dir = project_root / "results" / "tables"

processed_dir.mkdir(parents=True, exist_ok=True)
table_dir.mkdir(parents=True, exist_ok=True)

# Load configuration
config_path = raw_dir / "config.json"
config = json.loads(config_path.read_text(encoding="utf8"))

# Read raw tables with explicit dtypes
events = pd.read_csv(
    raw_dir / "events.csv",
    dtype={
        "run_id": "string",
        "detector": "category",
    },
)

runs = pd.read_csv(
    raw_dir / "runs.csv",
    dtype={
        "run_id": "string",
        "mode": "category",
    },
)

print("Events shape:", events.shape)
print("Runs shape:", runs.shape)

print("\nEvents dtypes:")
print(events.dtypes)

# Join event-level data with run-level metadata
df = events.merge(
    runs,
    on="run_id",
    how="left",
    validate="many_to_one",
    indicator=True,
)

# Check whether every event found matching run metadata.
if not (df["_merge"] == "both").all():
    missing = df.loc[df["_merge"] != "both", "run_id"].unique()
    raise ValueError(f"Missing metadata for run_id: {missing}")

df = df.drop(columns="_merge")

# Build masks for valid data
numeric_cols = [
    "energy_GeV",
    "chi2",
    "exposure_s",
    "signal",
    "field_T",
    "temperature_K",
]

finite = np.isfinite(df[numeric_cols]).all(axis=1)

physical = (
    (df["energy_GeV"] >= 0.0)
    & (df["chi2"] >= 0.0)
    & (df["exposure_s"] > 0.0)
)

analysis_cuts = (
    (df["energy_GeV"] >= config["min_energy_GeV"])
    & (df["chi2"] <= config["max_chi2"])
)

clean_mask = finite & physical & analysis_cuts

# Save diagnostics before removing rows
diagnostics = pd.Series({
    "n_raw_events": len(df),
    "n_nonfinite_or_missing": int((~finite).sum()),
    "n_unphysical": int((finite & ~physical).sum()),
    "n_failed_analysis_cuts": int((finite & physical & ~analysis_cuts).sum()),
    "n_clean_events": int(clean_mask.sum()),
})

print("\nCleaning diagnostics:")
print(diagnostics)

diagnostics.to_csv(
    table_dir / "cleaning_diagnostics.csv",
    header=["value"],
)

# Create a clean table with derived columns
clean = (
    df.loc[clean_mask]
      .assign(
          signal_rate=lambda d: d["signal"] / d["exposure_s"],
          energy_bin=lambda d: pd.cut(
              d["energy_GeV"],
              bins=[0.0, 2.0, 5.0, 10.0, np.inf],
              labels=["low", "medium", "high", "very_high"],
          ),
      )
      .copy()
)

clean_path = processed_dir / "clean_events.csv"
clean.to_csv(clean_path, index=False)

print(f"\nSaved clean table to: {clean_path}")
print(clean.head())