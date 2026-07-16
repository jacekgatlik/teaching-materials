from pathlib import Path
import json
import numpy as np
import pandas as pd

project_root = Path("lecture05_pipeline_demo")

processed_dir = project_root / "data" / "processed"
table_dir = project_root / "results" / "tables"
array_dir = project_root / "results" / "arrays"

table_dir.mkdir(parents=True, exist_ok=True)
array_dir.mkdir(parents=True, exist_ok=True)

# Load the clean table
clean = pd.read_csv(
    processed_dir / "clean_events.csv",
    dtype={
        "run_id": "string",
        "detector": "category",
        "mode": "category",
    },
)

# Split by run_id, not by individual rows
# This avoids putting events from the same run into both train and test sets.
unique_runs = (
    clean["run_id"]
    .drop_duplicates()
    .sample(frac=1.0, random_state=123)
    .to_numpy()
)

n_train = int(0.70 * len(unique_runs))
n_val = int(0.15 * len(unique_runs))

train_runs = set(unique_runs[:n_train])
val_runs = set(unique_runs[n_train:n_train + n_val])
test_runs = set(unique_runs[n_train + n_val:])

clean["split"] = "test"
clean.loc[clean["run_id"].isin(train_runs), "split"] = "train"
clean.loc[clean["run_id"].isin(val_runs), "split"] = "val"

train_mask = clean["split"] == "train"

# Define a simple binary target
# The threshold is computed from training data only.
rate_threshold = clean.loc[train_mask, "signal_rate"].quantile(0.75)

clean["large_rate"] = clean["signal_rate"] > rate_threshold

# Build a feature table
numeric_features = [
    "energy_GeV",
    "chi2",
    "field_T",
    "temperature_K",
    "exposure_s",
]

detector_features = pd.get_dummies(
    clean["detector"],
    prefix="detector",
    dtype=float,
)

feature_table = pd.concat(
    [clean[numeric_features], detector_features],
    axis=1,
)

feature_cols = feature_table.columns.tolist()

# Scale numerical features using training data only
mu = feature_table.loc[train_mask, numeric_features].mean()
scale = feature_table.loc[train_mask, numeric_features].std(ddof=0)
scale = scale.replace(0.0, 1.0)

X_table = feature_table.copy()
X_table.loc[:, numeric_features] = (
    X_table[numeric_features] - mu
) / scale

# Convert labelled tables to NumPy arrays
arrays = {}

for split in ["train", "val", "test"]:
    mask = clean["split"] == split

    arrays[f"X_{split}"] = X_table.loc[mask].to_numpy(dtype=float)
    arrays[f"y_{split}"] = clean.loc[mask, "large_rate"].to_numpy(dtype=bool)

np.savez(array_dir / "ml_ready_data.npz", **arrays)

# Save metadata needed to interpret the arrays
feature_info = pd.DataFrame({
    "feature": feature_cols,
    "kind": [
        "numeric" if col in numeric_features else "one_hot"
        for col in feature_cols
    ],
})

scaling_info = pd.DataFrame({
    "feature": numeric_features,
    "mean": mu.to_numpy(),
    "scale": scale.to_numpy(),
})

feature_info.to_csv(table_dir / "feature_columns.csv", index=False)
scaling_info.to_csv(table_dir / "feature_scaling.csv", index=False)

target_info = {
    "target": "large_rate",
    "definition": "signal_rate greater than the 75th percentile of the training set",
    "rate_threshold": float(rate_threshold),
}

(table_dir / "target_definition.json").write_text(
    json.dumps(target_info, indent=2),
    encoding="utf8",
)

print("Saved ML-ready arrays and metadata.")
print("Feature columns:")
print(feature_info)