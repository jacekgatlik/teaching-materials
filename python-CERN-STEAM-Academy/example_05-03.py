from pathlib import Path
import pandas as pd

project_root = Path("lecture05_pipeline_demo")
raw_dir = project_root / "data" / "raw"

# Load raw tables
events = pd.read_csv(
    raw_dir / "events.csv",
    dtype={"run_id": "string", "detector": "category"},
)

runs = pd.read_csv(
    raw_dir / "runs.csv",
    dtype={"run_id": "string"},
)

# Simulate an accidental duplicated metadata row
duplicated_row = runs.iloc[[0]].copy()
duplicated_row["temperature_K"] = 999.0  # obviously wrong value

bad_runs = pd.concat([runs, duplicated_row], ignore_index=True)

print("Original runs:", len(runs))
print("Bad runs:", len(bad_runs))

# Unsafe merge: it works, but silently duplicates events
unsafe = events.merge(
    bad_runs,
    on="run_id",
    how="left",
)

print("\nNumber of events before merge:", len(events))
print("Number of rows after unsafe merge:", len(unsafe))

# Safe merge: validate our expectation
try:
    safe = events.merge(
        bad_runs,
        on="run_id",
        how="left",
        validate="many_to_one",
    )
except pd.errors.MergeError as err:
    print("\nValidation caught a problem:")
    print(err)

# Diagnose duplicated keys
duplicates = bad_runs.loc[
    bad_runs["run_id"].duplicated(keep=False),
    ["run_id", "field_T", "temperature_K"],
]

print("\nDuplicated metadata rows:")
print(duplicates)

# One possible repair: remove duplicated metadata
fixed_runs = (
    bad_runs.sort_values("temperature_K")
            .drop_duplicates("run_id", keep="first")
)

fixed = events.merge(
    fixed_runs,
    on="run_id",
    how="left",
    validate="many_to_one",
    indicator=True,
)

print("\nMerge status after repair:")
print(fixed["_merge"].value_counts())