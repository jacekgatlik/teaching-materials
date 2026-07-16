from pathlib import Path
import json
import numpy as np
import pandas as pd

# Portable project layout
project_root = Path("lecture05_pipeline_demo")

raw_dir = project_root / "data" / "raw"
processed_dir = project_root / "data" / "processed"
table_dir = project_root / "results" / "tables"
figure_dir = project_root / "results" / "figures"

for directory in [raw_dir, processed_dir, table_dir, figure_dir]:
    directory.mkdir(parents=True, exist_ok=True)

# Synthetic run-level metadata
rng = np.random.default_rng(123)

n_runs = 8
run_ids = [f"run_{i:03d}" for i in range(n_runs)]

runs = pd.DataFrame({
    "run_id": run_ids,
    "field_T": rng.uniform(0.1, 1.2, size=n_runs),
    "temperature_K": 300.0 + rng.normal(0.0, 2.0, size=n_runs),
    "mode": rng.choice(["calibration", "physics"], size=n_runs, p=[0.25, 0.75]),
})

# Synthetic event-level measurements
n_events = 2500

events = pd.DataFrame({
    "event_id": np.arange(n_events),
    "run_id": rng.choice(run_ids, size=n_events),
    "detector": rng.choice(["A", "B", "C"], size=n_events, p=[0.40, 0.35, 0.25]),
    "energy_GeV": rng.gamma(shape=2.0, scale=2.0, size=n_events),
    "chi2": rng.chisquare(df=2.0, size=n_events),
    "exposure_s": rng.uniform(0.8, 1.2, size=n_events),
})

# The signal depends on event energy, run field, detector gain, and noise.
field_by_run = dict(zip(runs["run_id"], runs["field_T"]))
gain_by_detector = {"A": 1.00, "B": 0.92, "C": 1.08}

field_per_event = events["run_id"].map(field_by_run).to_numpy()
gain_per_event = events["detector"].map(gain_by_detector).to_numpy()

noise = rng.normal(0.0, 0.5, size=n_events)

events["signal"] = (
    gain_per_event
    * (4.0 + 1.3 * np.sqrt(events["energy_GeV"]) + 2.0 * field_per_event)
    + noise
)

# Introduce a few realistic data problems
bad_idx = rng.choice(n_events, size=30, replace=False)

events.loc[bad_idx[:10], "signal"] = np.nan        # failed measurement
events.loc[bad_idx[10:20], "energy_GeV"] = -1.0    # unphysical energy
events.loc[bad_idx[20:], "exposure_s"] = 0.0       # invalid exposure

# Save raw data and configuration
config = {
    "min_energy_GeV": 1.0,
    "max_chi2": 6.0,
    "random_seed": 123,
}

events_path = raw_dir / "events.csv"
runs_path = raw_dir / "runs.csv"
config_path = raw_dir / "config.json"

events.to_csv(events_path, index=False)
runs.to_csv(runs_path, index=False)
config_path.write_text(json.dumps(config, indent=2), encoding="utf8")

print(f"Created: {events_path}")
print(f"Created: {runs_path}")
print(f"Created: {config_path}")

print("\nEvents:")
print(events.head())

print("\nRuns:")
print(runs.head())