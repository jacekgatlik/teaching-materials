from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

project_root = Path("lecture05_pipeline_demo")

processed_dir = project_root / "data" / "processed"
table_dir = project_root / "results" / "tables"
figure_dir = project_root / "results" / "figures"

table_dir.mkdir(parents=True, exist_ok=True)
figure_dir.mkdir(parents=True, exist_ok=True)



# Load the clean table
clean = pd.read_csv(
    processed_dir / "clean_events.csv",
    dtype={
        "run_id": "string",
        "detector": "category",
    },
)

energy_order = ["low", "medium", "high", "very_high"]

clean["energy_bin"] = pd.Categorical(
    clean["energy_bin"],
    categories=energy_order,
    ordered=True,
)

# Group by detector and energy bin
summary = (
    clean.groupby(["detector", "energy_bin"],
                  as_index=False,
                  observed=True)
         .agg(
             n_events=("signal_rate", "size"),
             mean_rate=("signal_rate", "mean"),
             std_rate=("signal_rate", "std"),
             mean_energy_GeV=("energy_GeV", "mean"),
         )
)

summary["sem_rate"] = summary["std_rate"] / np.sqrt(summary["n_events"])

summary_path = table_dir / "summary_by_detector_and_energy.csv"
summary.to_csv(summary_path, index=False)

print(f"Saved summary table to: {summary_path}")
print(summary)

# Pivot table for quick inspection
pivot = summary.pivot_table(
    index="energy_bin",
    columns="detector",
    values="mean_rate",
    observed=True,
)

print("\nMean signal rate as labelled grid:")
print(pivot)

# Plot labelled summary
fig, ax = plt.subplots(figsize=(4))

x_positions = {label: i for i, label in enumerate(energy_order)}

for detector, group in summary.groupby("detector", observed=True):
    group = group.sort_values("energy_bin")

    x = [x_positions[str(label)] for label in group["energy_bin"]]

    ax.errorbar(
        x,
        group["mean_rate"],
        yerr=group["sem_rate"],
        fmt="o-",
        capsize=3,
        label=f"detector {detector}",
    )

ax.set_xticks(range(len(energy_order)))
ax.set_xticklabels(energy_order)

ax.set(
    xlabel="energy bin",
    ylabel="mean signal rate",
    title="Signal rate by detector and energy range",
)

ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()

figure_path = figure_dir / "signal_rate_by_detector_and_energy.pdf"
fig.savefig(figure_path, bbox_inches="tight")

print(f"\nSaved figure to: {figure_path}")

plt.show()