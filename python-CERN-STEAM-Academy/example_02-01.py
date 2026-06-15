import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=123)

t = np.linspace(0.0, 12.0, 500)

A = 1.4
gamma = 0.22
omega = 3.2
phase = 0.3

# Exact/reference signal
model = A * np.exp(-gamma * t) * np.sin(omega * t + phase)

# Measurement uncertainty increases slightly with time
sigma = 0.04 + 0.025 * (t / t.max())

# Synthetic measured data
data = model + sigma * rng.normal(size=t.size)

# Residual and standardized residual
residual = data - model
z_residual = residual / sigma

rms = np.sqrt(np.mean(residual**2))
z_mean = np.mean(z_residual)
z_std = np.std(z_residual)

print(f"RMS residual: {rms:.4f}")
print(f"Mean standardized residual: {z_mean:.4f}")
print(f"Std standardized residual:  {z_std:.4f}")

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.0, 5.0))

# Data and model
axes[0].plot(t, data, ".", markersize=3, label="synthetic data")
axes[0].plot(t, model, linewidth=2, label="reference model")
axes[0].fill_between(t, model - sigma, model + sigma, alpha=0.25, label=r"$\pm 1\sigma$")

axes[0].set_ylabel(r"signal $s(t)$")
axes[0].set_title("Data, reference model and uncertainty band")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Standardized residual
axes[1].axhline(0.0, linewidth=1)
axes[1].axhline(1.0, linestyle="--", linewidth=1)
axes[1].axhline(-1.0, linestyle="--", linewidth=1)
axes[1].plot(t, z_residual, ".", markersize=3)

axes[1].set_xlabel(r"time $t$")
axes[1].set_ylabel(r"$(data-model)/\sigma$")
axes[1].set_title("Standardized residual")
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
plt.show()