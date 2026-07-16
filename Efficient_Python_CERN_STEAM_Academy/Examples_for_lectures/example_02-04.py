import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Synthetic travelling front:
# u(t,x) = tanh((x - X_front(t)) / width(t))
# ------------------------------------------------------------

x = np.linspace(-10.0, 10.0, 700)
t = np.linspace(0.0, 10.0, 400)

T, X = np.meshgrid(t, x, indexing="ij")

# Front with a slowly modulated trajectory
X_front_exact = -6.0 + 1.15 * t + 0.35 * np.sin(2 * np.pi * t / 5.0)
width = 0.55 + 0.08 * np.sin(2 * np.pi * t / 4.0)

U = np.tanh((X - X_front_exact[:, None]) / width[:, None])

front_tracked = np.empty(t.size)

for i, row in enumerate(U):
    # Find where the sign changes
    candidates = np.where(np.signbit(row[:-1]) != np.signbit(row[1:]))[0]

    if candidates.size == 0:
        front_tracked[i] = np.nan
        continue

    j = candidates[0]

    # Linear interpolation:
    # row[j] + alpha * (row[j+1] - row[j]) = 0
    alpha = -row[j] / (row[j + 1] - row[j])
    front_tracked[i] = x[j] + alpha * (x[j + 1] - x[j])

# Numerical velocity of the tracked front
velocity = np.gradient(front_tracked, t)

tracking_error = front_tracked - X_front_exact

print(f"Max tracking error: {np.nanmax(np.abs(tracking_error)):.3e}")

fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.0), sharex=True)

# Space-time image
im = axes[0].imshow(
    U.T,
    extent=[t.min(), t.max(), x.min(), x.max()],
    origin="lower",
    aspect="auto",
    cmap="RdBu_r",
    vmin=-1.0,
    vmax=1.0
)

axes[0].plot(t, front_tracked, "--", linewidth=2, label="tracked front")
axes[0].plot(t, X_front_exact, linewidth=1, label="exact front")

axes[0].set_ylabel(r"$x$")
axes[0].set_title(r"Space-time plot of a travelling front")
axes[0].legend()

fig.colorbar(im, ax=axes[0], label=r"$u(t,x)$")

# Velocity of the tracked feature
axes[1].plot(t, velocity)
axes[1].set_xlabel(r"time $t$")
axes[1].set_ylabel(r"$dX/dt$")
axes[1].set_title("Velocity estimated from tracked front position")
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
plt.show()