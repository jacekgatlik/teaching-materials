import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5.0, 5.0, 300)
y = np.linspace(-3.0, 3.0, 200)

X, Y = np.meshgrid(x, y, indexing="ij")

dx = x[1] - x[0]
dy = y[1] - y[0]

# Two asymmetric Gaussian peaks
rho = (
    1.0 * np.exp(-((X - 1.0)**2 / (2 * 0.9**2)
                   + (Y + 0.4)**2 / (2 * 0.5**2)))
    +
    0.45 * np.exp(-((X + 1.8)**2 / (2 * 0.7**2)
                    + (Y - 0.8)**2 / (2 * 0.9**2)))
)

mass = np.sum(rho) * dx * dy

x_cm = np.sum(X * rho) * dx * dy / mass
y_cm = np.sum(Y * rho) * dx * dy / mass

imax, jmax = np.unravel_index(np.argmax(rho), rho.shape)
x_peak = x[imax]
y_peak = y[jmax]

print(f"Integral / mass:      {mass:.6f}")
print(f"Centre of mass:       ({x_cm:.4f}, {y_cm:.4f})")
print(f"Maximum location:     ({x_peak:.4f}, {y_peak:.4f})")
print(f"Field shape:          {rho.shape}")

fig, ax = plt.subplots(figsize=(7.0, 4.2))

pcm = ax.pcolormesh(X, Y, rho, shading="auto")
cs = ax.contour(X, Y, rho, levels=8, linewidths=0.8)

ax.plot(x_cm, y_cm, "x", markersize=10, label="centre of mass")
ax.plot(x_peak, y_peak, "o", markersize=6, label="maximum")

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_title("Two-dimensional field with diagnostics")
ax.set_aspect("equal")
ax.legend()

fig.colorbar(pcm, ax=ax, label=r"$\rho(x,y)$")
fig.tight_layout()
plt.show()