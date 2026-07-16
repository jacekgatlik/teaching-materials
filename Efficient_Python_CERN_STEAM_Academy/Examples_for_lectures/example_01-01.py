import numpy as np
import matplotlib.pyplot as plt

# Grid parameters
nx, ny = 300, 220
x = np.linspace(-5.0, 5.0, nx)
y = np.linspace(-4.0, 4.0, ny)

dx = x[1] - x[0]
dy = y[1] - y[0]

# X[i, j] = x[i], Y[i, j] = y[j]
X, Y = np.meshgrid(x, y, indexing="ij")

# Parameters of the Gaussian
x0, y0 = 1.0, -0.7
sigma_x, sigma_y = 1.0, 0.6

# Two-dimensional Gaussian field u(x, y)
u = np.exp(
    -0.5 * ((X - x0) / sigma_x)**2
    -0.5 * ((Y - y0) / sigma_y)**2
)

# Integral over space: sum over both spatial axes
mass = np.sum(u) * dx * dy

# Center of mass
x_cm = np.sum(X * u) * dx * dy / mass
y_cm = np.sum(Y * u) * dx * dy / mass

# Location of the maximum
i_max, j_max = np.unravel_index(np.argmax(u), u.shape)
x_max = x[i_max]
y_max = y[j_max]

print(f"Integral:       {mass:.6f}")
print(f"Center of mass: ({x_cm:.6f}, {y_cm:.6f})")
print(f"Maximum at:     ({x_max:.6f}, {y_max:.6f})")

# Visualization
plt.figure(figsize=(6, 4))
plt.pcolormesh(X, Y, u, shading="auto")
plt.plot(x_cm, y_cm, "o", label="center of mass")
plt.plot(x_max, y_max, "x", label="maximum")
plt.colorbar(label="u(x, y)")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.legend()
plt.tight_layout()
plt.show()