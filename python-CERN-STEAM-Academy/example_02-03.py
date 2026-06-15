import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Test problem:
# u(x,y) = sin(kx) sin(ly)
#
# It satisfies:
# u_xx + u_yy + (k^2 + l^2) u = 0
# ------------------------------------------------------------

def helmholtz_residual(n, k=2.0, ell=3.0):
    x = np.linspace(0.0, np.pi, n)
    y = np.linspace(0.0, np.pi, n)

    X, Y = np.meshgrid(x, y, indexing="ij")
    U = np.sin(k * X) * np.sin(ell * Y)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Second-order central differences on interior points
    Uxx = (U[2:, 1:-1] - 2 * U[1:-1, 1:-1] + U[:-2, 1:-1]) / dx**2
    Uyy = (U[1:-1, 2:] - 2 * U[1:-1, 1:-1] + U[1:-1, :-2]) / dy**2

    R = Uxx + Uyy + (k**2 + ell**2) * U[1:-1, 1:-1]

    Xi = X[1:-1, 1:-1]
    Yi = Y[1:-1, 1:-1]

    return dx, Xi, Yi, R

sizes = np.array([41, 81, 161, 321])

dx_values = []
errors = []

for n in sizes:
    dx, Xi, Yi, R = helmholtz_residual(n)
    dx_values.append(dx)
    errors.append(np.max(np.abs(R)))

dx_values = np.array(dx_values)
errors = np.array(errors)

orders = np.log(errors[:-1] / errors[1:]) / np.log(dx_values[:-1] / dx_values[1:])

for n1, n2, p in zip(sizes[:-1], sizes[1:], orders):
    print(f"{n1:4d} -> {n2:4d}: observed order {p:.2f}")

# Residual map for one selected grid
dx, Xi, Yi, R = helmholtz_residual(81)

fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))

# Residual map
limit = np.max(np.abs(R))

pcm = axes[0].pcolormesh(Xi, Yi, R, shading="auto",
                         cmap="RdBu_r", vmin=-limit, vmax=limit)

axes[0].set_xlabel(r"$x$")
axes[0].set_ylabel(r"$y$")
axes[0].set_title(fr"Residual map, $\|R\|_\infty={limit:.2e}$")
axes[0].set_aspect("equal")

fig.colorbar(pcm, ax=axes[0], label=r"$R(x,y)$")

# Convergence plot
axes[1].loglog(dx_values, errors, "o-", label="measured residual")
axes[1].loglog(dx_values,
               errors[-1] * (dx_values / dx_values[-1])**2,
               "--", label=r"$O(\Delta x^2)$")

axes[1].invert_xaxis()
axes[1].set_xlabel(r"grid spacing $\Delta x$")
axes[1].set_ylabel(r"$\|R\|_\infty$")
axes[1].set_title("Convergence of the finite-difference residual")
axes[1].grid(True, which="both", alpha=0.3)
axes[1].legend()

fig.tight_layout()
plt.show()