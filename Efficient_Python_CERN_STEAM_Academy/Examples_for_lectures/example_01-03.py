import numpy as np
import matplotlib.pyplot as plt

def residual_norm(nx, L=10.0):
    # Spatial grid
    x = np.linspace(-L, L, nx)
    dx = x[1] - x[0]

    # Exact solution of u_xx + u - u^3 = 0
    u = np.tanh(x / np.sqrt(2.0))

    # Second derivative on interior points
    u_xx = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2

    # Residual on interior points
    res = u_xx + u[1:-1] - u[1:-1]**3

    # RMS norm of the residual
    norm = np.sqrt(np.mean(res**2))

    return dx, norm

# Different grid resolutions
n_values = [51, 101, 201, 401, 801, 1601]

data = np.array([residual_norm(n) for n in n_values])
dx_values = data[:, 0]
res_values = data[:, 1]

# Estimate convergence order: residual ~ dx^p
p = np.polyfit(np.log(dx_values), np.log(res_values), 1)[0]

for dx, res in zip(dx_values, res_values):
    print(f"dx = {dx:.5f}, residual norm = {res:.6e}")

print(f"Estimated convergence order: {p:.2f}")

# Reference second-order line
reference = res_values[0] * (dx_values / dx_values[0])**2

plt.figure(figsize=(6, 4))
plt.loglog(dx_values, res_values, "o-", label="residual norm")
plt.loglog(dx_values, reference, "--", label="O(dx²) reference")
plt.xlabel("dx")
plt.ylabel("RMS residual norm")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.show()