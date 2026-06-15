import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(123)

# 1. True equation and noisy data
# ------------------------------------------------------------
# We assume that the unknown function should satisfy
#
#     u_t + gamma*u = 0,
#     u(0) = 1.
#
# The exact solution is u(t) = exp(-gamma*t).

gamma = 0.7
t_min, t_max = 0.0, 5.0

def u_exact(t):
    return np.exp(-gamma*t)

n_data = 12

t_data = np.linspace(t_min, t_max, n_data)
y_data = u_exact(t_data) + 0.03*rng.normal(size=n_data)

# Points where we enforce the differential equation residual
t_col = np.linspace(t_min, t_max, 120)

# 2. Model: polynomial basis in scaled time
degree = 8
n_params = degree + 1

def scaled_time(t):
    """Map t from [t_min, t_max] to [-1, 1]."""
    return 2.0*(t - t_min)/(t_max - t_min) - 1.0


def basis(t):
    """Polynomial basis: 1, s, s^2, ..., s^degree."""
    t = np.asarray(t)
    s = scaled_time(t)

    powers = np.arange(n_params)
    return s[:, None]**powers[None, :]


def basis_dt(t):
    """Time derivative of the polynomial basis."""
    t = np.asarray(t)
    s = scaled_time(t)
    ds_dt = 2.0/(t_max - t_min)

    Bdt = np.zeros((t.size, n_params))

    for j in range(1, n_params):
        Bdt[:, j] = j * s**(j - 1) * ds_dt

    return Bdt

# Precompute basis matrices
B_data = basis(t_data)
B_col = basis(t_col)
Bt_col = basis_dt(t_col)
B0 = basis(np.array([t_min]))

# Residual matrix for u_t + gamma*u
R_col = Bt_col + gamma*B_col

# 3. Training function
def train(alpha_ode, n_steps=3000, eta=0.03, lambda_reg=1.0e-5):
    """
    alpha_ode = 0.0 gives data-only fitting.
    alpha_ode > 0 adds the differential-equation residual.
    """
    theta = np.zeros(n_params)
    history = []

    for step in range(n_steps):
        # Data residual
        pred_data = B_data @ theta
        res_data = pred_data - y_data

        # ODE residual: u_t + gamma*u
        ode_res = R_col @ theta

        # Initial condition residual: u(0) - 1
        ic_res = (B0 @ theta - 1.0).ravel()

        # Loss components
        loss_data = 0.5*np.mean(res_data**2)
        loss_ode = 0.5*np.mean(ode_res**2)
        loss_ic = 0.5*ic_res[0]**2
        loss_reg = 0.5*lambda_reg*np.sum(theta**2)

        loss = loss_data + alpha_ode*loss_ode + loss_ic + loss_reg

        # Analytical gradient of the full loss
        grad = B_data.T @ res_data / res_data.size
        grad += alpha_ode * (R_col.T @ ode_res) / ode_res.size
        grad += B0.ravel() * ic_res[0]
        grad += lambda_reg * theta

        # Gradient descent update
        theta -= eta * grad

        history.append(loss)

        if not np.isfinite(loss):
            raise RuntimeError("non-finite loss")

    return theta, np.array(history)

# 4. Compare data-only and physics-informed fitting
theta_data, hist_data = train(alpha_ode=0.0)
theta_phys, hist_phys = train(alpha_ode=1.0)

t_plot = np.linspace(t_min, t_max, 400)

B_plot = basis(t_plot)
R_plot = basis_dt(t_plot) + gamma*basis(t_plot)

u_data_only = B_plot @ theta_data
u_phys = B_plot @ theta_phys
u_true = u_exact(t_plot)

res_data_only = R_plot @ theta_data
res_phys = R_plot @ theta_phys

rmse_data_only = np.sqrt(np.mean((u_data_only - u_true)**2))
rmse_phys = np.sqrt(np.mean((u_phys - u_true)**2))

rms_ode_data_only = np.sqrt(np.mean(res_data_only**2))
rms_ode_phys = np.sqrt(np.mean(res_phys**2))

print("RMSE data-only:", rmse_data_only)
print("RMSE physics-informed:", rmse_phys)
print("ODE residual RMS, data-only:", rms_ode_data_only)
print("ODE residual RMS, physics-informed:", rms_ode_phys)

# 5. Visual diagnostics
fig, ax = plt.subplots()
ax.scatter(t_data, y_data, label="noisy data")
ax.plot(t_plot, u_true, label="exact solution")
ax.plot(t_plot, u_data_only, "--", label="data-only fit")
ax.plot(t_plot, u_phys, label="physics-informed fit")
ax.set_xlabel("t")
ax.set_ylabel("u(t)")
ax.legend()
ax.set_title("Data fitting versus physics-informed fitting")

fig, ax = plt.subplots()
ax.plot(t_plot, res_data_only, "--", label="data-only residual")
ax.plot(t_plot, res_phys, label="physics-informed residual")
ax.axhline(0.0, linestyle=":")
ax.set_xlabel("t")
ax.set_ylabel(r"$u_t + \gamma u$")
ax.legend()
ax.set_title("Differential-equation residual")

fig, ax = plt.subplots()
ax.plot(hist_data, label="data-only loss")
ax.plot(hist_phys, label="physics-informed loss")
ax.set_xlabel("step")
ax.set_ylabel("loss")
ax.set_yscale("log")
ax.legend()
ax.set_title("Loss histories")

plt.show()