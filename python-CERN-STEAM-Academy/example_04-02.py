import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def validate_oscillator_params(params):
    """Check the minimum assumptions needed by the model."""
    required = {"gamma", "omega0", "beta", "drive", "omega_drive"}
    missing = required - set(params)

    if missing:
        raise ValueError(f"Missing parameters: {missing}")

    if params["gamma"] < 0.0:
        raise ValueError("gamma must be non-negative")

    if params["omega0"] <= 0.0:
        raise ValueError("omega0 must be positive")

def oscillator_rhs(t, y, params):
    """
    Right-hand side of a driven damped Duffing oscillator.

        x'' + gamma*x' + omega0^2*x + beta*x^3 = drive*cos(omega_drive*t)
    """
    x, v = y

    gamma = params["gamma"]
    omega0 = params["omega0"]
    beta = params["beta"]
    drive = params["drive"]
    omega_drive = params["omega_drive"]

    force = drive * np.cos(omega_drive*t)

    dxdt = v
    dvdt = -gamma*v - omega0**2*x - beta*x**3 + force

    return np.array([dxdt, dvdt])

def oscillator_energy(y, params):
    """
    Mechanical energy without the external drive contribution.

    For a damped/driven system this is not conserved, but it is still
    a useful diagnostic.
    """
    x = y[0]
    v = y[1]

    omega0 = params["omega0"]
    beta = params["beta"]

    kinetic = 0.5 * v**2
    potential = 0.5 * omega0**2 * x**2 + 0.25 * beta * x**4

    return kinetic + potential

def run_oscillator(params, y0, *, t_span, n_eval=3000):
    """
    Solve the oscillator and return arrays plus diagnostics.
    """
    validate_oscillator_params(params)

    y0 = np.asarray(y0, dtype=float)

    if y0.shape != (2,):
        raise ValueError("y0 must have shape (2,), representing [x0, v0]")

    t_eval = np.linspace(t_span[0], t_span[1], n_eval)

    # The solver expects rhs(t, y).
    # Our model uses rhs(t, y, params), so lambda adapts the interface.
    sol = solve_ivp(
        lambda t, y: oscillator_rhs(t, y, params),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-9,
        atol=1e-11,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    energy = oscillator_energy(sol.y, params)

    # Diagnostics are explicit returned values, not hidden prints.
    last_part = slice(int(0.8*sol.t.size), None)

    diagnostics = {
        "max_abs_x": float(np.max(np.abs(sol.y[0]))),
        "rms_x_last_20_percent": float(np.sqrt(np.mean(sol.y[0, last_part]**2))),
        "mean_energy_last_20_percent": float(np.mean(energy[last_part])),
    }

    return sol.t, sol.y, energy, diagnostics

params = {
    "gamma": 0.08,
    "omega0": 1.0,
    "beta": 0.2,
    "drive": 0.25,
    "omega_drive": 0.9,
}

y0 = [1.0, 0.0]

t, y, energy, diagnostics = run_oscillator(
    params,
    y0,
    t_span=(0.0, 80.0),
)

print("Diagnostics:")
for name, value in diagnostics.items():
    print(f"  {name}: {value:.6g}")

plt.figure(figsize=(8, 4))
plt.plot(t, y[0], label="x(t)")
plt.plot(t, y[1], label="v(t)", alpha=0.75)
plt.xlabel("t")
plt.title("Driven damped nonlinear oscillator")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(t, energy)
plt.xlabel("t")
plt.ylabel("diagnostic energy")
plt.title("Energy-like diagnostic")
plt.tight_layout()
plt.show()