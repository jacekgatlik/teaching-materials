import os

# Avoid accidental nested parallelism in some numerical libraries.
# These should be set before importing NumPy in a real benchmark script.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from time import perf_counter

try:
    from joblib import Parallel, delayed
except ImportError as exc:
    raise SystemExit("This example requires joblib. Install it with: pip install joblib") from exc

def run_one(alpha):
    """One independent simulation for a given parameter alpha."""
    nx = 12_000
    nsteps = 400

    x = np.linspace(-30.0, 30.0, nx)
    dx = x[1] - x[0]

    # Small stable time step for this explicit demonstration.
    dt = 0.10 * dx**2

    # Perturbed kink-like initial condition.
    u = np.tanh(x) + 0.02*np.sin(3.0*x)

    for _ in range(nsteps):
        lap = (u[:-2] - 2.0*u[1:-1] + u[2:]) / dx**2

        # In-place update of the interior points.
        u[1:-1] += dt * (
            alpha * lap
            - u[1:-1] * (u[1:-1]**2 - 1.0)
        )

    ux = np.gradient(u, dx)
    energy_density = 0.5*ux**2 + 0.25*(u**2 - 1.0)**2
    energy = np.trapezoid(energy_density, x)

    return alpha, energy

def run_sweep(params, n_jobs):
    """Run the full parameter sweep and measure wall-clock time."""
    t0 = perf_counter()

    results = Parallel(n_jobs=n_jobs)(
        delayed(run_one)(float(alpha)) for alpha in params
    )

    return perf_counter() - t0, results

if __name__ == "__main__":
    params = np.linspace(0.2, 1.2, 16)

    timings = {}
    reference = None

    for n_jobs in [1, 2, 4]:
        dt, results = run_sweep(params, n_jobs=n_jobs)
        timings[n_jobs] = dt

        values = np.array([energy for alpha, energy in results])

        if reference is None:
            reference = values
        else:
            assert np.allclose(values, reference)

        speedup = timings[1] / dt
        print(
            f"n_jobs = {n_jobs:2d} | "
            f"time = {dt:7.3f} s | "
            f"speedup = {speedup:5.2f}"
        )

    print()
    print("First few results:")
    for alpha, energy in results[:5]:
        print(f"alpha = {alpha:.3f}, energy = {energy:.6f}")