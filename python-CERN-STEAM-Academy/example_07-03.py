import numpy as np
from time import perf_counter

try:
    from numba import njit, prange
except ImportError as exc:
    raise SystemExit("This example requires Numba. Install it with: pip install numba") from exc

def make_field(n):
    """Construct a slightly perturbed kink-like profile."""
    x = np.linspace(-10.0, 10.0, n)
    dx = x[1] - x[0]
    u = np.tanh(x) + 0.05*np.sin(7.0*x)
    return x, u, dx

def residual_python(u, dx, out):
    """Baseline Python loop."""
    inv_dx2 = 1.0 / dx**2

    out[0] = 0.0
    out[-1] = 0.0

    for i in range(1, u.size - 1):
        u_xx = (u[i-1] - 2.0*u[i] + u[i+1]) * inv_dx2
        out[i] = -u_xx + u[i]*(u[i]**2 - 1.0)

def residual_numpy(u, dx, out):
    """Vectorized NumPy version."""
    inv_dx2 = 1.0 / dx**2

    out[0] = 0.0
    out[-1] = 0.0

    u_mid = u[1:-1]
    u_xx = (u[:-2] - 2.0*u_mid + u[2:]) * inv_dx2

    out[1:-1] = -u_xx + u_mid*(u_mid**2 - 1.0)

@njit
def residual_numba(u, dx, out):
    """Compiled scalar loop."""
    inv_dx2 = 1.0 / dx**2

    out[0] = 0.0
    out[-1] = 0.0

    for i in range(1, u.size - 1):
        u_xx = (u[i-1] - 2.0*u[i] + u[i+1]) * inv_dx2
        out[i] = -u_xx + u[i]*(u[i]**2 - 1.0)

@njit(parallel=True)
def residual_numba_parallel(u, dx, out):
    """Compiled parallel loop.

    This is safe because every iteration writes to a different out[i].
    """
    inv_dx2 = 1.0 / dx**2

    out[0] = 0.0
    out[-1] = 0.0

    for i in prange(1, u.size - 1):
        u_xx = (u[i-1] - 2.0*u[i] + u[i+1]) * inv_dx2
        out[i] = -u_xx + u[i]*(u[i]**2 - 1.0)

def best_time(func, repeat, *args):
    """Measure best runtime over several repetitions."""
    best = np.inf

    for _ in range(repeat):
        t0 = perf_counter()
        func(*args)
        dt = perf_counter() - t0
        best = min(best, dt)

    return best

# Problem setup
n = 300_000
x, u, dx = make_field(n)

out_py = np.empty_like(u)
out_np = np.empty_like(u)
out_jit = np.empty_like(u)
out_par = np.empty_like(u)

# First Numba calls compile the functions 
residual_numba(u, dx, out_jit)
residual_numba_parallel(u, dx, out_par)

# Validate all versions against the baseline
residual_python(u, dx, out_py)
residual_numpy(u, dx, out_np)
residual_numba(u, dx, out_jit)
residual_numba_parallel(u, dx, out_par)

assert np.allclose(out_np, out_py)
assert np.allclose(out_jit, out_py)
assert np.allclose(out_par, out_py)

print("Validation passed.")

# Benchmark steady-state execution 
benchmarks = [
    ("Python loop", residual_python, 1, out_py),
    ("NumPy vectorized", residual_numpy, 20, out_np),
    ("Numba njit", residual_numba, 20, out_jit),
    ("Numba prange", residual_numba_parallel, 20, out_par),
]

for name, func, repeat, out in benchmarks:
    dt = best_time(func, repeat, u, dx, out)
    print(f"{name:18s}: {dt:.6f} s")