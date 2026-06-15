import timeit
import numpy as np

def make_problem(n):
    """Create a simple 1D field on a uniform grid."""
    x = np.linspace(-10.0, 10.0, n)
    dx = x[1] - x[0]
    u = np.tanh(x)
    return u, dx

def residual_loop(u, dx):
    """Clear baseline: finite-difference residual using a Python loop."""
    r = np.zeros_like(u)

    for i in range(1, u.size - 1):
        u_xx = (u[i - 1] - 2.0*u[i] + u[i + 1]) / dx**2
        r[i] = -u_xx + u[i]*(u[i]**2 - 1.0)

    return r

def residual_vectorized(u, dx):
    """Candidate implementation: the same residual using NumPy slicing."""
    r = np.zeros_like(u)

    r[1:-1] = (
        -(u[:-2] - 2.0*u[1:-1] + u[2:]) / dx**2
        + u[1:-1]*(u[1:-1]**2 - 1.0)
    )

    return r

def best_and_median_time(function, u, dx, number=20, repeat=5):
    """Return best and median time per function call."""
    samples = timeit.repeat(
        lambda: function(u, dx),
        number=number,
        repeat=repeat,
    )

    samples = np.array(samples) / number
    return samples.min(), np.median(samples)

sizes = [1_000, 3_000, 10_000, 30_000, 100_000]

print(f"{'n':>10} {'loop [s]':>12} {'vector [s]':>12} {'speedup':>10}")
print("-" * 50)

for n in sizes:
    u, dx = make_problem(n)

    # Correctness before performance comparison.
    r_loop = residual_loop(u, dx)
    r_vec = residual_vectorized(u, dx)
    assert np.allclose(r_loop, r_vec, rtol=1e-12, atol=1e-12)

    # Use fewer repetitions for larger arrays.
    number = 50 if n <= 10_000 else 10

    t_loop, _ = best_and_median_time(residual_loop, u, dx, number=number)
    t_vec, _ = best_and_median_time(residual_vectorized, u, dx, number=number)

    print(f"{n:10d} {t_loop:12.4e} {t_vec:12.4e} {t_loop/t_vec:10.1f}")