import cProfile
import io
import pstats
from pstats import SortKey
import numpy as np

def make_problem(n):
    x = np.linspace(-10.0, 10.0, n)
    dx = x[1] - x[0]
    u = np.tanh(x)
    return u, dx

def local_force_scalar(value):
    return value*(value**2 - 1.0)

def residual_many_calls(u, dx):
    r = np.zeros_like(u)

    for i in range(1, u.size - 1):
        u_xx = (u[i - 1] - 2.0*u[i] + u[i + 1]) / dx**2
        r[i] = -u_xx + local_force_scalar(u[i])

    return r

def residual_vectorized(u, dx):
    r = np.zeros_like(u)

    r[1:-1] = (
        -(u[:-2] - 2.0*u[1:-1] + u[2:]) / dx**2
        + u[1:-1]*(u[1:-1]**2 - 1.0)
    )

    return r

def step(u, dx, dt, residual_function):
    return u - dt*residual_function(u, dx)

def run_simulation(n, n_steps, residual_function):
    u, dx = make_problem(n)
    dt = 1e-10

    for _ in range(n_steps):
        u = step(u, dx, dt, residual_function)

    return u

def profile_function(label, function, n_lines=12):
    profiler = cProfile.Profile()

    profiler.enable()
    result = function()
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)

    stats.strip_dirs()
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(n_lines)

    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)
    print(stream.getvalue())

    return result

n = 20_000
n_steps = 50

u_loop = profile_function(
    "Profile: residual with many Python-level calls",
    lambda: run_simulation(n, n_steps, residual_many_calls),
)

u_vec = profile_function(
    "Profile: vectorized residual",
    lambda: run_simulation(n, n_steps, residual_vectorized),
)

# The two implementations should describe the same numerical update.
difference = np.max(np.abs(u_loop - u_vec))
print(f"Maximum difference between final states: {difference:.3e}")