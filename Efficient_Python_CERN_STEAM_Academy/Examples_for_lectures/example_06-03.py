from time import perf_counter
import gc
import tracemalloc

import numpy as np


def expression_with_temporaries(u):
    sin_u = np.sin(u)
    cos_u = np.cos(u)

    sin_squared = np.square(sin_u)
    cos_squared = np.square(cos_u)

    return np.add(sin_squared, cos_squared)


def expression_with_buffers(u):
    tmp = np.empty_like(u)
    y = np.empty_like(u)

    # y = sin(u)^2
    np.sin(u, out=tmp)
    np.square(tmp, out=y)

    # tmp = cos(u)^2
    np.cos(u, out=tmp)
    np.square(tmp, out=tmp)

    # y = y + tmp
    np.add(y, tmp, out=y)

    return y


def trace_function(label, function, u):
    gc.collect()

    tracemalloc.start()
    tracemalloc.reset_peak()

    start = perf_counter()
    y = function(u)
    elapsed = perf_counter() - start

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    error = max(
        abs(y_min - 1.0),
        abs(y_max - 1.0),
    )

    mib = 1024**2

    print(f"\n{label}")
    print(f"elapsed time:          {elapsed:.4f} s")
    print(f"traced current:        {current / mib:.2f} MiB")
    print(f"traced peak:           {peak / mib:.2f} MiB")
    print(f"peak / one array:      {peak / u.nbytes:.2f} x")
    print(f"u + traced peak:       {(u.nbytes + peak) / mib:.2f} MiB")
    print(f"max error from 1.0:    {error:.3e}")

    del y
    gc.collect()


n = 50_000_000

u = np.linspace(
    -10.0,
    10.0,
    n,
    dtype=np.float64,
)

print(f"input array size: {u.nbytes / 1024**2:.2f} MiB")


u_small = u[:10_000]

expected = expression_with_temporaries(u_small)
actual = expression_with_buffers(u_small)

np.testing.assert_allclose(
    actual,
    expected,
    rtol=0.0,
    atol=0.0,
)

del expected, actual
gc.collect()


trace_function(
    "Expression keeping all temporary arrays alive",
    expression_with_temporaries,
    u,
)

trace_function(
    "Memory-aware expression using explicit buffers",
    expression_with_buffers,
    u,
)