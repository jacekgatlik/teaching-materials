from time import perf_counter
import tracemalloc
import numpy as np

def expression_with_temporaries(u):
    """
    Clear mathematical expression.
    It may create several array-sized temporary objects.
    """
    return np.sin(u)**2 + np.cos(u)**2

def expression_with_buffers(u):
    """
    Less elegant, but more memory-aware.
    We explicitly reuse temporary buffers.
    """
    tmp = np.empty_like(u)
    y = np.empty_like(u)

    np.sin(u, out=tmp)
    np.square(tmp, out=y)

    np.cos(u, out=tmp)
    np.square(tmp, out=tmp)

    y += tmp

    return y

def trace_function(label, function, u):
    """
    Measure elapsed time and traced peak memory.
    tracemalloc tracks Python-level allocations; for NumPy-heavy code,
    combine this with .nbytes and system-level tools in real projects.
    """
    tracemalloc.start()

    start = perf_counter()
    y = function(u)
    elapsed = perf_counter() - start

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Check correctness after stopping tracemalloc, so this check does not
    # influence the measured peak memory of the function itself.
    error = np.max(np.abs(y - 1.0))

    print(f"\n{label}")
    print(f"elapsed time:      {elapsed:.4f} s")
    print(f"traced peak:       {peak / 1024**2:.2f} MB")
    print(f"max error:         {error:.3e}")

n = 2_000_000
u = np.linspace(-10.0, 10.0, n)

print(f"array size: {u.nbytes / 1024**2:.2f} MB")

# Warm-up on a small slice.
expression_with_temporaries(u[:1000])
expression_with_buffers(u[:1000])

trace_function(
    "Mathematical expression with temporaries",
    expression_with_temporaries,
    u,
)

trace_function(
    "Memory-aware version with explicit buffers",
    expression_with_buffers,
    u,
)