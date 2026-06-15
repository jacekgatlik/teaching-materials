import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from time import perf_counter

def sum_numpy_chunk(chunk):
    """NumPy-heavy operation on one chunk.

    The mathematical result is almost equal to len(chunk), because
    sin(x)^2 + cos(x)^2 = 1, but the computation still does real work.
    """
    y = np.sin(chunk)**2 + np.cos(chunk)**2
    return float(np.sum(y))

def run_serial(chunks):
    """Serial reference."""
    t0 = perf_counter()
    total = sum(sum_numpy_chunk(chunk) for chunk in chunks)
    return perf_counter() - t0, total

def run_pool(executor_class, chunks, max_workers):
    """Run the same chunked computation using an executor."""
    t0 = perf_counter()

    with executor_class(max_workers=max_workers) as executor:
        partial_results = list(executor.map(sum_numpy_chunk, chunks))

    total = sum(partial_results)
    return perf_counter() - t0, total

if __name__ == "__main__":
    n_workers = 4
    n = 6_000_000

    x = np.linspace(0.0, 100.0, n)
    chunks = np.array_split(x, n_workers)

    print(f"Array size: {x.nbytes / 1e6:.1f} MB")
    print(f"Number of chunks: {len(chunks)}")
    print()

    t_serial, total_serial = run_serial(chunks)

    t_threads, total_threads = run_pool(
        ThreadPoolExecutor,
        chunks,
        max_workers=n_workers,
    )

    t_processes, total_processes = run_pool(
        ProcessPoolExecutor,
        chunks,
        max_workers=n_workers,
    )

    print(f"serial:    {t_serial:.3f} s")
    print(f"threads:   {t_threads:.3f} s")
    print(f"processes: {t_processes:.3f} s")

    print()
    print("Validation:")
    print(abs(total_threads - total_serial))
    print(abs(total_processes - total_serial))

    assert np.allclose(total_threads, total_serial)
    assert np.allclose(total_processes, total_serial)