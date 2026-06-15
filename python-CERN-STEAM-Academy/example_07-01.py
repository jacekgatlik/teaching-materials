import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

def make_signal(n, seed=123):
    """Construct a noisy damped oscillatory signal."""
    rng = np.random.default_rng(seed)

    t = np.linspace(0.0, 200.0, n, endpoint=False)
    x = np.exp(-0.01*t) * np.sin(2*np.pi*0.12*t)
    x += 0.15 * rng.normal(size=n)

    # Remove the mean before computing autocorrelation.
    return x - x.mean()

def autocorr_direct(x, max_lag):
    """Direct autocorrelation for selected lags: O(n * max_lag)."""
    c = np.empty(max_lag)

    for lag in range(max_lag):
        c[lag] = np.dot(x[:x.size-lag], x[lag:]) / (x.size - lag)

    return c

def autocorr_fft(x):
    """FFT-based autocorrelation: approximately O(n log n)."""
    n = x.size

    # Zero-padding avoids circular correlation.
    size = 1 << (2*n - 1).bit_length()

    f = np.fft.rfft(x, size)
    c = np.fft.irfft(f * np.conj(f), size)[:n]

    # Normalize by the number of overlapping samples.
    norm = np.arange(n, 0, -1)
    return c / norm

def best_time(func, *args, repeat=3):
    """Return best runtime and last computed output."""
    best = np.inf
    output = None

    for _ in range(repeat):
        t0 = perf_counter()
        output = func(*args)
        dt = perf_counter() - t0
        best = min(best, dt)

    return best, output


# Correctness check on one representative input
n0 = 8_000
x = make_signal(n0)
max_lag = n0 // 4

t_direct, c_direct = best_time(autocorr_direct, x, max_lag, repeat=1)
t_fft, c_fft_full = best_time(autocorr_fft, x, repeat=3)
c_fft = c_fft_full[:max_lag]

print("Validation:")
print("max |direct - FFT| =", np.max(np.abs(c_direct - c_fft)))
print(f"direct time = {t_direct:.4f} s")
print(f"FFT time    = {t_fft:.4f} s")

assert np.allclose(c_direct, c_fft)

# Scaling experiment
sizes = np.array([2_000, 4_000, 8_000, 12_000])
times_direct = []
times_fft = []

for n in sizes:
    x = make_signal(n)
    max_lag = n // 4

    t_direct, _ = best_time(autocorr_direct, x, max_lag, repeat=1)
    t_fft, _ = best_time(autocorr_fft, x, repeat=3)

    times_direct.append(t_direct)
    times_fft.append(t_fft)

    print(f"n = {n:6d} | direct = {t_direct:8.4f} s | FFT = {t_fft:8.4f} s")

# Visual comparison of scaling
plt.figure()
plt.loglog(sizes, times_direct, "o-", label="direct")
plt.loglog(sizes, times_fft, "o-", label="FFT")
plt.xlabel("number of samples")
plt.ylabel("best runtime [s]")
plt.title("Autocorrelation: algorithmic scaling")
plt.legend()
plt.grid(True, which="both")
plt.show()