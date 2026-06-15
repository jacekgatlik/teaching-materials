import numpy as np
from time import perf_counter

def snapshot(x, t):
    """A moving oscillatory wave packet."""
    envelope = np.exp(-0.15 * (x - 0.3*t)**2)
    carrier = np.cos(6.0*x - 1.5*t)
    return envelope * carrier

def energy_of_snapshot(u, dx):
    """Simple quadratic diagnostic for one field snapshot."""
    ux = np.gradient(u, dx)
    density = 0.5*ux**2 + 0.5*u**2
    return np.sum(density) * dx

def store_then_analyze(x, t_values):
    """Store all snapshots first, then compute diagnostics."""
    dx = x[1] - x[0]

    snapshots = np.empty((t_values.size, x.size))

    for k, t in enumerate(t_values):
        snapshots[k] = snapshot(x, t)

    energies = np.empty(t_values.size)

    for k in range(t_values.size):
        energies[k] = energy_of_snapshot(snapshots[k], dx)

    return energies.mean(), snapshots.nbytes

def stream_analyze(x, t_values):
    """Compute diagnostics without storing the full time history."""
    dx = x[1] - x[0]
    total_energy = 0.0

    for t in t_values:
        u = snapshot(x, t)
        total_energy += energy_of_snapshot(u, dx)

    return total_energy / t_values.size


# Problem setup 
nx = 40_000
nt = 150

x = np.linspace(-40.0, 40.0, nx)
t_values = np.linspace(0.0, 30.0, nt)

# Store all snapshots 
t0 = perf_counter()
mean_energy_stored, snapshot_bytes = store_then_analyze(x, t_values)
time_stored = perf_counter() - t0

# Stream snapshots 
t0 = perf_counter()
mean_energy_streamed = stream_analyze(x, t_values)
time_streamed = perf_counter() - t0


# Compare results and memory needs 
print("Mean energy, stored:  ", mean_energy_stored)
print("Mean energy, streamed:", mean_energy_streamed)
print("Difference:           ", abs(mean_energy_stored - mean_energy_streamed))

print(f"Stored snapshot array: {snapshot_bytes / 1e6:.1f} MB")
print(f"Time, stored version:  {time_stored:.3f} s")
print(f"Time, streamed version:{time_streamed:.3f} s")

assert np.allclose(mean_energy_stored, mean_energy_streamed)