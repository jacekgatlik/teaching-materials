import numpy as np

def initial_condition(n):
    """A simple localized initial condition for a diffusion-like example."""
    x = np.linspace(-6.0, 6.0, n)
    dx = x[1] - x[0]
    u = np.exp(-x**2)
    return x, u, dx

def diffusion_step_in_place(u, dx, dt, work):
    """
    Explicit diffusion step.

    The update is performed safely using a work array:
    first compute the new state into 'work',
    then copy it back into 'u'.
    """
    work[0] = u[0]
    work[-1] = u[-1]

    work[1:-1] = (
        u[1:-1]
        + dt*(u[:-2] - 2.0*u[1:-1] + u[2:]) / dx**2
    )

    u[:] = work
    return u

def run_with_bad_snapshots(n=1001, n_steps=200, save_every=40):
    """
    Buggy version: snapshots store references to the same mutable array.
    All saved snapshots will point to the same data buffer.
    """
    x, u, dx = initial_condition(n)
    dt = 0.2*dx**2
    work = np.empty_like(u)

    snapshots = []

    for k in range(n_steps + 1):
        assert np.isfinite(u).all()

        if k % save_every == 0:
            snapshots.append(u)       # BUG: no copy

        if k < n_steps:
            diffusion_step_in_place(u, dx, dt, work)

    return x, snapshots

def run_with_good_snapshots(n=1001, n_steps=200, save_every=40):
    """
    Correct version: each saved snapshot is an independent copy.
    This costs memory, but preserves the history.
    """
    x, u, dx = initial_condition(n)
    dt = 0.2*dx**2
    work = np.empty_like(u)

    snapshots = []

    for k in range(n_steps + 1):
        assert np.isfinite(u).all()

        if k % save_every == 0:
            snapshots.append(u.copy())  # independent snapshot

        if k < n_steps:
            diffusion_step_in_place(u, dx, dt, work)

    return x, snapshots

x, bad = run_with_bad_snapshots()
x, good = run_with_good_snapshots()

print("Bad snapshots:")
print("share memory first-last:", np.shares_memory(bad[0], bad[-1]))
print("max difference first-last:", np.max(np.abs(bad[0] - bad[-1])))

print("\nGood snapshots:")
print("share memory first-last:", np.shares_memory(good[0], good[-1]))
print("max difference first-last:", np.max(np.abs(good[0] - good[-1])))

good_memory = sum(snapshot.nbytes for snapshot in good) / 1024**2
print(f"\nMemory used by good snapshots: {good_memory:.2f} MB")