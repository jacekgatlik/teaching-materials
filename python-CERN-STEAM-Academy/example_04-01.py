import numpy as np
import matplotlib.pyplot as plt

def initial_condition(x):
    """Initial localized field."""
    return np.exp(-(x + 1.5)**2) + 0.6*np.exp(-4.0*(x - 1.0)**2)

def heat_step_in_place(u, *, D, dt, dx):
    """
    One explicit heat-equation step.

    The function intentionally modifies u in place.
    """
    r = D * dt / dx**2

    if r > 0.5:
        raise ValueError("Explicit heat step is unstable: D*dt/dx^2 must be <= 0.5")

    # Copy the old state before overwriting u.
    old = u.copy()

    # Interior update.
    u[1:-1] = old[1:-1] + r*(old[:-2] - 2.0*old[1:-1] + old[2:])

    # Simple zero-gradient boundary condition.
    u[0] = u[1]
    u[-1] = u[-2]

def solve_bad(u0, *, D, dt, dx, nsteps, save_every):
    """
    Bad version: all saved snapshots refer to the same mutable array.
    """
    u = u0.copy()
    snapshots = []

    for step in range(nsteps + 1):
        if step % save_every == 0:
            snapshots.append(u)        # BUG: no copy here

        if step < nsteps:
            heat_step_in_place(u, D=D, dt=dt, dx=dx)

    return snapshots

def solve_good(u0, *, D, dt, dx, nsteps, save_every):
    """
    Good version: each saved snapshot is an independent array.
    """
    u = u0.copy()
    snapshots = []

    for step in range(nsteps + 1):
        if step % save_every == 0:
            snapshots.append(u.copy()) # independent snapshot

        if step < nsteps:
            heat_step_in_place(u, D=D, dt=dt, dx=dx)

    return snapshots

# Grid and numerical parameters.
x = np.linspace(-5.0, 5.0, 401)
dx = x[1] - x[0]

D = 0.1
dt = 0.4 * dx**2 / D
nsteps = 300
save_every = 50

u0 = initial_condition(x)

bad = solve_bad(u0, D=D, dt=dt, dx=dx, nsteps=nsteps, save_every=save_every)
good = solve_good(u0, D=D, dt=dt, dx=dx, nsteps=nsteps, save_every=save_every)

print("bad[0] is bad[-1]:", bad[0] is bad[-1])
print("good[0] is good[-1]:", good[0] is good[-1])

print("Difference between first and last bad snapshot:",
      np.max(np.abs(bad[0] - bad[-1])))

print("Difference between first and last good snapshot:",
      np.max(np.abs(good[0] - good[-1])))

# Visual check.
plt.figure(figsize=(8, 4))

plt.plot(x, bad[0],  label="bad: first snapshot")
plt.plot(x, bad[-1], "-.", label="bad: last snapshot")

plt.plot(x, good[0],  label="good: first snapshot")
plt.plot(x, good[-1], "--", label="good: last snapshot")

plt.xlabel("x")
plt.ylabel("u")
plt.title("Hidden mutation: snapshots without copy collapse to the final state")
plt.legend()
plt.tight_layout()
plt.show()