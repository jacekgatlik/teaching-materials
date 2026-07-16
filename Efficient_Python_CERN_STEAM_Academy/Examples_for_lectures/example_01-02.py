import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)

# Number of independent vectors / rotations
n = 200

# Angles of rotation
theta = np.linspace(0.0, 2*np.pi, n, endpoint=False)

# Random vectors: vectors[i] is a 2D vector
vectors = rng.normal(size=(n, 2))

# Rotation matrices: R[i] is a 2 x 2 matrix
R = np.zeros((n, 2, 2))

c = np.cos(theta)
s = np.sin(theta)

R[:, 0, 0] = c
R[:, 0, 1] = -s
R[:, 1, 0] = s
R[:, 1, 1] = c

# Batched matrix-vector product:
# rotated[i, :] = R[i, :, :] @ vectors[i, :]
rotated = np.einsum("nij,nj->ni", R, vectors)

# Rotation should preserve vector lengths
length_before = np.linalg.norm(vectors, axis=1)
length_after = np.linalg.norm(rotated, axis=1)

print("vectors shape:", vectors.shape)
print("R shape:      ", R.shape)
print("rotated shape:", rotated.shape)
print("maximum length error:", np.max(np.abs(length_before - length_after)))

# Plot only a small subset to keep the figure readable
m = 30

plt.figure(figsize=(5, 5))
plt.plot(vectors[:m, 0], vectors[:m, 1], "o", label="original")
plt.plot(rotated[:m, 0], rotated[:m, 1], "x", label="rotated")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.legend()
plt.tight_layout()
plt.show()