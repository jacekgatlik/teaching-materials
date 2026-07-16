import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(123)

# 1. Synthetic data: an energy-like quantity
N = 900

q = rng.uniform(-2.0, 2.0, size=N)       # position-like variable
p = rng.uniform(-1.5, 1.5, size=N)       # momentum-like variable
A = rng.uniform(-1.0, 1.0, size=N)       # external-field-like variable

noise = 0.08 * rng.normal(size=N)

# True relation: nonlinear in raw variables,
# but linear in a suitable feature representation.
y = (
    0.5*p**2
    + 0.8*q**2
    + 0.15*q**4
    - 0.4*A*q
    + noise
)

X_raw = np.column_stack([q, p, A])

# 2. Feature map
def feature_map(X_raw):
    q = X_raw[:, 0]
    p = X_raw[:, 1]
    A = X_raw[:, 2]

    return np.column_stack([
        q,
        p,
        A,
        q**2,
        p**2,
        q**4,
        A*q,
    ])

X = feature_map(X_raw)

print("X_raw shape:", X_raw.shape)
print("X shape:", X.shape)
print("y shape:", y.shape)

# 3. Train / validation / test split
idx = rng.permutation(N)

n_train = int(0.70 * N)
n_val = int(0.15 * N)

train_idx = idx[:n_train]
val_idx = idx[n_train:n_train + n_val]
test_idx = idx[n_train + n_val:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

# 4. Scaling fitted only on training data
mu = X_train.mean(axis=0)
scale = X_train.std(axis=0)
scale[scale == 0.0] = 1.0

X_train_s = (X_train - mu) / scale
X_val_s = (X_val - mu) / scale
X_test_s = (X_test - mu) / scale

# 5. Linear model trained by gradient descent
theta = np.zeros(X_train_s.shape[1])
bias = 0.0

eta = 0.05
lambda_reg = 1.0e-3
n_steps = 1500

train_history = []
val_history = []

for step in range(n_steps):
    # Forward pass
    y_pred = X_train_s @ theta + bias
    residual = y_pred - y_train

    # Loss: data term + small L2 regularization
    loss_data = 0.5 * np.mean(residual**2)
    loss = loss_data + 0.5 * lambda_reg * np.sum(theta**2)

    # Analytical gradients
    grad_theta = X_train_s.T @ residual / residual.size
    grad_theta += lambda_reg * theta
    grad_bias = residual.mean()

    # Parameter update
    theta -= eta * grad_theta
    bias -= eta * grad_bias

    # Diagnostics
    train_history.append(loss)

    y_val_pred = X_val_s @ theta + bias
    val_loss = 0.5 * np.mean((y_val_pred - y_val)**2)
    val_history.append(val_loss)

    if not np.isfinite(loss):
        raise RuntimeError("non-finite loss")

# 6. Test diagnostics
y_test_pred = X_test_s @ theta + bias
test_residual = y_test_pred - y_test

rmse_test = np.sqrt(np.mean(test_residual**2))
max_error_test = np.max(np.abs(test_residual))

print("theta shape:", theta.shape)
print("final training loss:", train_history[-1])
print("final validation loss:", val_history[-1])
print("test RMSE:", rmse_test)
print("test max error:", max_error_test)
print("all finite:", np.isfinite(test_residual).all())

# 7. Visual diagnostics
fig, ax = plt.subplots()
ax.plot(train_history, label="training")
ax.plot(val_history, label="validation")
ax.set_xlabel("step")
ax.set_ylabel("loss")
ax.set_yscale("log")
ax.legend()
ax.set_title("Training and validation loss")

fig, ax = plt.subplots()
ax.scatter(y_test, y_test_pred, s=20)

lims = [
    min(y_test.min(), y_test_pred.min()),
    max(y_test.max(), y_test_pred.max()),
]

ax.plot(lims, lims, "r--")
ax.set_xlabel("true test value")
ax.set_ylabel("predicted test value")
ax.set_title("Prediction quality on the test set")

plt.show()