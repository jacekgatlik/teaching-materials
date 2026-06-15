import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(123)

# 1. Synthetic two-class dataset
N0 = 400
N1 = 400

mean0 = np.array([-1.0, -0.8])
mean1 = np.array([1.0, 0.9])

cov = np.array([
    [0.8, 0.35],
    [0.35, 0.6],
])

X0 = rng.multivariate_normal(mean0, cov, size=N0)
X1 = rng.multivariate_normal(mean1, cov, size=N1)

X = np.vstack([X0, X1])
y = np.concatenate([np.zeros(N0), np.ones(N1)])

# Shuffle the full dataset
idx = rng.permutation(X.shape[0])
X = X[idx]
y = y[idx]

# 2. Train / validation / test split
N = X.shape[0]

n_train = int(0.70 * N)
n_val = int(0.15 * N)

X_train, y_train = X[:n_train], y[:n_train]
X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

# 3. Scaling fitted only on training data
mu = X_train.mean(axis=0)
scale = X_train.std(axis=0)
scale[scale == 0.0] = 1.0

X_train_s = (X_train - mu) / scale
X_val_s = (X_val - mu) / scale
X_test_s = (X_test - mu) / scale

# 4. Numerically stable functions
def sigmoid_stable(z):
    """Stable sigmoid for positive and negative logits."""
    out = np.empty_like(z, dtype=float)

    pos = z >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))

    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)

    return out


def bce_with_logits(logits, y):
    """Stable binary cross-entropy evaluated directly from logits."""
    terms = (
        np.maximum(logits, 0.0)
        - logits*y
        + np.log1p(np.exp(-np.abs(logits)))
    )
    return np.mean(terms)


def accuracy(X, y, theta, bias):
    logits = X @ theta + bias
    prob = sigmoid_stable(logits)
    pred = (prob >= 0.5).astype(float)
    return np.mean(pred == y)

# 5. Mini-batch logistic regression
theta = np.zeros(X_train_s.shape[1])
bias = 0.0

eta = 0.15
lambda_reg = 1.0e-3
batch_size = 64
n_epochs = 120

train_loss_history = []
val_loss_history = []
val_acc_history = []

N_train = X_train_s.shape[0]

for epoch in range(n_epochs):
    order = rng.permutation(N_train)

    for start in range(0, N_train, batch_size):
        batch_idx = order[start:start + batch_size]

        Xb = X_train_s[batch_idx]
        yb = y_train[batch_idx]

        # Forward pass
        logits = Xb @ theta + bias
        prob = sigmoid_stable(logits)

        # Gradient of BCE with logits for logistic regression
        error = prob - yb

        grad_theta = Xb.T @ error / error.size
        grad_theta += lambda_reg * theta
        grad_bias = error.mean()

        # Parameter update
        theta -= eta * grad_theta
        bias -= eta * grad_bias

    # Diagnostics after each epoch
    train_logits = X_train_s @ theta + bias
    val_logits = X_val_s @ theta + bias

    train_loss = bce_with_logits(train_logits, y_train)
    train_loss += 0.5 * lambda_reg*np.sum(theta**2)

    val_loss = bce_with_logits(val_logits, y_val)
    val_acc = accuracy(X_val_s, y_val, theta, bias)

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)

    if not np.isfinite(train_loss):
        raise RuntimeError("non-finite training loss")

# 6. Final test diagnostics
test_logits = X_test_s @ theta + bias
test_loss = bce_with_logits(test_logits, y_test)
test_acc = accuracy(X_test_s, y_test, theta, bias)

print("theta:", theta)
print("bias:", bias)
print("final validation accuracy:", val_acc_history[-1])
print("test loss:", test_loss)
print("test accuracy:", test_acc)
print("finite parameters:", np.isfinite(theta).all() and np.isfinite(bias))

# 7. Loss and accuracy curves
fig, ax = plt.subplots()
ax.plot(train_loss_history, label="training loss")
ax.plot(val_loss_history, label="validation loss")
ax.set_xlabel("epoch")
ax.set_ylabel("binary cross-entropy")
ax.legend()
ax.set_title("Training and validation loss")

fig, ax = plt.subplots()
ax.plot(val_acc_history)
ax.set_xlabel("epoch")
ax.set_ylabel("validation accuracy")
ax.set_title("Validation accuracy")

# 8. Decision map in the original coordinates
x1 = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200)
x2 = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200)

G1, G2 = np.meshgrid(x1, x2)
grid_raw = np.column_stack([G1.ravel(), G2.ravel()])
grid_s = (grid_raw - mu) / scale

prob_grid = sigmoid_stable(grid_s @ theta + bias)
prob_grid = prob_grid.reshape(G1.shape)

fig, ax = plt.subplots()
cont = ax.contourf(G1, G2, prob_grid, levels=20, alpha=0.8)
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=25, edgecolor="k")
ax.set_xlabel("feature 1")
ax.set_ylabel("feature 2")
ax.set_title("Predicted probability and test samples")
fig.colorbar(cont, ax=ax, label="P(class = 1)")

plt.show()