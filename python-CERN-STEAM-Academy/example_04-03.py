from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class TrainingConfig:
    """Configuration of the training loop."""
    learning_rate: float = 0.05
    batch_size: int = 64
    epochs: int = 80

    def __post_init__(self):
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.epochs <= 0:
            raise ValueError("epochs must be positive")

@dataclass(frozen=True)
class Standardizer:
    """Store preprocessing parameters fitted on the training data."""
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, X):
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array")

        mean = X.mean(axis=0)
        scale = X.std(axis=0)

        # Avoid division by zero for constant features.
        scale = np.where(scale == 0.0, 1.0, scale)

        return cls(mean=mean, scale=scale)

    def transform(self, X):
        return (X - self.mean) / self.scale

@dataclass
class FitResult:
    """Structured output of the training procedure."""
    theta: np.ndarray
    bias: float
    train_loss: list
    val_loss: list

def make_synthetic_data(n_samples, rng):
    """
    Create a synthetic regression dataset.

    The true model is linear, so we know what the algorithm should recover.
    """
    X = rng.normal(size=(n_samples, 3))

    theta_true = np.array([1.5, -2.0, 0.7])
    bias_true = -0.4

    noise = 0.3 * rng.normal(size=n_samples)

    y = X @ theta_true + bias_true + noise

    return X, y, theta_true, bias_true

def train_val_split(X, y, *, train_fraction, rng):
    """Split data without using hidden global randomness."""
    if X.ndim != 2:
        raise ValueError("X must have shape (n_samples, n_features)")

    if y.ndim != 1:
        raise ValueError("y must have shape (n_samples,)")

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples")

    n = len(X)
    indices = rng.permutation(n)

    n_train = int(train_fraction*n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

def batches(X, y, batch_size, rng):
    """Yield mini-batches lazily."""
    indices = rng.permutation(len(X))

    for start in range(0, len(X), batch_size):
        batch_idx = indices[start:start + batch_size]

        yield X[batch_idx], y[batch_idx]

def predict(X, theta, bias):
    """Linear model."""
    return X @ theta + bias

def mse_loss(y_pred, y):
    """Mean squared error."""
    residual = y_pred - y
    return float(np.mean(residual**2))

def train_linear_model(X_train, y_train, X_val, y_val, *, config, rng):
    """
    Train a linear model using mini-batch gradient descent.

    All data flow is explicit: data, configuration, and RNG enter as arguments.
    """
    n_features = X_train.shape[1]

    theta = np.zeros(n_features)
    bias = 0.0

    train_loss = []
    val_loss = []

    for epoch in range(config.epochs):
        for X_batch, y_batch in batches(X_train, y_train, config.batch_size, rng):
            y_pred = predict(X_batch, theta, bias)
            residual = y_pred - y_batch

            # Analytical gradients for mean squared error.
            grad_theta = 2.0 * (X_batch.T @ residual) / len(X_batch)
            grad_bias = 2.0 * np.mean(residual)

            theta -= config.learning_rate * grad_theta
            bias -= config.learning_rate * grad_bias

        train_loss.append(mse_loss(predict(X_train, theta, bias), y_train))
        val_loss.append(mse_loss(predict(X_val, theta, bias), y_val))

    return FitResult(
        theta=theta,
        bias=float(bias),
        train_loss=train_loss,
        val_loss=val_loss,
    )


# Explicit source of randomness.
rng = np.random.default_rng(123)

# Data generation.
X, y, theta_true, bias_true = make_synthetic_data(1000, rng)

# Split before preprocessing.
X_train, y_train, X_val, y_val = train_val_split(
    X, y,
    train_fraction=0.8,
    rng=rng,
)

# Fit preprocessing on training data only.
scaler = Standardizer.fit(X_train)

X_train_s = scaler.transform(X_train)
X_val_s = scaler.transform(X_val)

config = TrainingConfig(
    learning_rate=0.05,
    batch_size=64,
    epochs=80,
)

result = train_linear_model(
    X_train_s,
    y_train,
    X_val_s,
    y_val,
    config=config,
    rng=rng,
)

# Convert fitted parameters back to original feature coordinates.
theta_original = result.theta / scaler.scale
bias_original = result.bias - scaler.mean @ theta_original

print("True parameters:")
print("  theta:", theta_true)
print("  bias: ", bias_true)

print("\nRecovered parameters:")
print("  theta:", theta_original)
print("  bias: ", bias_original)

print("\nFinal losses:")
print("  train:", result.train_loss[-1])
print("  val:  ", result.val_loss[-1])

plt.figure(figsize=(8, 4))
plt.semilogy(result.train_loss, label="train loss")
plt.semilogy(result.val_loss, label="validation loss")
plt.xlabel("epoch")
plt.ylabel("MSE loss")
plt.title("Mini-batch training with explicit data flow")
plt.legend()
plt.tight_layout()
plt.show()