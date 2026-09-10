"""
NeuralCore: Zero-dependency, pure-NumPy multi-layer neural network with
exact gradient tracking, gradient clipping, and momentum optimizer.
"""

from typing import Tuple, List, Dict, Optional
import numpy as np


class PureNumPyMLP:
    """
    Multi-layer perceptron neural network designed for instant execution
    within the AcceleratorAI turbine framework without external dependencies.
    """

    def __init__(
        self,
        layer_sizes: Optional[List[int]] = None,
        activation: str = "tanh",
        momentum: float = 0.9,
    ):
        self.layer_sizes = list(layer_sizes) if layer_sizes is not None else [10, 32, 16, 2]
        self.activation = activation
        self.momentum = momentum

        # Initialize weights (He / Xavier initialization)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self.grad_weights: List[np.ndarray] = []
        self.grad_biases: List[np.ndarray] = []
        self.v_weights: List[np.ndarray] = []
        self.v_biases: List[np.ndarray] = []

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            limit = np.sqrt(2.0 / (fan_in + fan_out))
            w = np.random.randn(fan_in, fan_out) * limit
            b = np.zeros((1, fan_out))
            self.weights.append(w)
            self.biases.append(b)
            self.grad_weights.append(np.zeros_like(w))
            self.grad_biases.append(np.zeros_like(b))
            self.v_weights.append(np.zeros_like(w))
            self.v_biases.append(np.zeros_like(b))

        self.last_activations: List[np.ndarray] = []
        self.last_y_true: np.ndarray = np.array([])
        self.last_predictions: np.ndarray = np.array([])

    def _act(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return np.tanh(z)
        elif self.activation == "relu":
            return np.maximum(0.0, z)
        return z

    def _act_grad(self, a: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return 1.0 - a ** 2
        elif self.activation == "relu":
            return (a > 0.0).astype(float)
        return np.ones_like(a)

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / (np.sum(exp_z, axis=-1, keepdims=True) + 1e-8)

    def forward_and_loss(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray = None,
    ) -> Tuple[np.ndarray, float]:
        """Runs forward pass through layers and evaluates cross-entropy loss.

        Args:
            x: Input features.
            y: Target labels (hard int or soft float).
            sample_weights: Optional per-sample curriculum weights from VGT
                           PortManifold. Shape (n,). If provided, loss is a
                           weighted mean instead of uniform mean.
        """
        self.last_activations = [x]
        self.last_sample_weights = sample_weights
        current = x

        for i in range(len(self.weights) - 1):
            z = np.dot(current, self.weights[i]) + self.biases[i]
            current = self._act(z)
            self.last_activations.append(current)

        # Output layer logits
        logits = np.dot(current, self.weights[-1]) + self.biases[-1]
        probs = self._softmax(logits)
        self.last_activations.append(probs)
        self.last_predictions = probs
        self.last_y_true = y

        # Compute Cross-Entropy Loss (optionally curriculum-weighted)
        eps = 1e-9
        y_flat = y.ravel()
        is_soft = np.issubdtype(y_flat.dtype, np.floating) and np.any((y_flat > 0.0) & (y_flat < 1.0))

        if is_soft:
            p1 = probs[:, 1]
            per_sample = -(y_flat * np.log(p1 + eps) + (1.0 - y_flat) * np.log(1.0 - p1 + eps))
        elif y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
            y_indices = np.round(y).astype(int).ravel()
            n = len(y_indices)
            per_sample = -np.log(probs[np.arange(n), y_indices] + eps)
        else:
            per_sample = -np.sum(y * np.log(probs + eps), axis=-1)

        # Weighted or uniform mean
        if sample_weights is not None and len(sample_weights) == len(per_sample):
            loss = float(np.sum(per_sample * sample_weights) / (np.sum(sample_weights) + eps))
        else:
            loss = float(np.mean(per_sample))

        return probs, float(loss)

    def backward(self) -> float:
        """Runs backward pass, storing gradients and returning total gradient norm.

        If sample_weights were provided to forward_and_loss, the output gradient
        delta is scaled per-sample by curriculum weight, so hyper-flow samples
        contribute proportionally more torque to the gradient turbine.
        """
        probs = self.last_activations[-1]
        y = self.last_y_true
        n = probs.shape[0]
        sw = getattr(self, 'last_sample_weights', None)

        # Output gradient
        y_flat = y.ravel()
        is_soft = np.issubdtype(y_flat.dtype, np.floating) and np.any((y_flat > 0.0) & (y_flat < 1.0))

        if is_soft:
            delta = probs.copy()
            delta[:, 1] -= y_flat
            delta[:, 0] -= (1.0 - y_flat)
            delta /= n
        elif y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
            y_indices = np.round(y).astype(int).ravel()
            delta = probs.copy()
            delta[np.arange(n), y_indices] -= 1.0
            delta /= n
        else:
            delta = (probs - y) / n

        # Apply per-sample curriculum weighting to the gradient
        if sw is not None and len(sw) == n:
            delta = delta * sw[:, np.newaxis]

        # Backprop through layers
        total_sq_norm = 0.0
        num_layers = len(self.weights)

        for i in reversed(range(num_layers)):
            a_prev = self.last_activations[i]
            grad_w = np.dot(a_prev.T, delta)
            grad_b = np.sum(delta, axis=0, keepdims=True)

            self.grad_weights[i] = grad_w
            self.grad_biases[i] = grad_b

            total_sq_norm += np.sum(grad_w ** 2) + np.sum(grad_b ** 2)

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self._act_grad(a_prev)

        return float(np.sqrt(total_sq_norm))

    def clip_gradients(self, max_norm: float) -> None:
        """Wastegate clipping: rescales gradients if total norm exceeds threshold."""
        total_sq = sum(
            np.sum(gw ** 2) + np.sum(gb ** 2)
            for gw, gb in zip(self.grad_weights, self.grad_biases)
        )
        total_norm = np.sqrt(total_sq)
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-8)
            for i in range(len(self.grad_weights)):
                self.grad_weights[i] *= scale
                self.grad_biases[i] *= scale

    def apply_updates(self, learning_rate: float) -> None:
        """Drive Shaft parameter update with momentum."""
        for i in range(len(self.weights)):
            self.v_weights[i] = (self.momentum * self.v_weights[i]) - (learning_rate * self.grad_weights[i])
            self.v_biases[i] = (self.momentum * self.v_biases[i]) - (learning_rate * self.grad_biases[i])
            self.weights[i] += self.v_weights[i]
            self.biases[i] += self.v_biases[i]
