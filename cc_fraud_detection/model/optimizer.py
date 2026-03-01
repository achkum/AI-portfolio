import torch

class ManualOptimizer:
    """Hand-coded Gradient Descent Optimizer."""
    def __init__(self, params, lr=0.001):
        self.params = list(params)
        self.lr = lr

    def step(self):
        """Update weights based on gradients."""
        with torch.no_grad():
            for param in self.params:
                if param.grad is not None:
                    param -= self.lr * param.grad

    def zero_grad(self):
        """Zero out gradients before the next pass."""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
