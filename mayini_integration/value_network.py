
"""
Value Network - Critic for actor-critic RL
"""

import numpy as np
from loguru import logger

try:
    import mayini as mn
    MAYINI_AVAILABLE = True
except ImportError:
    MAYINI_AVAILABLE = False


class MayiniValueNetwork:
    """Value network for estimating state values."""
    
    def __init__(self, state_dim: int = 512, hidden_dim: int = 256):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        if not MAYINI_AVAILABLE:
            self._build_numpy_network()
        
        logger.info("Value Network initialized")
    
    def _build_numpy_network(self):
        """Build simulated network."""
        self.W1 = np.random.randn(self.state_dim, self.hidden_dim) * 0.01
        self.W2 = np.random.randn(self.hidden_dim, 1) * 0.01
        self.b1 = np.zeros(self.hidden_dim)
        self.b2 = np.zeros(1)
    
    def forward(self, state: np.ndarray) -> float:
        """Estimate state value."""
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        h1 = np.maximum(0, np.dot(state, self.W1) + self.b1)
        value = np.dot(h1, self.W2) + self.b2
        
        return float(value)
