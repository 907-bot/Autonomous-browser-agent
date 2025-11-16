import numpy as np
from typing import Tuple, Optional
from loguru import logger
from pathlib import Path

try:
    import mayini as mn
    from mayini.nn import Sequential, Linear, LSTM, ReLU, Dropout, Softmax
    from mayini.optim import Adam
    MAYINI_AVAILABLE = True
except ImportError:
    MAYINI_AVAILABLE = False
    logger.warning("MAYINI framework not available, using NumPy simulation")


class MayiniPolicyNetwork:
    """
    Policy network implemented with MAYINI framework for autonomous decision-making.
    Uses LSTM for sequential reasoning and fully connected layers for action prediction.
    """
    
    def __init__(
        self,
        state_dim: int = 512,
        hidden_dim: int = 256,
        action_dim: int = 50,
        num_lstm_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize MAYINI policy network.
        
        Args:
            state_dim: Input state embedding dimension
            hidden_dim: Hidden layer dimension
            action_dim: Number of possible actions
            num_lstm_layers: Number of stacked LSTM layers
            dropout: Dropout probability for regularization
        """
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_lstm_layers = num_lstm_layers
        self.dropout_rate = dropout
        
        if MAYINI_AVAILABLE:
            self._build_mayini_network()
            logger.info("MAYINI network architecture built successfully")
        else:
            self._build_numpy_network()
            logger.info("NumPy simulation network initialized")
    
    def _build_mayini_network(self):
        """Build network using MAYINI framework."""
        try:
            self.network = Sequential([
                # LSTM layers for sequential processing of action history
                LSTM(input_size=self.state_dim, hidden_size=self.hidden_dim, 
                     num_layers=self.num_lstm_layers, batch_first=True),
                Dropout(p=self.dropout_rate),
                
                # Fully connected layers for policy output
                Linear(self.hidden_dim, self.hidden_dim // 2),
                ReLU(),
                Dropout(p=self.dropout_rate),
                
                Linear(self.hidden_dim // 2, self.action_dim),
                Softmax(dim=-1)
            ])
            
            self.optimizer = Adam(self.network.parameters(), lr=0.001)
            logger.info("MAYINI network and optimizer initialized")
        except Exception as e:
            logger.error(f"Failed to build MAYINI network: {str(e)}")
            MAYINI_AVAILABLE = False
            self._build_numpy_network()
    
    def _build_numpy_network(self):
        """Build simulated network using NumPy (for testing without MAYINI)."""
        # Layer 1: Input to hidden
        self.W1 = np.random.randn(self.state_dim, self.hidden_dim) * 0.01
        self.b1 = np.zeros(self.hidden_dim)
        
        # Layer 2: Hidden to intermediate
        self.W2 = np.random.randn(self.hidden_dim, self.hidden_dim // 2) * 0.01
        self.b2 = np.zeros(self.hidden_dim // 2)
        
        # Layer 3: Intermediate to output (actions)
        self.W3 = np.random.randn(self.hidden_dim // 2, self.action_dim) * 0.01
        self.b3 = np.zeros(self.action_dim)
        
        # LSTM hidden and cell states for sequential processing
        self.lstm_hidden = np.zeros(self.hidden_dim)
        self.lstm_cell = np.zeros(self.hidden_dim)
        
        logger.info("NumPy network weights initialized")
    
    def forward(self, state: np.ndarray, return_distribution: bool = False) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            state: Input state embedding (numpy array)
            return_distribution: If True, return probability distribution, else log-probs
            
        Returns:
            Action logits or probability distribution
        """
        if MAYINI_AVAILABLE:
            return self._forward_mayini(state, return_distribution)
        else:
            return self._forward_numpy(state, return_distribution)
    
    def _forward_mayini(self, state: np.ndarray, return_distribution: bool) -> np.ndarray:
        """Forward pass using MAYINI framework."""
        try:
            # Convert numpy array to MAYINI tensor
            state_tensor = mn.Tensor(state.reshape(1, 1, -1))
            
            # Forward through network
            output = self.network(state_tensor)
            
            # Convert back to numpy
            action_probs = output.data.numpy()
            
            if return_distribution:
                return action_probs.flatten()
            else:
                return np.log(action_probs.flatten() + 1e-8)
        except Exception as e:
            logger.error(f"MAYINI forward pass failed: {str(e)}, falling back to NumPy")
            return self._forward_numpy(state, return_distribution)
    
    def _forward_numpy(self, state: np.ndarray, return_distribution: bool) -> np.ndarray:
        """Forward pass using NumPy simulation."""
        # Ensure correct shape
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        state = state[0]
        
        # LSTM-like processing (simplified)
        # Compute gates
        forget_gate = self._sigmoid(np.dot(state, self.W1[:, :self.hidden_dim // 4]))
        input_gate = self._sigmoid(np.dot(state, self.W1[:, self.hidden_dim // 4:self.hidden_dim // 2]))
        cell_candidate = np.tanh(np.dot(state, self.W1[:, self.hidden_dim // 2:3 * self.hidden_dim // 4]))
        output_gate = self._sigmoid(np.dot(state, self.W1[:, 3 * self.hidden_dim // 4:]))
        
        # Update cell and hidden states
        self.lstm_cell = forget_gate * self.lstm_cell + input_gate * cell_candidate
        self.lstm_hidden = output_gate * np.tanh(self.lstm_cell)
        
        # Fully connected layers
        h1 = np.maximum(0, np.dot(self.lstm_hidden, self.W2[:self.hidden_dim, :]) + self.b2)  # ReLU
        h2 = np.dot(h1, self.W3[:len(h1), :]) + self.b3
        
        if return_distribution:
            # Softmax to get probability distribution
            exp_logits = np.exp(h2 - np.max(h2))
            return exp_logits / np.sum(exp_logits)
        else:
            # Return log-probabilities
            exp_logits = np.exp(h2 - np.max(h2))
            probs = exp_logits / np.sum(exp_logits)
            return np.log(probs + 1e-8)
    
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def reset_hidden_state(self):
        """Reset LSTM hidden and cell states."""
        if not MAYINI_AVAILABLE:
            self.lstm_hidden = np.zeros(self.hidden_dim)
            self.lstm_cell = np.zeros(self.hidden_dim)
            logger.info("LSTM hidden state reset")
    
    def get_action(self, state: np.ndarray, deterministic: bool = False, temperature: float = 1.0) -> int:
        """
        Sample action from policy network.
        
        Args:
            state: Current state embedding
            deterministic: If True, return argmax action (greedy), else sample
            temperature: Temperature for sampling (higher = more random)
            
        Returns:
            Action index
        """
        action_probs = self.forward(state, return_distribution=True)
        
        if deterministic:
            return int(np.argmax(action_probs))
        else:
            # Temperature-scaled sampling
            probs = action_probs ** (1.0 / temperature)
            probs = probs / np.sum(probs)
            return int(np.random.choice(len(action_probs), p=probs))
    
    def load_weights(self, path: str):
        """
        Load pre-trained weights from file.
        
        Args:
            path: Path to weights file (.npz)
        """
        try:
            if Path(path).exists():
                weights = np.load(path, allow_pickle=True)
                if not MAYINI_AVAILABLE:
                    self.W1 = weights['W1']
                    self.W2 = weights['W2']
                    self.W3 = weights['W3']
                    self.b1 = weights['b1']
                    self.b2 = weights['b2']
                    self.b3 = weights['b3']
                logger.info(f"Loaded weights from {path}")
            else:
                logger.warning(f"Weight file not found: {path}")
        except Exception as e:
            logger.error(f"Failed to load weights: {str(e)}")
    
    def save_weights(self, path: str):
        """
        Save network weights to file.
        
        Args:
            path: Path to save weights (.npz)
        """
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            if not MAYINI_AVAILABLE:
                np.savez(path,
                    W1=self.W1, W2=self.W2, W3=self.W3,
                    b1=self.b1, b2=self.b2, b3=self.b3
                )
            logger.info(f"Saved weights to {path}")
        except Exception as e:
            logger.error(f"Failed to save weights: {str(e)}")
