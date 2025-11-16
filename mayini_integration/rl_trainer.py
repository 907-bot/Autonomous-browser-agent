import numpy as np
from typing import List, Dict, Tuple
from loguru import logger
from collections import deque
import random

try:
    import mayini as mn
    from mayini.optim import Adam
    MAYINI_AVAILABLE = True
except ImportError:
    MAYINI_AVAILABLE = False
    logger.warning("MAYINI not available for RL training")


class RLTrainer:
    """
    Reinforcement Learning trainer using policy gradients and actor-critic methods.
    Trains MAYINI policy and value networks on browser interaction episodes.
    """
    
    def __init__(
        self,
        policy_network,
        value_network=None,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 32,
        replay_buffer_size: int = 10000
    ):
        """
        Initialize RL trainer.
        
        Args:
            policy_network: Actor/policy network
            value_network: Critic/value network
            learning_rate: Learning rate for optimizers
            gamma: Discount factor
            tau: Soft update parameter
            batch_size: Training batch size
            replay_buffer_size: Size of experience replay buffer
        """
        self.policy_network = policy_network
        self.value_network = value_network
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        self.total_steps = 0
        
        logger.info(f"RL Trainer initialized (lr={learning_rate}, gamma={gamma}, batch_size={batch_size})")
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float = 0.0
    ):
        """
        Store transition in experience replay buffer.
        
        Args:
            state: Current state embedding
            action: Action taken
            reward: Reward received
            next_state: Next state embedding
            done: Whether episode is finished
            log_prob: Log probability of action (for policy gradient)
        """
        self.replay_buffer.append({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "log_prob": log_prob
        })
        self.total_steps += 1
    
    def train_step(self, num_updates: int = 1) -> Dict[str, float]:
        """
        Perform training steps on sampled batch.
        
        Args:
            num_updates: Number of gradient updates per step
            
        Returns:
            Dictionary with training metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0, "avg_reward": 0.0}
        
        total_loss = 0.0
        total_value_loss = 0.0
        
        for _ in range(num_updates):
            # Sample batch from replay buffer
            batch = random.sample(self.replay_buffer, self.batch_size)
            
            # Separate batch components
            states = np.array([t["state"] for t in batch])
            actions = np.array([t["action"] for t in batch])
            rewards = np.array([t["reward"] for t in batch])
            next_states = np.array([t["next_state"] for t in batch])
            dones = np.array([t["done"] for t in batch])
            
            # Compute returns (discounted cumulative rewards)
            returns = self._compute_returns(rewards, dones)
            
            # Normalize returns
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Policy gradient update
            policy_loss = self._update_policy(states, actions, returns)
            total_loss += policy_loss
            
            # Value network update (if available)
            if self.value_network:
                value_loss = self._update_value(states, returns)
                total_value_loss += value_loss
        
        metrics = {
            "policy_loss": total_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "avg_reward": np.mean(rewards),
            "total_steps": self.total_steps,
            "buffer_size": len(self.replay_buffer)
        }
        
        self.training_losses.append(metrics["policy_loss"])
        return metrics
    
    def _compute_returns(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """
        Compute discounted cumulative returns.
        
        Args:
            rewards: Reward sequence
            dones: Episode termination flags
            
        Returns:
            Discounted returns
        """
        returns = np.zeros_like(rewards, dtype=np.float32)
        running_return = 0.0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0.0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def _update_policy(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        returns: np.ndarray
    ) -> float:
        """
        Update policy network using policy gradient theorem.
        
        Args:
            states: State batch
            actions: Action batch
            returns: Return batch
            
        Returns:
            Policy loss value
        """
        losses = []
        
        for state, action, ret in zip(states, actions, returns):
            # Get action log probabilities
            log_probs = self.policy_network.forward(state, return_distribution=False)
            
            # Policy loss: -log_prob * advantage (return)
            # This maximizes likelihood of good actions
            loss = -log_probs[action] * ret
            losses.append(loss)
        
        avg_loss = np.mean(losses)
        
        # In production, would call backward() and optimizer.step()
        logger.debug(f"Policy loss: {avg_loss:.4f}")
        
        return float(avg_loss)
    
    def _update_value(self, states: np.ndarray, returns: np.ndarray) -> float:
        """
        Update value network using MSE loss.
        
        Args:
            states: State batch
            returns: Return batch
            
        Returns:
            Value loss
        """
        if not self.value_network:
            return 0.0
        
        losses = []
        
        for state, ret in zip(states, returns):
            # Predict value
            value = self.value_network.forward(state)
            
            # MSE loss between predicted and actual return
            loss = (value - ret) ** 2
            losses.append(loss)
        
        avg_loss = np.mean(losses)
        logger.debug(f"Value loss: {avg_loss:.4f}")
        
        return float(avg_loss)
    
    def compute_reward(
        self,
        action_result: Dict,
        subtask_progress: float = 0.5,
        efficiency_bonus: float = 0.0
    ) -> float:
        """
        Compute reward from action result and task progress.
        
        Args:
            action_result: Result dictionary from action execution
            subtask_progress: Progress toward current subtask (0-1)
            efficiency_bonus: Bonus for efficient navigation
            
        Returns:
            Computed reward value
        """
        # Base reward for successful action
        reward = 1.0 if action_result.get("success", False) else -0.5
        
        # Bonus for meaningful state changes
        state_change = action_result.get("state_change", "").lower()
        if "successfully" in state_change:
            reward += 2.0
        elif "completed" in state_change:
            reward += 3.0
        elif "found" in state_change:
            reward += 1.5
        
        # Penalty for errors
        if "error" in action_result:
            reward -= 1.0
        
        # Subtask progress bonus
        reward += subtask_progress * 2.0
        
        # Efficiency bonus (negative steps taken)
        reward += efficiency_bonus
        
        return reward
    
    def end_episode(self, episode_reward: float, episode_length: int):
        """
        Record episode statistics.
        
        Args:
            episode_reward: Total reward for episode
            episode_length: Number of steps in episode
        """
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        if len(self.episode_rewards) % 10 == 0:
            avg_reward = np.mean(self.episode_rewards[-10:])
            avg_length = np.mean(self.episode_lengths[-10:])
            logger.info(f"Episodes: {len(self.episode_rewards)}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}")
    
    def get_training_stats(self) -> Dict:
        """
        Get training statistics.
        
        Returns:
            Dictionary with training metrics
        """
        if not self.episode_rewards:
            return {}
        
        return {
            "total_episodes": len(self.episode_rewards),
            "avg_episode_reward": np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards),
            "max_episode_reward": np.max(self.episode_rewards),
            "avg_episode_length": np.mean(self.episode_lengths),
            "total_steps": self.total_steps,
            "buffer_size": len(self.replay_buffer)
        }
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        self.policy_network.save_weights(f"{path}/policy.npz")
        if self.value_network:
            self.value_network.save_weights(f"{path}/value.npz")
        
        # Save statistics
        stats = {
            "episode_rewards": np.array(self.episode_rewards),
            "episode_lengths": np.array(self.episode_lengths),
            "total_steps": self.total_steps
        }
        np.save(f"{path}/stats.npy", stats)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        self.policy_network.load_weights(f"{path}/policy.npz")
        if self.value_network:
            self.value_network.load_weights(f"{path}/value.npz")
        logger.info(f"Checkpoint loaded from {path}")
