"""
State Manager - State representation and encoding
"""

import numpy as np
from typing import Dict, List, Any
from loguru import logger
import hashlib


class StateManager:
    """Manages state representation and encoding."""
    
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.state_cache = {}
        logger.info(f"State Manager initialized (dim={embedding_dim})")
    
    def encode_state(
        self,
        visual_features: Dict,
        url: str,
        task: str,
        history: List[Dict]
    ) -> np.ndarray:
        """Encode current state into embedding."""
        
        # Get visual embedding
        vit_embedding = visual_features.get("embedding", np.zeros(self.embedding_dim // 2))
        
        # Encode URL
        url_features = self._encode_url(url)
        
        # Encode task
        task_features = self._encode_text(task)
        
        # Encode history
        history_features = self._encode_history(history)
        
        # Combine features
        state_embedding = np.concatenate([
            vit_embedding[:self.embedding_dim // 4],
            url_features[:self.embedding_dim // 4],
            task_features[:self.embedding_dim // 4],
            history_features[:self.embedding_dim // 4]
        ])
        
        # Normalize
        state_embedding = state_embedding / (np.linalg.norm(state_embedding) + 1e-8)
        
        return state_embedding.astype(np.float32)
    
    def _encode_url(self, url: str) -> np.ndarray:
        """Encode URL into feature vector."""
        url_hash = hashlib.md5(url.encode()).digest()
        features = np.frombuffer(url_hash, dtype=np.uint8).astype(np.float32)
        features = features / 255.0
        
        repeat_factor = (self.embedding_dim // 4) // len(features) + 1
        features = np.tile(features, repeat_factor)[:self.embedding_dim // 4]
        
        return features
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text into feature vector."""
        text_bytes = text.lower().encode('utf-8', errors='ignore')[:self.embedding_dim // 4]
        features = np.zeros(self.embedding_dim // 4, dtype=np.float32)
        
        for i, byte in enumerate(text_bytes):
            if i < len(features):
                features[i] = byte / 255.0
        
        return features
    
    def _encode_history(self, history: List[Dict]) -> np.ndarray:
        """Encode action history."""
        features = np.zeros(self.embedding_dim // 4, dtype=np.float32)
        
        if not history:
            return features
        
        recent_history = history[-10:]
        action_types = {"click": 0.2, "type": 0.4, "scroll": 0.6, "navigate": 0.8}
        
        for i, hist_item in enumerate(recent_history):
            action_type = hist_item.get("action_type", "unknown")
            value = action_types.get(action_type, 0.1)
            
            idx = i * 2
            if idx < len(features):
                features[idx] = value
                features[idx + 1] = 1.0
        
        return features

