
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration class."""
    
    # Browser Settings
    HEADLESS = os.getenv("HEADLESS", "false").lower() == "true"
    BROWSER_TYPE = os.getenv("BROWSER_TYPE", "chromium")
    VIEWPORT_WIDTH = int(os.getenv("VIEWPORT_WIDTH", "1280"))
    VIEWPORT_HEIGHT = int(os.getenv("VIEWPORT_HEIGHT", "720"))
    TIMEOUT = int(os.getenv("TIMEOUT", "30000"))
    
    # Model Paths
    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = BASE_DIR / "models"
    POLICY_MODEL_PATH = os.getenv("POLICY_MODEL_PATH", str(MODELS_DIR / "policy_network.pth"))
    VALUE_MODEL_PATH = os.getenv("VALUE_MODEL_PATH", str(MODELS_DIR / "value_network.pth"))
    VIT_MODEL_NAME = os.getenv("VIT_MODEL_NAME", "google/vit-base-patch16-224")
    
    # MAYINI Configuration
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "512"))
    HIDDEN_DIM = int(os.getenv("HIDDEN_DIM", "256"))
    NUM_LSTM_LAYERS = int(os.getenv("NUM_LSTM_LAYERS", "2"))
    NUM_ACTIONS = int(os.getenv("NUM_ACTIONS", "50"))
    
    # Training Hyperparameters
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
    GAMMA = float(os.getenv("GAMMA", "0.99"))
    TAU = float(os.getenv("TAU", "0.005"))
    EPSILON_START = float(os.getenv("EPSILON_START", "1.0"))
    EPSILON_END = float(os.getenv("EPSILON_END", "0.01"))
    EPSILON_DECAY = float(os.getenv("EPSILON_DECAY", "0.995"))
    
    # Reinforcement Learning
    REPLAY_BUFFER_SIZE = int(os.getenv("REPLAY_BUFFER_SIZE", "10000"))
    UPDATE_FREQUENCY = int(os.getenv("UPDATE_FREQUENCY", "4"))
    TARGET_UPDATE_FREQUENCY = int(os.getenv("TARGET_UPDATE_FREQUENCY", "10"))
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "7860"))
    DEBUG = os.getenv("DEBUG", "true").lower() == "true"
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "logs/agent.log")
    TENSORBOARD_DIR = os.getenv("TENSORBOARD_DIR", "runs/")
    
    # Hugging Face
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    
    @classmethod
    def validate(cls):
        """Validate critical configuration values."""
        assert cls.EMBEDDING_DIM > 0, "EMBEDDING_DIM must be positive"
        assert cls.HIDDEN_DIM > 0, "HIDDEN_DIM must be positive"
        assert cls.LEARNING_RATE > 0, "LEARNING_RATE must be positive"
        assert 0 <= cls.EPSILON_START <= 1, "EPSILON_START must be in [0, 1]"
        assert 0 <= cls.EPSILON_END <= 1, "EPSILON_END must be in [0, 1]"
        assert 0 < cls.GAMMA <= 1, "GAMMA must be in (0, 1]"
        return True


# Create config instance
config = Config()

# Validate on import
if config.DEBUG:
    config.validate()
