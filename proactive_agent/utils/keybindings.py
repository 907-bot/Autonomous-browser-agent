#!/usr/bin/env python3
"""
⌨️ Keybindings Module
====================
Global hotkey management.
"""

import sys
import logging
from typing import Dict, Callable, Optional, List
from functools import wraps

logger = logging.getLogger("Keybindings")

# Try keyboard library
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    keyboard = None


class Keybinder:
    """
    Global hotkey manager.
    """
    
    def __init__(self):
        self.available = KEYBOARD_AVAILABLE
        self.registered_keys: Dict[str, str] = {}  # key -> callback name
        self.callbacks: Dict[str, Callable] = {}
        
        logger.info(f"⌨️ Keybinder initialized (available={KEYBOARD_AVAILABLE})")
    
    def register(
        self,
        key: str,
        callback: Callable,
        name: str = None
    ) -> bool:
        """
        Register a global hotkey.
        
        Args:
            key: Key combination (e.g., "ctrl+shift+a")
            callback: Function to call
            name: Optional name for the binding
            
        Returns:
            True if successful
        """
        if not self.available:
            logger.warning(f"Keyboard not available, cannot register {key}")
            return False
        
        key_normalized = self._normalize_key(key)
        
        # Generate name if not provided
        if not name:
            name = callback.__name__
        
        self.callbacks[name] = callback
        self.registered_keys[key_normalized] = name
        
        try:
            keyboard.add_hotkey(key_normalized, callback)
            logger.info(f"⌨️ Registered: {key_normalized} -> {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register {key}: {e}")
            return False
    
    def unregister(self, key: str) -> bool:
        """Unregister a hotkey."""
        key_normalized = self._normalize_key(key)
        
        if key_normalized in self.registered_keys:
            del self.registered_keys[key_normalized]
            logger.info(f"⌨️ Unregistered: {key_normalized}")
            return True
        
        return False
    
    def _normalize_key(self, key: str) -> str:
        """Normalize key string."""
        return key.lower().replace(" ", "").replace("_", "+")
    
    def get_registered(self) -> Dict[str, str]:
        """Get all registered bindings."""
        return self.registered_keys.copy()
    
    def is_registered(self, key: str) -> bool:
        """Check if key is registered."""
        return self._normalize_key(key) in self.registered_keys


# Convenience decorators

def hotkey(key: str, name: str = None):
    """Decorator to register a function as a hotkey."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        wrapper._hotkey_key = key
        wrapper._hotkey_name = name or func.__name__
        
        return wrapper
    
    return decorator


# Predefined keys
PRESET_KEYS = {
    "show_panel": "ctrl+shift+a",
    "quick_search": "ctrl+shift+s",
    "emergency_stop": "ctrl+shift+x",
    "screenshot": "ctrl+shift+p",
    "clipboard": "ctrl+shift+v",
    "new_task": "ctrl+shift+n",
    "settings": "ctrl+shift+,",
}


# Common key patterns
KEY_PATTERNS = {
    "modifier": ["ctrl", "alt", "shift", "cmd", "meta"],
    "navigation": ["up", "down", "left", "right", "home", "end", "pgup", "pgdn"],
    "function": [f"f{i}" for i in range(1, 13)],
    "letters": [chr(i) for i in range(ord("a"), ord("z") + 1)],
    "numbers": [str(i) for i in range(10)],
}


# Singleton
_keybinder: Optional[Keybinder] = None


def get_keybinder() -> Keybinder:
    """Get or create keybinder singleton."""
    global _keybinder
    if _keybinder is None:
        _keybinder = Keybinder()
    return _keybinder


def is_hotkey_available() -> bool:
    """Check if hotkeys are available."""
    return KEYBOARD_AVAILABLE


# Helper to wait for key press (blocking)
def wait_for_key(key: str = "escape", timeout: float = None):
    """Wait for a specific key press."""
    if not KEYBOARD_AVAILABLE:
        return False
    
    try:
        if timeout:
            keyboard.wait(key, timeout)
        else:
            keyboard.wait(key)
        return True
    except:
        return False


__all__ = [
    "Keybinder",
    "get_keybinder",
    "hotkey",
    "PRESET_KEYS",
    "is_hotkey_available",
    "wait_for_key"
]