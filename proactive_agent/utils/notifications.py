#!/usr/bin/env python3
"""
🔔 Notifications Module
======================
Cross-platform desktop notifications.
"""

from typing import Optional
from pathlib import Path
import logging

logger = logging.getLogger("Notifier")

# Try plyer first (cross-platform)
try:
    from plyer import notification
    PLYER_AVAILABLE = True
except ImportError:
    PLYER_AVAILABLE = False
    notification = None

# Try pgi for Linux notifications
try:
    import pgi
    pgi.require_version("Notify", "0.7")
    from pgi import Notify
    Notify.init("proactive_agent")
    PGI_AVAILABLE = True
except Exception:
    PGI_AVAILABLE = False


class Notifier:
    """Desktop notification manager."""
    
    def __init__(self, app_name: str = "Proactive Agent", icon: str = None):
        self.app_name = app_name
        self.icon = icon or self._get_icon_path()
        self.plyer = PLYER_AVAILABLE
        self.pgi = PGI_AVAILABLE
        
        logger.info(f"🔔 Notifier initialized (plyer={PLYER_AVAILABLE}, pgi={PGI_AVAILABLE})")
    
    def _get_icon_path(self) -> Optional[str]:
        """Get default icon path."""
        return None
    
    def notify(
        self,
        title: str,
        message: str,
        timeout: int = 5,
        urgency: str = "normal"
    ):
        """
        Send a desktop notification.
        
        Args:
            title: Notification title
            message: Notification body
            timeout: Duration in seconds
            urgency: "low", "normal", "critical"
        """
        if PLYER_AVAILABLE:
            self._notify_plyer(title, message, timeout)
        elif PGI_AVAILABLE:
            self._notify_pgi(title, message)
        else:
            logger.warning(f"Notification: {title} - {message}")
    
    def _notify_plyer(self, title: str, message: str, timeout: int):
        """Send via plyer."""
        try:
            notification.notify(
                title=title,
                message=message,
                timeout=timeout,
                app_name=self.app_name
            )
        except Exception as e:
            logger.error(f"Plyer notification failed: {e}")
    
    def _notify_pgi(self, title: str, message: str):
        """Send via pgi/Notify."""
        try:
            n = Notify.Notification.new(self.app_name, title, message)
            n.show()
        except Exception as e:
            logger.error(f"pgi notification failed: {e}")
    
    def notify_success(self, message: str):
        """Send success notification."""
        self.notify("✅ Success", message)
    
    def notify_error(self, message: str):
        """Send error notification."""
        self.notify("❌ Error", message, urgency="critical")
    
    def notify_warning(self, message: str):
        """Send warning notification."""
        self.notify("⚠️ Warning", message)
    
    def notify_info(self, message: str):
        """Send info notification."""
        self.notify("ℹ️ Info", message)


# Sound effects
SOUNDS = {
    "success": "🔔",
    "error": "⛔",
    "warning": "⚠️",
    "info": "ℹ️",
    "task_complete": "✅",
}


def play_sound(sound_type: str = "success"):
    """Play a notification sound effect."""
    emoji = SOUNDS.get(sound_type, "•")
    print(f"\a{sound_type}", end="", flush=True)
    return emoji


# Singleton
_notifier: Optional[Notifier] = None


def get_notifier(app_name: str = "Proactive Agent") -> Notifier:
    """Get or create notifier singleton."""
    global _notifier
    if _notifier is None:
        _notifier = Notifier(app_name)
    return _notifier


__all__ = ["Notifier", "get_notifier", "play_sound", "SOUNDS"]