# 🤖 Proactive Agent Package
"""
Ultimate Proactive Agent - An always-on desktop companion.

Quick Start:
    python -m proactive_agent.main

Or from within your app:
    from proactive_agent import ProactiveAgent
    agent = ProactiveAgent()
    agent.launch()
"""

from .core.task_engine import TaskExecutor, quick_execute, get_runner
from .core.monitor import AutonomousMonitor, Trigger, get_monitor
from .utils.notifications import Notifier, get_notifier, play_sound
from .utils.keybindings import Keybinder, get_keybinder

# Version
__version__ = "1.0.0"

# Quick launch
def launch(config_path: str = None, headless: bool = False):
    """Launch the proactive agent from another app."""
    from .main import ProactiveAgentApp
    app = ProactiveAgentApp()
    app.start()

__all__ = [
    "launch",
    "TaskExecutor",
    "quick_execute",
    "get_runner",
    "AutonomousMonitor", 
    "Trigger",
    "get_monitor",
    "Notifier",
    "get_notifier",
    "play_sound",
    "Keybinder",
    "get_keybinder",
    "__version__"
]