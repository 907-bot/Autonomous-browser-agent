#!/usr/bin/env python3
"""
🤖 Ultimate Proactive Agent - Main Entry Point
=====================================
An always-on desktop assistant living in your system tray.
"""

import sys
import os
import threading
import time
import json
import socket
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import logging
import signal

# Third-party imports for desktop functionality
try:
    import pystray
    from PIL import Image, ImageDraw
except ImportError:
    print("Installing pystray and Pillow...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pystray", "Pillow", "-q"])
    import pystray
    from PIL import Image, ImageDraw

try:
    import customtkinter as ctk
except ImportError:
    print("Installing customtkinter...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "customtkinter", "-q"])
    import customtkinter as ctk

try:
    from plyer import notification
except ImportError:
    print("Installing plyer...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plyer", "-q"])
    from plyer import notification

try:
    import keyboard
except ImportError:
    print("Installing keyboard...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "keyboard", "-q"])
    import keyboard

# Setup logging
LOG_DIR = Path.home() / ".proactive_agent" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(level)-8s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "agent.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ProactiveAgent")

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    "hotkey": "ctrl+shift+a",
    "start_minimized": False,
    "auto_start": True,
    "notifications": True,
    "autonomous_mode": True,
    "check_interval": 60,  # seconds between autonomous checks
    "theme": "dark-blue",
    "window_size": {"width": 420, "height": 600},
    "position": {"x": 50, "y": 50},
}

CONFIG_FILE = Path.home() / ".proactive_agent" / "config.json"


def load_config() -> Dict[str, Any]:
    """Load configuration from file."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except Exception as e:
            logger.warning(f"Config load error: {e}")
    return DEFAULT_CONFIG.copy()


def save_config(config: Dict[str, Any]):
    """Save configuration to file."""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


# ============================================================================
# SYSTEM TRAY ICON
# ============================================================================

def create_tray_icon(size: int = 64) -> Image.Image:
    """Create the tray icon image."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a robot face
    center = size // 2
    radius = size // 2 - 4
    
    # Circle face
    draw.ellipse([4, 4, size - 4, size - 4], fill="#4A90D9", outline="#2E5C9A", width=3)
    
    # Eyes
    eye_offset = size // 5
    eye_size = 8
    draw.ellipse([center - eye_offset - eye_size//2, center - eye_offset - eye_size//2,
                 center - eye_offset + eye_size//2, center - eye_offset + eye_size//2], fill="white")
    draw.ellipse([center + eye_offset - eye_size//2, center - eye_offset - eye_size//2,
                 center + eye_offset + eye_size//2, center - eye_offset + eye_size//2], fill="white")
    
    # Pupils
    pupil_size = 4
    draw.ellipse([center - eye_offset - pupil_size//2, center - eye_offset - pupil_size//2,
                 center - eye_offset + pupil_size//2, center - eye_offset + pupil_size//2], fill="#1a1a2e")
    draw.ellipse([center + eye_offset - pupil_size//2, center - eye_offset - pupil_size//2,
                 center + eye_offset + pupil_size//2, center - eye_offset + pupil_size//2], fill="#1a1a2e")
    
    # Smile
    mouth_y = center + size // 6
    draw.arc([center - 15, mouth_y - 10, center + 15, mouth_y + 10], 
            start=0, end=180, fill="white", width=3)
    
    return img


# ============================================================================
# AUTONOMOUS AGENT CORE
# ============================================================================

class ProactiveAgentCore:
    """
    Core autonomous agent that monitors and acts proactively.
    Integrates with the existing browser agent logic.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = False
        self.task_history: List[Dict] = []
        self.queued_tasks: List[Dict] = []
        self.active_task: Optional[Dict] = None
        
        # State
        self.status = "idle"
        self.last_action = ""
        self.last_update = datetime.now()
        
        logger.info("🤖 Proactive Agent Core initialized")
    
    def start(self):
        """Start autonomous monitoring."""
        self.running = True
        self.status = "monitoring"
        logger.info("▶️ Autonomous mode started")
    
    def stop(self):
        """Stop autonomous monitoring."""
        self.running = False
        self.status = "idle"
        logger.info("⏹️ Autonomous mode stopped")
    
    def queue_task(self, task: str, url: str = "", context: Dict = None):
        """Add a task to the queue."""
        task_obj = {
            "id": len(self.task_history) + 1,
            "task": task,
            "url": url,
            "context": context or {},
            "created_at": datetime.now().isoformat(),
            "status": "pending",
        }
        self.queued_tasks.append(task_obj)
        logger.info(f"📝 Task queued: {task}")
        return task_obj["id"]
    
    def get_next_task(self) -> Optional[Dict]:
        """Get next task from queue."""
        if self.queued_tasks:
            task = self.queued_tasks.pop(0)
            self.active_task = task
            task["status"] = "running"
            return task
        return None
    
    def complete_task(self, task_id: int, success: bool, result: Any = None):
        """Mark task as completed."""
        if self.active_task and self.active_task["id"] == task_id:
            self.active_task["completed_at"] = datetime.now().isoformat()
            self.active_task["success"] = success
            self.active_task["result"] = result
            self.task_history.append(self.active_task)
            self.active_task = None
            
            if success:
                self.status = "completed"
                self.last_action = f"Completed: {result}"
            else:
                self.status = "error"
                self.last_action = f"Failed: {result}"
            
            self.last_update = datetime.now()
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            "status": self.status,
            "active_task": self.active_task,
            "queue_length": len(self.queued_tasks),
            "history_count": len(self.task_history),
            "last_action": self.last_action,
            "last_update": self.last_update.isoformat(),
        }


# ============================================================================
# FLOATING PANEL UI
# ============================================================================

class AgentPanel(ctk.CTkToplevel):
    """
    Floating mini-panel for quick interaction.
    Appears on hotkey press or tray click.
    """
    
    def __init__(self, parent, agent_core: ProactiveAgentCore):
        super().__init__(parent)
        
        self.agent_core = agent_core
        self.config = parent.config
        
        # Window setup
        self.title("🤖 Proactive Agent")
        self.overrideredirect(True)  # Frameless
        self.attributes("-topmost", True)
        self.attributes("-alpha", 0.95)
        
        # Size and position
        w, h = self.config["window_size"]["width"], self.config["window_size"]["height"]
        x, y = self.config["position"]["x"], self.config["position"]["y"]
        self.geometry(f"{w}x{h}+{x}+{y}")
        
        # Make resizable but compact
        self.resizable(False, False)
        
        # Dragging
        self._drag_x = None
        self._drag_y = None
        self.bind("<Button-1>", self._start_drag)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", self._stop_drag)
        
        # Build UI
        self._setup_ui()
        
        # Start update loop
        self._update_loop()
        
        logger.info("🖼️ Agent panel created")
    
    def _setup_ui(self):
        """Build the panel UI."""
        # Theme
        ctk.set_appearance_mode(self.config.get("theme", "dark"))
        ctk.set_default_color_theme("blue")
        
        # Main container
        self.main_frame = ctk.CTkFrame(self, corner_radius=15)
        self.main_frame.pack(fill="both", expand=True, padx=4, pady=4)
        
        # Header
        header = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(10, 5))
        
        self.status_label = ctk.CTkLabel(
            header, 
            text="🤖 Online",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#4ADE80"
        )
        self.status_label.pack(side="left")
        
        close_btn = ctk.CTkButton(
            header,
            text="✕",
            width=30,
            height=30,
            command=self.hide,
            fg_color="transparent",
            hover_color="#DC2626"
        )
        close_btn.pack(side="right")
        
        # Status card
        status_card = ctk.CTkFrame(self.main_frame, corner_radius=12)
        status_card.pack(fill="x", padx=10, pady=5)
        
        self.status_icon = ctk.CTkLabel(
            status_card,
            text="⚡",
            font=ctk.CTkFont(size=32)
        )
        self.status_icon.pack(side="left", padx=15, pady=15)
        
        status_info = ctk.CTkFrame(status_card, fg_color="transparent")
        status_info.pack(side="left", fill="both", expand=True, pady=10)
        
        self.status_text = ctk.CTkLabel(
            status_info,
            text="Ready",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.status_text.pack(anchor="w")
        
        self.status_detail = ctk.CTkLabel(
            status_info,
            text="Monitoring for tasks...",
            font=ctk.CTkFont(size=11),
            text_color="gray70"
        )
        self.status_detail.pack(anchor="w")
        
        # Input section
        input_card = ctk.CTkFrame(self.main_frame, corner_radius=12)
        input_card.pack(fill="both", expand=True, padx=10, pady=10)
        
        input_label = ctk.CTkLabel(
            input_card,
            text="What would you like me to do?",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        input_label.pack(anchor="w", padx=15, pady=(10, 5))
        
        self.task_input = ctk.CTkTextbox(
            input_card,
            height=80,
            corner_radius=8,
            font=ctk.CTkFont(size=13),
            wrap="word"
        )
        self.task_input.pack(fill="x", padx=10, pady=5)
        self.task_input.insert("1.0", "Enter task...")
        self.task_input.bind("<FocusIn>", lambda e: self._clear_placeholder())
        
        # URL input
        url_frame = ctk.CTkFrame(input_card, fg_color="transparent")
        url_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(url_frame, text="🌐 URL:", font=ctk.CTkFont(size=11)).pack(side="left")
        
        self.url_input = ctk.CTkEntry(
            url_frame,
            placeholder_text="https:// (optional)"
        )
        self.url_input.pack(side="left", fill="both", expand=True, padx=(5, 0))
        
        # Quick actions
        quick_label = ctk.CTkLabel(
            input_card,
            text="⚡ Quick Actions",
            font=ctk.CTkFont(size=11, weight="bold")
        )
        quick_label.pack(anchor="w", padx=15, pady=(10, 5))
        
        quick_buttons = ctk.CTkFrame(input_card, fg_color="transparent")
        quick_buttons.pack(padx=10, pady=5)
        
        # Quick action buttons
        for i, (label, task) in enumerate([
            ("🔍 Quick Search", "search"),
            ("📰 News", "news"),
            ("📧 Email", "email"),
            ("🛒 Shop", "shop"),
        ]):
            btn = ctk.CTkButton(
                quick_buttons,
                text=label,
                width=80,
                height=32,
                command=lambda t=task: self._quick_action(t),
                corner_radius=8,
                font=ctk.CTkFont(size=10)
            )
            btn.grid(row=i//2, column=i%2, padx=3, pady=3)
        
        # Execute button
        execute_btn = ctk.CTkButton(
            self.main_frame,
            text="▶️ Execute",
            command=self._execute_task,
            height=44,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#2563EB",
            hover_color="#1D4ED8",
            corner_radius=10
        )
        execute_btn.pack(fill="x", padx=15, pady=(0, 10))
    
    def _clear_placeholder(self):
        """Clear placeholder text."""
        if self.task_input.get("1.0", "end-1c").strip() == "Enter task...":
            self.task_input.delete("1.0", "end")
    
    def _quick_action(self, action: str):
        """Execute a quick action."""
        templates = {
            "search": ("Search the web for {query}", "https://www.google.com"),
            "news": ("Show latest {topic} news", "https://news.google.com"),
            "email": ("Compose new email", "https://mail.google.com"),
            "shop": ("Browse {category} deals", "https://amazon.com"),
        }
        task_template, url = templates.get(action, ("", ""))
        self.task_input.delete("1.0", "end")
        self.task_input.insert("1.0", action.title() + " for ")
        self.url_input.delete(0, "end")
        self.url_input.insert(0, url)
    
    def _execute_task(self):
        """Execute the entered task."""
        task = self.task_input.get("1.0", "end-1c").strip()
        url = self.url_input.get().strip()
        
        if not task or task == "Enter task...":
            self.flash_red()
            return
        
        # Queue task
        self.agent_core.queue_task(task, url)
        
        # Clear inputs
        self.task_input.delete("1.0", "end")
        self.task_input.insert("1.0", "Enter task...")
        self.url_input.delete(0, "end")
        
        # Feedback
        self.status_text.configure(text="Task Queued!")
        self.status_detail.configure(text=f'"{task[:30]}..."')
        
        if self.config.get("notifications"):
            notification.notify(
                title="🤖 Task Queued",
                message=task[:50],
                timeout=3
            )
        
        logger.info(f"📝 Task queued: {task}")
    
    def flash_red(self):
        """Flash red to indicate error."""
        self.status_icon.configure(text="⚠️")
        self.after(500, lambda: self.status_icon.configure(text="⚡"))
    
    def _update_loop(self):
        """Periodic UI update."""
        status = self.agent_core.get_status()
        
        # Update status display
        if status["active_task"]:
            self.status_text.configure(text="Running Task")
            self.status_detail.configure(text=status["active_task"].get("task", "")[:40])
        else:
            status_map = {
                "idle": (" Ready", "Waiting for tasks..."),
                "monitoring": ("⚡ Monitoring", "Watching for opportunities..."),
                "running": ("🔄 Working", "Executing task..."),
                "completed": ("✓ Complete", "Task finished"),
                "error": ("⚠️ Error", "Check logs"),
            }
            text, detail = status_map.get(status["status"], ("•", status["status"]))
            self.status_text.configure(text=text)
            self.status_detail.configure(text=detail)
        
        # Update queue count
        if status["queue_length"] > 0:
            self.status_label.configure(text=f"📋 {status['queue_length']} pending")
        
        self.after(1000, self._update_loop)
    
    def _start_drag(self, event):
        """Start window drag."""
        self._drag_x = event.x
        self._drag_y = event.y
    
    def _drag(self, event):
        """Drag window."""
        if self._drag_x is not None:
            deltax = event.x - self._drag_x
            deltay = event.y - self._drag_y
            x = self.winfo_x() + deltax
            y = self.winfo_y() + deltay
            self.geometry(f"+{x}+{y}")
    
    def _stop_drag(self, event):
        """Stop window drag."""
        self._drag_x = None
        self._drag_y = None
    
    def show(self):
        """Show and bring to front."""
        self.deiconify()
        self.lift()
        self.focus_force()
    
    def hide(self):
        """Hide window (minimize to tray instead)."""
        self.withdraw()


# ============================================================================
# MAIN APPLICATION
# ============================================================================

class ProactiveAgentApp:
    """
    Main application controller.
    Manages system tray, UI, and autonomous core.
    """
    
    def __init__(self):
        # Load config
        self.config = load_config()
        
        # Core agent
        self.agent_core = ProactiveAgentCore(self.config)
        
        # UI Elements
        self.icon: Optional[pystray.Icon] = None
        self.panel: Optional[AgentPanel] = None
        
        # Running state
        self.running = False
        self._hotkey_registered = False
        
        logger.info("🚀 Proactive Agent App initializing...")
    
    def start(self):
        """Start the application."""
        logger.info("▶️ Starting Proactive Agent...")
        
        # Start autonomous core
        if self.config.get("autonomous_mode", True):
            self.agent_core.start()
        
        # Create tray icon
        self._create_tray()
        
        # Register hotkey
        self._register_hotkey()
        
        # Auto-start if configured
        if not self.config.get("start_minimized", False):
            self.show_panel()
        
        self.running = True
        
        # Send startup notification
        if self.config.get("notifications"):
            notification.notify(
                title="🤖 Proactive Agent",
                message="I'm online and watching! Press Ctrl+Shift+A for quick access.",
                timeout=5
            )
        
        logger.info("✅ Proactive Agent running")
        logger.info(f"   Hotkey: {self.config['hotkey'].upper()}")
        
        # Main loop (keyboard requires this)
        try:
            keyboard.wait()
        except KeyboardInterrupt:
            self.shutdown()
    
    def _create_tray(self):
        """Create system tray icon."""
        # Create image
        image = create_tray_icon()
        
        # Menu items
        menu = pystray.Menu(
            pystray.MenuItem("Show Agent", self.show_panel, default=True),
            pystray.MenuItem("───", None),
            pystray.MenuItem("Toggle Autonomous", self.toggle_autonomous),
            pystray.MenuItem("───", None),
            pystray.MenuItem("Quit", self.shutdown)
        )
        
        # Create icon
        self.icon = pystray.Icon(
            "proactive_agent",
            image,
            "🤖 Proactive Agent",
            menu
        )
        
        # Run in separate thread
        self.icon_thread = threading.Thread(target=self.icon.run, daemon=True)
        self.icon_thread.start()
        
        logger.info("🖥️ System tray icon created")
    
    def _register_hotkey(self):
        """Register global hotkey."""
        hotkey = self.config.get("hotkey", "ctrl+shift+a")
        
        try:
            keyboard.add_hotkey(hotkey, self.toggle_panel)
            self._hotkey_registered = True
            logger.info(f"⌨️ Hotkey registered: {hotkey.upper()}")
        except Exception as e:
            logger.error(f"Hotkey registration failed: {e}")
    
    def show_panel(self, *args):
        """Show the agent panel."""
        if self.panel is None:
            # Create hidden root window for CTk
            self.root = ctk.CTk()
            self.root.withdraw()
            self.root.attributes("-topmost", True)
            self.panel = AgentPanel(self.root, self.agent_core)
        
        self.panel.show()
    
    def toggle_panel(self):
        """Toggle panel visibility."""
        if self.panel is None or not self.panel.winfo_exists():
            self.show_panel()
        elif self.panel.winfo_viewable():
            self.panel.hide()
        else:
            self.panel.show()
    
    def toggle_autonomous(self):
        """Toggle autonomous mode."""
        if self.agent_core.running:
            self.agent_core.stop()
            if self.config.get("notifications"):
                notification.notify(
                    title="🤖 Proactive Agent",
                    message="Autonomous mode disabled",
                    timeout=3
                )
        else:
            self.agent_core.start()
            if self.config.get("notifications"):
                notification.notify(
                    title="🤖 Proactive Agent",
                    message="Autonomous mode enabled",
                    timeout=3
                )
    
    def shutdown(self, *args=None):
        """Shutdown the application."""
        logger.info("🛑 Shutting down...")
        
        self.running = False
        self.agent_core.stop()
        
        # Cleanup
        if self.icon:
            self.icon.stop()
        
        if keyboard.is_modifier(self.config.get("hotkey")):
            try:
                keyboard.unblock_key(self.config.get("hotkey"))
            except:
                pass
        
        if self.config.get("notifications"):
            notification.notify(
                title="🤖 Proactive Agent",
                message="Goodbye! I'll be here when you need me.",
                timeout=3
            )
        
        logger.info("👋 Proactive Agent stopped")
        
        # Exit
        sys.exit(0)


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    import warnings
    warnings.filterwarnings("ignore")
    
    # Handle signals
    signal.signal(signal.SIGINT, lambda s, f: ProactiveAgentApp().shutdown())
    signal.signal(signal.SIGTERM, lambda s, f: ProactiveAgentApp().shutdown())
    
    # Check for daemon mode
    daemon = "--daemon" in sys.argv
    
    if daemon:
        # Fork to background
        import daemonize
        pid_file = Path("/tmp/proactive_agent.pid")
        
        app = ProactiveAgentApp()
        
        def run():
            logger.info("🔄 Running in daemon mode...")
            while True:
                time.sleep(1)
                if not app.running:
                    break
        
        daemonize.daemonize(pid_file, run)
    else:
        # Normal start
        app = ProactiveAgentApp()
        app.start()


if __name__ == "__main__":
    main()