#!/usr/bin/env python3
"""
⚡ Autonomous Monitor
===================
Monitors system state and triggers proactive actions.
"""

import time
import threading
from typing import Dict, List, Callable, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from pathlib import Path

logger = logging.getLogger("Monitor")


class MonitorState(Enum):
    IDLE = "idle"
    MONITORING = "monitoring"
    TRIGGERED = "triggered"
    ACTING = "acting"
    ERROR = "error"


@dataclass
class Trigger:
    """Represents a condition that can trigger an action."""
    name: str
    condition: Callable[[], bool]
    action: Callable[[], Any]
    interval: int = 60  # seconds
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    
    def check(self) -> bool:
        """Check if trigger fires."""
        if not self.enabled:
            return False
        if self.last_triggered:
            # Debounce - don't trigger too often
            if datetime.now() - self.last_triggered < timedelta(seconds=self.interval):
                return False
        try:
            if self.condition():
                self.last_triggered = datetime.now()
                return True
        except Exception as e:
            logger.error(f"Trigger {self.name} error: {e}")
        return False
    
    def execute(self):
        """Execute trigger action."""
        if self.check():
            logger.info(f"🔥 Trigger fired: {self.name}")
            try:
                self.action()
            except Exception as e:
                logger.error(f"Trigger {self.name} action failed: {e}")


@dataclass
class ScheduledTask:
    """A task scheduled for execution."""
    id: str
    task: Callable[[], Any]
    schedule: str  # cron-like: "every_5min", "hourly", "daily"
    enabled: bool = True
    last_run: Optional[datetime] = None
    
    def should_run(self) -> bool:
        """Check if task should run now."""
        if not self.enabled:
            return False
        
        now = datetime.now()
        
        if self.schedule == "every_1min":
            if self.last_run is None or (now - self.last_run).seconds >= 60:
                return True
        elif self.schedule == "every_5min":
            if self.last_run is None or (now - self.last_run).seconds >= 300:
                return True
        elif self.schedule == "hourly":
            if self.last_run is None or now.hour != self.last_run.hour:
                return True
        elif self.schedule == "daily":
            if self.last_run is None or now.date() != self.last_run.date():
                return True
        
        return False


class AutonomousMonitor:
    """
    Monitors conditions and executes proactive tasks.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.state = MonitorState.IDLE
        
        self.triggers: List[Trigger] = []
        self.scheduled_tasks: List[ScheduledTask] = []
        
        self.callbacks: Dict[str, List[Callable]] = {
            "on_trigger": [],
            "on_task_complete": [],
            "on_error": []
        }
        
        self.monitoring = False
        self._thread: Optional[threading.Thread] = None
        
        # Load saved state
        self.state_file = Path.home() / ".proactive_agent" / "monitor_state.json"
        self._load_state()
        
        logger.info("📺 Autonomous Monitor initialized")
    
    def add_trigger(
        self,
        name: str,
        condition: Callable[[], bool],
        action: Callable[[], Any],
        interval: int = 60
    ):
        """Add a trigger."""
        trigger = Trigger(name, condition, action, interval)
        self.triggers.append(trigger)
        logger.info(f"➕ Added trigger: {name}")
    
    def add_scheduled_task(
        self,
        task_id: str,
        task: Callable[[], Any],
        schedule: str = "every_5min"
    ):
        """Add a scheduled task."""
        scheduled = ScheduledTask(task_id, task, schedule)
        self.scheduled_tasks.append(scheduled)
        logger.info(f"📅 Added scheduled task: {task_id} ({schedule})")
    
    def on(self, event: str, callback: Callable):
        """Register event callback."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def start(self):
        """Start monitoring."""
        self.monitoring = True
        self.state = MonitorState.MONITORING
        
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        
        logger.info("▶️ Monitor started")
    
    def stop(self):
        """Stop monitoring."""
        self.monitoring = False
        self.state = MonitorState.IDLE
        
        logger.info("⏹️ Monitor stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Check triggers
                for trigger in self.triggers:
                    if trigger.check():
                        self.state = MonitorState.TRIGGERED
                        
                        # Execute action
                        for callback in self.callbacks["on_trigger"]:
                            try:
                                callback(trigger)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")
                        
                        trigger.execute()
                        self.state = MonitorState.MONITORING
                
                # Check scheduled tasks
                for task in self.scheduled_tasks:
                    if task.should_run():
                        self.state = MonitorState.ACTING
                        
                        logger.info(f"⏰ Executing scheduled task: {task.id}")
                        task.task()
                        task.last_run = datetime.now()
                        
                        for callback in self.callbacks["on_task_complete"]:
                            try:
                                callback(task)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")
                        
                        self.state = MonitorState.MONITORING
                
                # Sleep
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                self.state = MonitorState.ERROR
                
                for callback in self.callbacks["on_error"]:
                    try:
                        callback(e)
                    except:
                        pass
                
                time.sleep(10)
    
    def _load_state(self):
        """Load saved state."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                    logger.info(f"📂 Loaded state: {len(data.get('tasks', []))} tasks")
            except Exception as e:
                logger.warning(f"State load error: {e}")
    
    def _save_state(self):
        """Save current state."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump({
                    "scheduled": [
                        {"id": t.id, "last_run": t.last_run.isoformat() if t.last_run else None}
                        for t in self.scheduled_tasks
                    ]
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"State save error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitor status."""
        return {
            "state": self.state.value,
            "triggers": len(self.triggers),
            "scheduled": len(self.scheduled_tasks),
            "monitoring": self.monitoring
        }


# Built-in triggers

def create_file_trigger(path: str, check_exists: bool = True):
    """Create a file-based trigger."""
    file_path = Path(path)
    
    def condition():
        return file_path.exists() if check_exists else not file_path.exists()
    
    return condition


def create_time_trigger(hour: int, minute: int = 0):
    """Create a daily time trigger."""
    from datetime import time as dt_time
    
    target = dt_time(hour, minute)
    
    def condition():
        now = datetime.now().time()
        # Within 1 minute window
        return abs((now.hour * 60 + now.minute) - (hour * 60 + minute)) < 1
    
    return condition


# Singleton
_monitor: Optional[AutonomousMonitor] = None


def get_monitor() -> AutonomousMonitor:
    """Get or create monitor singleton."""
    global _monitor
    if _monitor is None:
        _monitor = AutonomousMonitor()
    return _monitor


__all__ = [
    "AutonomousMonitor",
    "Trigger",
    "ScheduledTask",
    "get_monitor",
    "create_file_trigger",
    "create_time_trigger",
    "MonitorState"
]