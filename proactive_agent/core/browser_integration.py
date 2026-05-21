#!/usr/bin/env python3
"""
🔗 Browser Agent Integration
============================
Bridges the proactive agent with the browser automation engine.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import traceback

logger = logging.getLogger("BrowserIntegration")

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

BROWSER_AGENT_AVAILABLE = False


def init_browser_agent():
    """Initialize the browser agent."""
    global BROWSER_AGENT_AVAILABLE
    
    try:
        from agent.autonomous_agent import AutonomousBrowserAgent
        from agent.planner_agent import PlannerAgent
        
        BROWSER_AGENT_AVAILABLE = True
        
        logger.info("✅ Browser agent integration ready")
        return True
    except ImportError as e:
        logger.warning(f"Browser agent not available: {e}")
        return False


class BrowserBridge:
    """
    Bridge between proactive agent and browser automation.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.agent = None
        self.planner = None
        self.browser_running = False
        self.current_task = None
        
        if not init_browser_agent():
            logger.warning("Browser bridge running in limited mode")
        
        BrowserBridge._initialized = True
    
    async def start_browser(self, headless: bool = False) -> bool:
        """Start browser automation."""
        if not BROWSER_AGENT_AVAILABLE:
            logger.warning("Browser not available")
            return False
        
        try:
            from agent.autonomous_agent import AutonomousBrowserAgent
            
            self.agent = AutonomousBrowserAgent(
                headless=headless,
                browser_type="chromium"
            )
            await self.agent.initialize_browser()
            
            self.browser_running = True
            logger.info("🖥️ Browser started")
            return True
        except Exception as e:
            logger.error(f"Browser start failed: {e}")
            traceback.print_exc()
            return False
    
    async def execute_task(
        self,
        task: str,
        url: str = "https://www.google.com",
        max_steps: int = 20
    ) -> Dict[str, Any]:
        """
        Execute a task in the browser.
        
        Args:
            task: Natural language task
            url: Starting URL
            max_steps: Maximum steps
            
        Returns:
            Result dictionary
        """
        if not self.browser_running:
            await self.start_browser()
        
        if not self.agent:
            return {
                "success": False,
                "error": "Browser not available",
                "task": task
            }
        
        try:
            self.current_task = task
            result = await self.agent.execute_task(
                task=task,
                url=url,
                max_steps=max_steps,
                mode="autonomous"
            )
            self.current_task = None
            return result
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "task": task
            }
    
    async def close(self):
        """Close browser."""
        if self.agent:
            await self.agent.close()
            self.agent = None
        self.browser_running = False
    
    def get_browser_status(self) -> Dict[str, Any]:
        """Get browser status."""
        return {
            "running": self.browser_running,
            "available": BROWSER_AGENT_AVAILABLE,
            "current_task": self.current_task
        }


# Quick execution helpers

_bridge: Optional[BrowserBridge] = None


def get_bridge() -> BrowserBridge:
    """Get browser bridge singleton."""
    global _bridge
    if _bridge is None:
        _bridge = BrowserBridge()
    return _bridge


async def browse(url: str, task: str = "Navigate to page", max_steps: int = 10) -> Dict[str, Any]:
    """Quick browse execution."""
    bridge = get_bridge()
    result = await bridge.execute_task(task, url, max_steps)
    return result


def browse_sync(url: str, task: str = "Navigate to page", max_steps: int = 10) -> Dict[str, Any]:
    """Synchronous browse."""
    return asyncio.run(browse(url, task, max_steps))


# Export
__all__ = [
    "BrowserBridge",
    "get_bridge",
    "browse",
    "browse_sync"
]


if __name__ == "__main__":
    # Test
    async def test():
        bridge = get_bridge()
        status = bridge.get_browser_status()
        print(f"Status: {status}")
        
        # Quick test execute
        result = await bridge.execute_task(
            "Search for Python tutorials",
            "https://google.com",
            max_steps=5
        )
        print(f"Result: {result}")
    
    asyncio.run(test())