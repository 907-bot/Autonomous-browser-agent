#!/usr/bin/env python3
"""
🤖 Autonomous Task Engine
======================
Executes browser-based tasks autonomously.
Connects with the existing autonomous browser agent.
"""

import asyncio
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger("TaskEngine")


class TaskExecutor:
    """
    Executes tasks using the browser automation engine.
    Supports synchronous and async execution.
    """
    
    def __init__(self):
        self.browser = None
        self.playwright = None
        self.history: List[Dict] = []
        self.active_task: Optional[Dict] = None
        
        # Attempt to import browser agent
        try:
            from agent.autonomous_agent import AutonomousBrowserAgent
            self.browser_agent_class = AutonomousBrowserAgent
            self.browser_available = True
            logger.info("✅ Browser agent integration loaded")
        except ImportError:
            self.browser_agent_class = None
            self.browser_available = False
            logger.warning("⚠️ Browser agent not available, using simulation mode")
    
    async def start_browser(self, headless: bool = False):
        """Start the browser."""
        if not self.browser_available:
            return
        
        try:
            self.playwright = await AsyncPlaywright().start()
            self.browser = await self.playwright.chromium.launch(headless=headless)
            logger.info("🖥️ Browser started")
        except Exception as e:
            logger.error(f"Browser start failed: {e}")
            self.browser_available = False
    
    async def execute_task_async(
        self,
        task: str,
        url: str = "",
        max_steps: int = 30
    ) -> Dict[str, Any]:
        """
        Execute a task asynchronously.
        
        Args:
            task: Task description
            url: Starting URL
            max_steps: Maximum steps
            
        Returns:
            Result dictionary
        """
        if not self.browser_available:
            return {
                "success": False,
                "error": "Browser not available",
                "task": task,
                "simulated": True
            }
        
        try:
            # Execute via browser agent
            result = await self.browser_agent_class.execute_task(
                task=task,
                url=url or "https://www.google.com",
                max_steps=max_steps
            )
            return result
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "task": task
            }
    
    def execute_task_sync(
        self,
        task: str,
        url: str = "",
        max_steps: int = 30
    ) -> Dict[str, Any]:
        """Execute a task synchronously."""
        return asyncio.run(self.execute_task_async(task, url, max_steps))
    
    async def close(self):
        """Close browser."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    def get_history(self) -> List[Dict]:
        """Get task execution history."""
        return self.history


# Simplified runner for quick execution
_runner: Optional[TaskExecutor] = None


def get_runner() -> TaskExecutor:
    """Get or create task executor singleton."""
    global _runner
    if _runner is None:
        _runner = TaskExecutor()
    return _runner


async def quick_execute(task: str, url: str = "", max_steps: int = 30) -> Dict[str, Any]:
    """
    Quick task execution helper.
    
    Args:
        task: What to do
        url: Starting URL
        max_steps: Max browser steps
        
    Returns:
        Result dictionary
    """
    runner = get_runner()
    result = await runner.execute_task_async(task, url, max_steps)
    return result


def quick_execute_sync(task: str, url: str = "", max_steps: int = 30) -> Dict[str, Any]:
    """Synchronous version of quick_execute."""
    return asyncio.run(quick_execute(task, url, max_steps))


# Export commonly used items
__all__ = [
    "TaskExecutor",
    "quick_execute",
    "quick_execute_sync",
    "get_runner"
]