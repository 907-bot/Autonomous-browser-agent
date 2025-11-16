
import asyncio
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from pathlib import Path
import io
from PIL import Image

from loguru import logger

# Try importing MAYINI
try:
    import mayini as mn
    MAYINI_AVAILABLE = True
except ImportError:
    MAYINI_AVAILABLE = False
    logger.warning("MAYINI not available, running in simulation mode")

from playwright.async_api import async_playwright, Page, Browser

from .planner_agent import PlannerAgent
from .browser_navigator import BrowserNavigator
from .state_manager import StateManager
from mayini_integration.policy_network import MayiniPolicyNetwork
from vision.page_understanding import PageUnderstandingModule


class AutonomousBrowserAgent:
    """
    Main autonomous browser agent that orchestrates all components:
    - Task planning (decomposition into sub-goals)
    - Visual understanding (Vision Transformer page analysis)
    - Decision-making (MAYINI policy network)
    - Browser control (Playwright automation)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        headless: bool = False,
        browser_type: str = "chromium",
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        num_actions: int = 50
    ):
        """
        Initialize the autonomous browser agent.
        
        Args:
            model_path: Path to pre-trained policy network weights
            headless: Run browser in headless mode (no visual)
            browser_type: Browser type (chromium, firefox, webkit)
            embedding_dim: Dimension of state embeddings
            hidden_dim: Hidden layer dimension for MAYINI networks
            num_actions: Number of possible actions
        """
        self.headless = headless
        self.browser_type = browser_type
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        
        # Initialize neural policy network with MAYINI
        self.policy_network = MayiniPolicyNetwork(
            state_dim=embedding_dim,
            hidden_dim=hidden_dim,
            action_dim=num_actions
        )
        
        if model_path and Path(model_path).exists():
            self.policy_network.load_weights(model_path)
            logger.info(f"Loaded pre-trained policy network from {model_path}")
        
        # Initialize planning agent
        self.planner = PlannerAgent()
        
        # Initialize state encoder
        self.state_manager = StateManager(embedding_dim=embedding_dim)
        
        # Initialize vision module (ViT-based page understanding)
        self.vision_module = PageUnderstandingModule()
        
        # Browser components
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.navigator: Optional[BrowserNavigator] = None
        
        # Episode tracking
        self.episode_history: List[Dict] = []
        self.current_state = None
        self.epsilon = 1.0  # Exploration rate
        
        logger.info("Autonomous Browser Agent initialized successfully")
    
    async def initialize_browser(self):
        """Initialize Playwright browser instance."""
        try:
            playwright = await async_playwright().start()
            
            if self.browser_type == "chromium":
                self.browser = await playwright.chromium.launch(headless=self.headless)
            elif self.browser_type == "firefox":
                self.browser = await playwright.firefox.launch(headless=self.headless)
            elif self.browser_type == "webkit":
                self.browser = await playwright.webkit.launch(headless=self.headless)
            else:
                raise ValueError(f"Unsupported browser type: {self.browser_type}")
            
            self.page = await self.browser.new_page()
            self.navigator = BrowserNavigator(self.page)
            
            logger.info(f"{self.browser_type} browser initialized (headless={self.headless})")
        except Exception as e:
            logger.error(f"Failed to initialize browser: {str(e)}")
            raise
    
    async def execute_task(
        self,
        task: str,
        url: str,
        max_steps: int = 50,
        mode: str = "autonomous"
    ) -> Dict[str, Any]:
        """
        Execute a high-level task on a website.
        
        Args:
            task: Natural language task description
            url: Starting URL
            max_steps: Maximum number of steps to attempt
            mode: Execution mode (autonomous or human-in-loop)
            
        Returns:
            Dictionary containing task execution results
        """
        logger.info(f"Starting task execution: '{task}'")
        logger.info(f"Starting URL: {url}")
        
        # Initialize browser if not already done
        if not self.browser:
            await self.initialize_browser()
        
        # Navigate to starting URL
        await self.page.goto(url, wait_until="networkidle", timeout=30000)
        await asyncio.sleep(2)  # Allow page to stabilize
        
        # Decompose task into sub-tasks using planner
        sub_tasks = self.planner.decompose_task(task)
        logger.info(f"Task decomposed into {len(sub_tasks)} sub-tasks")
        
        # Initialize results
        results = {
            "task": task,
            "url": url,
            "sub_tasks": sub_tasks,
            "steps": [],
            "success": False,
            "extracted_data": {},
            "total_time": 0
        }
        
        # Execute each sub-task
        for i, sub_task in enumerate(sub_tasks):
            logger.info(f"Executing sub-task {i+1}/{len(sub_tasks)}: {sub_task}")
            
            # Allocate steps for this sub-task
            steps_for_subtask = max_steps // len(sub_tasks)
            
            # Execute sub-task
            sub_result = await self._execute_subtask(sub_task, max_steps=steps_for_subtask)
            
            # Record results
            results["steps"].extend(sub_result["steps"])
            results["extracted_data"].update(sub_result.get("data", {}))
            
            if not sub_result["success"] and mode == "autonomous":
                logger.warning(f"Sub-task failed but continuing (mode={mode})")
            elif not sub_result["success"]:
                logger.error(f"Sub-task failed, stopping (mode={mode})")
                break
        
        # Overall success: all sub-tasks succeeded
        results["success"] = all(s.get("success", False) for s in results["steps"]) if results["steps"] else False
        
        logger.info(f"Task execution complete. Success: {results['success']}")
        return results
    
    async def _execute_subtask(
        self, sub_task: str, max_steps: int = 20
    ) -> Dict[str, Any]:
        """
        Execute a single sub-task through observation-action loop.
        
        Args:
            sub_task: Description of sub-task to execute
            max_steps: Maximum number of action steps
            
        Returns:
            Sub-task results
        """
        steps = []
        
        for step in range(max_steps):
            try:
                # Capture current page state
                screenshot = await self.page.screenshot()
                html = await self.page.content()
                url = self.page.url
                
                logger.debug(f"Step {step+1}: Capturing page state from {url}")
                
                # Visual understanding using ViT
                visual_features = await self.vision_module.analyze_page(screenshot, html)
                logger.debug(f"Visual features extracted: {len(visual_features.get('interactive_elements', []))} elements detected")
                
                # Encode state
                state_embedding = self.state_manager.encode_state(
                    visual_features=visual_features,
                    url=url,
                    task=sub_task,
                    history=self.episode_history[-10:]  # Last 10 actions for context
                )
                
                # Select next action using policy network
                action_idx, action_type, target_info = await self._select_action(
                    state_embedding, visual_features
                )
                
                logger.debug(f"Selected action {step+1}: {action_type} (confidence: {self._get_action_confidence(action_idx)})")
                
                # Execute action in browser
                action_result = await self.navigator.execute_action(action_type, target_info)
                
                # Record step
                steps.append({
                    "step": step + 1,
                    "action": action_type,
                    "target": target_info,
                    "success": action_result["success"],
                    "state_change": action_result.get("state_change", "")
                })
                
                # Record in history for learning
                self.episode_history.append({
                    "state": state_embedding,
                    "action": action_idx,
                    "action_type": action_type,
                    "result": action_result
                })
                
                logger.debug(f"Action result: {action_result.get('state_change', 'unknown')}")
                
                # Check if sub-task is complete
                if self._check_subtask_completion(sub_task, action_result):
                    logger.success(f"Sub-task completed in {step+1} steps")
                    return {
                        "success": True,
                        "steps": steps,
                        "data": action_result.get("data", {})
                    }
                
                # Small delay between actions
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error during step {step+1}: {str(e)}")
                steps.append({
                    "step": step + 1,
                    "action": "error",
                    "success": False,
                    "error": str(e)
                })
                break
        
        logger.warning(f"Sub-task did not complete within {max_steps} steps")
        return {"success": False, "steps": steps, "data": {}}
    
    async def _select_action(
        self, state_embedding: np.ndarray, visual_features: Dict
    ) -> Tuple[int, str, Dict]:
        """
        Select next action using epsilon-greedy policy with MAYINI network.
        
        Args:
            state_embedding: Current state representation
            visual_features: Visual features from ViT
            
        Returns:
            Tuple of (action_index, action_type, target_info)
        """
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            # Explore: random action
            action_idx = np.random.randint(0, self.num_actions)
            logger.debug(f"Exploration: random action {action_idx}")
        else:
            # Exploit: use policy network
            action_logits = self.policy_network.forward(state_embedding)
            action_idx = int(np.argmax(action_logits))
            logger.debug(f"Exploitation: policy action {action_idx}")
        
        # Decay exploration rate
        self.epsilon = max(0.01, self.epsilon * 0.995)
        
        # Map action index to specific page element and action type
        action_type, target_info = await self._map_action_to_element(action_idx, visual_features)
        
        return action_idx, action_type, target_info
    
    async def _map_action_to_element(
        self, action_idx: int, visual_features: Dict
    ) -> Tuple[str, Dict]:
        """
        Map action index to specific page element and action type.
        
        Args:
            action_idx: Action index from policy network
            visual_features: Detected elements and features
            
        Returns:
            Tuple of (action_type, target_info)
        """
        # Get available interactive elements
        available_elements = visual_features.get("interactive_elements", [])
        
        if not available_elements:
            logger.debug("No interactive elements detected, waiting")
            return "wait", {}
        
        # Define action type space
        action_types = ["click", "type", "scroll", "navigate", "extract", "wait"]
        
        # Map action index to action type
        action_type_idx = action_idx % len(action_types)
        action_type = action_types[action_type_idx]
        
        # Select target element
        element_idx = (action_idx // len(action_types)) % len(available_elements)
        target_element = available_elements[element_idx]
        
        logger.debug(f"Mapped action {action_idx} -> {action_type} on element {element_idx}")
        
        return action_type, target_element
    
    def _get_action_confidence(self, action_idx: int) -> float:
        """Get confidence level for action (0-1)."""
        return 0.5 + (action_idx % 10) / 20.0
    
    def _check_subtask_completion(self, sub_task: str, action_result: Dict) -> bool:
        """
        Check if a sub-task has been successfully completed.
        
        Args:
            sub_task: Sub-task description
            action_result: Result from last action
            
        Returns:
            True if sub-task appears to be complete
        """
        # Check for success indicators in state change
        success_indicators = [
            "successfully",
            "completed",
            "found",
            "extracted",
            "submitted",
            "confirmed"
        ]
        
        state_change = action_result.get("state_change", "").lower()
        return any(indicator in state_change for indicator in success_indicators)
    
    async def close(self):
        """Close browser and cleanup resources."""
        if self.browser:
            await self.browser.close()
            logger.info("Browser closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize_browser()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
