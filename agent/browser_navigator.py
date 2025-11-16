"""
Browser Navigator - Low-level browser interaction and action execution
"""

from typing import Dict, List, Optional, Any
from playwright.async_api import Page
from loguru import logger
import asyncio


class BrowserNavigator:
    """
    Handles low-level browser interactions using Playwright.
    """
    
    def __init__(self, page: Page):
        self.page = page
        self.action_history: List[Dict] = []
        logger.info("Browser Navigator initialized")
    
    async def execute_action(self, action_type: str, target_info: Dict) -> Dict[str, Any]:
        """Execute a browser action."""
        try:
            if action_type == "click":
                return await self._click_element(target_info)
            elif action_type == "type":
                return await self._type_text(target_info)
            elif action_type == "scroll":
                return await self._scroll_page(target_info)
            elif action_type == "navigate":
                return await self._navigate(target_info)
            elif action_type == "extract":
                return await self._extract_data(target_info)
            elif action_type == "wait":
                return await self._wait(target_info)
            else:
                return {"success": False, "error": f"Unknown action: {action_type}"}
        except Exception as e:
            logger.error(f"Action failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _click_element(self, target_info: Dict) -> Dict:
        """Click on an element."""
        selector = target_info.get("selector")
        try:
            if selector:
                await self.page.click(selector, timeout=5000)
            else:
                x = target_info.get("x", 0)
                y = target_info.get("y", 0)
                await self.page.mouse.click(x, y)
            
            await self.page.wait_for_load_state("networkidle", timeout=10000)
            logger.info(f"Clicked: {selector}")
            
            return {
                "success": True,
                "action": "click",
                "target": selector,
                "state_change": "Element clicked successfully"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _type_text(self, target_info: Dict) -> Dict:
        """Type text into input field."""
        selector = target_info.get("selector")
        text = target_info.get("text", "")
        
        try:
            if selector:
                await self.page.fill(selector, text)
            else:
                await self.page.focus("input")
                await self.page.keyboard.type(text)
            
            logger.info(f"Typed: {text[:50]}")
            return {
                "success": True,
                "action": "type",
                "text": text,
                "state_change": "Text entered successfully"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _scroll_page(self, target_info: Dict) -> Dict:
        """Scroll the page."""
        direction = target_info.get("direction", "down")
        amount = target_info.get("amount", 500)
        
        try:
            if direction == "down":
                await self.page.evaluate(f"window.scrollBy(0, {amount})")
            elif direction == "up":
                await self.page.evaluate(f"window.scrollBy(0, -{amount})")
            elif direction == "bottom":
                await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            
            await asyncio.sleep(0.5)
            return {
                "success": True,
                "action": "scroll",
                "direction": direction,
                "state_change": f"Scrolled {direction}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _navigate(self, target_info: Dict) -> Dict:
        """Navigate to URL."""
        url = target_info.get("url", "")
        
        try:
            await self.page.goto(url, wait_until="networkidle", timeout=30000)
            logger.info(f"Navigated to: {url}")
            return {
                "success": True,
                "action": "navigate",
                "url": url,
                "state_change": f"Navigated to {url}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _extract_data(self, target_info: Dict) -> Dict:
        """Extract data from page."""
        selector = target_info.get("selector")
        
        try:
            if selector:
                elements = await self.page.query_selector_all(selector)
                data = [await e.text_content() for e in elements]
            else:
                data = await self.page.evaluate("""
                    () => Array.from(document.body.querySelectorAll('p, h1, h2, h3'))
                        .map(el => el.textContent.trim())
                        .filter(text => text.length > 0)
                """)
            
            return {
                "success": True,
                "action": "extract",
                "data": data,
                "state_change": f"Extracted {len(data)} items"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _wait(self, target_info: Dict) -> Dict:
        """Wait for duration."""
        duration = target_info.get("duration", 1.0)
        await asyncio.sleep(duration)
        return {"success": True, "action": "wait", "duration": duration}

