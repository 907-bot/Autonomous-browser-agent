from typing import List, Dict, Optional
from loguru import logger


class PlannerAgent:
    """
    High-level planner that decomposes complex tasks into manageable sub-tasks.
    Uses Hierarchical Task Network (HTN) planning approach.
    """
    
    def __init__(self):
        """Initialize the planner agent."""
        self.task_templates = self._load_task_templates()
        self.decomposition_history = []
        logger.info("Planner Agent initialized with HTN planning")
    
    def decompose_task(self, task: str) -> List[str]:
        """
        Decompose a high-level natural language task into sub-tasks.
        
        Args:
            task: Natural language task description
            
        Returns:
            List of sub-task descriptions
        """
        task_lower = task.lower()
        logger.info(f"Decomposing task: {task}")
        
        # Pattern matching for common task types
        if "search" in task_lower or "find" in task_lower or "look for" in task_lower:
            sub_tasks = self._decompose_search_task(task)
        elif "buy" in task_lower or "purchase" in task_lower or "add to cart" in task_lower:
            sub_tasks = self._decompose_purchase_task(task)
        elif "fill" in task_lower or "form" in task_lower or "submit" in task_lower:
            sub_tasks = self._decompose_form_task(task)
        elif "extract" in task_lower or "scrape" in task_lower or "get data" in task_lower:
            sub_tasks = self._decompose_extraction_task(task)
        elif "navigate" in task_lower or "go to" in task_lower or "visit" in task_lower:
            sub_tasks = self._decompose_navigation_task(task)
        elif "login" in task_lower or "sign in" in task_lower:
            sub_tasks = self._decompose_login_task(task)
        else:
            # Generic decomposition for unknown task types
            sub_tasks = self._decompose_generic_task(task)
        
        # Record decomposition history
        self.decomposition_history.append({"task": task, "sub_tasks": sub_tasks})
        
        logger.info(f"Decomposed into {len(sub_tasks)} sub-tasks")
        return sub_tasks
    
    def _decompose_search_task(self, task: str) -> List[str]:
        """Decompose search/find tasks."""
        search_query = self._extract_search_query(task)
        return [
            "Locate search box or search field on the page",
            f"Click on search input field",
            f"Enter search query: '{search_query}'",
            "Submit search (press Enter or click search button)",
            "Wait for search results to load",
            "Verify that search results are displayed"
        ]
    
    def _decompose_purchase_task(self, task: str) -> List[str]:
        """Decompose e-commerce purchase tasks."""
        return [
            "Search for the desired product",
            "Navigate through search results and select product",
            "Review product details and pricing",
            "Select product options (size, color, quantity, etc.)",
            "Add product to shopping cart",
            "Navigate to shopping cart",
            "Review items in cart",
            "Proceed to checkout",
            "Fill in shipping address information",
            "Select shipping method",
            "Enter payment information",
            "Review order summary",
            "Place order/Confirm purchase"
        ]
    
    def _decompose_form_task(self, task: str) -> List[str]:
        """Decompose form filling and submission tasks."""
        return [
            "Locate the form on the page",
            "Identify all required form fields",
            "Fill text input fields with appropriate data",
            "Select options from dropdown menus",
            "Check/uncheck checkboxes as needed",
            "Select radio button options",
            "Upload files if required",
            "Review all filled information",
            "Click submit button",
            "Verify form submission success"
        ]
    
    def _decompose_extraction_task(self, task: str) -> List[str]:
        """Decompose data extraction tasks."""
        return [
            "Navigate to target page or section",
            "Identify data elements to extract",
            "Scroll page if needed to view all data",
            "Extract text content from identified elements",
            "Extract structured data (tables, lists, etc.)",
            "Format extracted data appropriately",
            "Verify completeness and accuracy of extraction"
        ]
    
    def _decompose_navigation_task(self, task: str) -> List[str]:
        """Decompose navigation tasks."""
        return [
            "Identify navigation elements (menu, links, buttons)",
            "Locate target link or navigation option",
            "Click on target navigation element",
            "Wait for page to load",
            "Verify correct page/section has been reached"
        ]
    
    def _decompose_login_task(self, task: str) -> List[str]:
        """Decompose login tasks."""
        return [
            "Navigate to login page",
            "Locate username/email input field",
            "Enter username or email",
            "Locate password input field",
            "Enter password",
            "Click login button",
            "Wait for authentication to complete",
            "Verify successful login (check for dashboard or user profile)"
        ]
    
    def _decompose_generic_task(self, task: str) -> List[str]:
        """Generic task decomposition for unknown types."""
        return [
            "Analyze current page structure and content",
            "Identify relevant interactive elements",
            "Execute primary action based on task description",
            "Monitor page state changes",
            "Verify action outcome matches expected result"
        ]
    
    def _extract_search_query(self, task: str) -> str:
        """
        Extract search query from task description.
        
        Args:
            task: Task description
            
        Returns:
            Extracted search query
        """
        keywords = [
            "search for",
            "find",
            "look for",
            "search",
            "lookup",
            "query"
        ]
        
        task_lower = task.lower()
        
        for keyword in keywords:
            if keyword in task_lower:
                parts = task_lower.split(keyword)
                if len(parts) > 1:
                    query = parts[-1].strip()
                    # Clean up the query
                    query = query.replace("on ", "").replace("in ", "").rstrip(".")
                    return query
        
        return task
    
    def _load_task_templates(self) -> Dict[str, List[str]]:
        """
        Load pre-defined task decomposition templates.
        
        Returns:
            Dictionary of task templates
        """
        return {
            "login": [
                "Find login button or link",
                "Click login",
                "Enter username/email",
                "Enter password",
                "Click submit"
            ],
            "register": [
                "Find registration form",
                "Fill required fields",
                "Accept terms and conditions",
                "Submit registration"
            ],
            "contact": [
                "Find contact form",
                "Fill name and email",
                "Enter message content",
                "Submit form"
            ],
            "subscribe": [
                "Find subscription form",
                "Enter email address",
                "Select subscription tier",
                "Enter payment information",
                "Complete subscription"
            ]
        }
    
    def verify_task_completion(self, task: str, results: Dict) -> bool:
        """
        Verify if a task has been successfully completed.
        
        Args:
            task: Original task description
            results: Execution results
            
        Returns:
            True if task appears to be complete
        """
        # Check if all steps succeeded
        if "steps" in results:
            completed_steps = sum(1 for step in results["steps"] if step.get("success", False))
            total_steps = len(results["steps"])
            
            # Task is successful if most steps succeeded
            success_rate = completed_steps / total_steps if total_steps > 0 else 0
            is_complete = success_rate >= 0.7  # 70% success threshold
            
            logger.info(f"Task completion check: {success_rate:.0%} success rate - {'PASS' if is_complete else 'FAIL'}")
            return is_complete
        
        return results.get("success", False)
    
    def get_decomposition_history(self) -> List[Dict]:
        """
        Get history of task decompositions.
        
        Returns:
            List of decomposition records
        """
        return self.decomposition_history
    
    def clear_history(self):
        """Clear decomposition history."""
        self.decomposition_history.clear()
        logger.info("Decomposition history cleared")
