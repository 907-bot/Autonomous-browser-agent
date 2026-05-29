"""
Minimal HTTP Backend for Autonomous Browser Agent
Using Starlette directly to avoid pydantic issues on Render
"""

import json
from datetime import datetime
from typing import Optional, List, Dict, Any

from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.requests import Request

# Configure logging
try:
    from loguru import logger
    logger.remove()
    logger.add(__file__, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


# ============================================================================
# STATE MANAGEMENT
# ============================================================================

class AgentState:
    """Global state for the agent"""
    
    def __init__(self):
        self.agent = None
        self.task_history: List[Dict[str, Any]] = []
        self.max_history = 10
        self.status = "idle"
        self.last_action = ""
        self.last_update = datetime.now().isoformat()
        self.active_task: Optional[str] = None


state = AgentState()


# ============================================================================
# HANDLERS
# ============================================================================

async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "agent_available": False,
        "timestamp": datetime.now().isoformat(),
    })


async def execute_task(request: Request) -> JSONResponse:
    """Execute a web automation task"""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    
    task = data.get("task", "")
    url = data.get("url", "")
    headless = data.get("headless", True)
    max_steps = data.get("max_steps", 30)
    
    if not task.strip():
        return JSONResponse({"error": "Task description cannot be empty"}, status_code=400)
    
    if not url.strip():
        return JSONResponse({"error": "URL cannot be empty"}, status_code=400)
    
    try:
        logger.info(f"Executing task: {task}")
        logger.info(f"URL: {url}")
        
        state.status = "running"
        state.active_task = task
        
        # Demo response (actual agent not available)
        history_entry = {
            "id": str(len(state.task_history) + 1),
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "url": url,
            "success": True,
            "steps_completed": 5,
            "status": "completed",
        }
        state.task_history.append(history_entry)
        
        if len(state.task_history) > state.max_history:
            state.task_history = state.task_history[-state.max_history:]
        
        state.status = "completed"
        state.last_action = f"Completed: {task[:50]}..."
        state.last_update = datetime.now().isoformat()
        state.active_task = None
        
        return JSONResponse({
            "success": True,
            "status_text": f"Success! Task: {task} completed.",
            "results_json": json.dumps({"success": True, "demo": True, "task": task}),
            "history_text": format_history(),
        })
        
    except Exception as e:
        logger.error(f"Task execution failed: {str(e)}")
        state.status = "error"
        state.last_action = f"Failed: {str(e)}"
        state.last_update = datetime.now().isoformat()
        state.active_task = None
        return JSONResponse({"error": str(e)}, status_code=500)


async def decompose_task(request: Request) -> JSONResponse:
    """Decompose a task into sub-tasks"""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    
    task = data.get("task", "")
    
    if not task.strip():
        return JSONResponse({"error": "Task description cannot be empty"}, status_code=400)
    
    # Demo decomposition
    sub_tasks = [
        f"Step 1: Analyze the task '{task[:30]}...'",
        f"Step 2: Navigate to the target website",
        f"Step 3: Locate the required elements",
        f"Step 4: Execute the required actions",
        f"Step 5: Verify the results",
    ]
    
    result = f"Task Decomposition\n\nOriginal Task: {task}\n\nSub-tasks ({len(sub_tasks)} steps)\n\n"
    for i, sub_task in enumerate(sub_tasks, 1):
        result += f"{i}. {sub_task}\n"
    
    return JSONResponse({
        "decomposition": result,
        "error": None,
    })


async def get_status(request: Request) -> JSONResponse:
    """Get agent status"""
    return JSONResponse({
        "status": state.status,
        "active_task": state.active_task,
        "queue_length": len([t for t in state.task_history if t.get("status") == "pending"]),
        "history_count": len(state.task_history),
        "last_action": state.last_action,
        "last_update": state.last_update,
    })


async def get_history(request: Request) -> JSONResponse:
    """Get task history"""
    return JSONResponse({
        "history": state.task_history,
        "count": len(state.task_history),
    })


def format_history() -> str:
    """Format task history for display"""
    if not state.task_history:
        return "No tasks executed yet"
    
    history_text = "Recent Tasks\n\n"
    for i, task in enumerate(reversed(state.task_history), 1):
        status = "Success" if task["success"] else "Warning"
        history_text += f"{i}. {status} {task['task']}\n"
        history_text += f"   URL: {task['url']}\n"
        history_text += f"   Steps: {task['steps_completed']}\n\n"
    
    return history_text


# ============================================================================
# APP
# ============================================================================

routes = [
    Route("/health", health_check),
    Route("/execute", execute_task, methods=["POST"]),
    Route("/decompose", decompose_task, methods=["POST"]),
    Route("/status", get_status),
    Route("/history", get_history),
]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    ),
]

app = Starlette(routes=routes, middleware=middleware)


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)