"""
FastAPI Backend for Autonomous Browser Agent
Serves the Next.js frontend and handles agent operations
"""

import asyncio
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

# Configure logging
logger.remove()
logger.add(__file__, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

# Try importing agent components
try:
    from agent.autonomous_agent import AutonomousBrowserAgent
    from agent.planner_agent import PlannerAgent
    AGENT_AVAILABLE = True
    logger.info("Agent components loaded successfully")
except ImportError as e:
    AGENT_AVAILABLE = False
    logger.warning(f"Agent components not available: {e}")


# ============================================================================
# MODELS
# ============================================================================

class ExecuteTaskRequest(BaseModel):
    task: str = Field(..., min_length=1, description="Task description")
    url: str = Field(..., min_length=1, description="Starting URL")
    headless: bool = Field(default=True, description="Run in headless mode")
    max_steps: int = Field(default=30, ge=5, le=100, description="Maximum steps")


class DecomposeTaskRequest(BaseModel):
    task: str = Field(..., min_length=1, description="Task to decompose")


class TaskHistoryItem(BaseModel):
    id: str
    task: str
    url: str
    status: str
    success: bool
    steps_completed: int
    timestamp: str


class AgentStatusResponse(BaseModel):
    status: str
    active_task: Optional[str]
    queue_length: int
    history_count: int
    last_action: str
    last_update: str


# ============================================================================
# STATE MANAGEMENT
# ============================================================================

class AgentState:
    """Global state for the agent"""
    
    def __init__(self):
        self.agent: Optional[AutonomousBrowserAgent] = None
        self.task_history: List[Dict[str, Any]] = []
        self.max_history = 10
        self.status = "idle"
        self.last_action = ""
        self.last_update = datetime.now().isoformat()
        self.active_task: Optional[str] = None


state = AgentState()


# ============================================================================
# LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    logger.info("🚀 Starting Autonomous Browser Agent API")
    logger.info(f"Agent Available: {AGENT_AVAILABLE}")
    yield
    logger.info("👋 Shutting down...")
    if state.agent:
        try:
            await state.agent.close()
        except Exception:
            pass


# ============================================================================
# APP
# ============================================================================

app = FastAPI(
    title="Autonomous Browser Agent API",
    description="Backend API for the autonomous browser agent powered by MAYINI Framework",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# HEALTH
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent_available": AGENT_AVAILABLE,
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================================
# EXECUTE TASK
# ============================================================================

@app.post("/execute")
async def execute_task(request: ExecuteTaskRequest):
    """
    Execute a web automation task
    """
    if not AGENT_AVAILABLE:
        return {
            "success": False,
            "status_text": "❌ Demo Mode: Agent not available. This is a demo interface.",
            "results_json": json.dumps({"error": "Agent components not loaded", "demo": True}, indent=2),
            "history_text": "No tasks executed yet (demo mode)",
        }
    
    if not request.task.strip():
        raise HTTPException(status_code=400, detail="Task description cannot be empty")
    
    if not request.url.strip():
        raise HTTPException(status_code=400, detail="URL cannot be empty")
    
    try:
        logger.info(f"📝 Executing task: {request.task}")
        logger.info(f"🌐 URL: {request.url}")
        logger.info(f"⚙️ Headless: {request.headless}, Max Steps: {request.max_steps}")
        
        state.status = "running"
        state.active_task = request.task
        
        # Initialize agent
        state.agent = AutonomousBrowserAgent(
            headless=request.headless,
            browser_type="chromium",
            embedding_dim=512,
            hidden_dim=256,
            num_actions=50
        )
        
        # Execute task
        results = await state.agent.execute_task(
            task=request.task,
            url=request.url,
            max_steps=request.max_steps,
            mode="autonomous"
        )
        
        # Save to history
        history_entry = {
            "id": str(len(state.task_history) + 1),
            "timestamp": datetime.now().isoformat(),
            "task": request.task,
            "url": request.url,
            "success": results.get("success", False),
            "steps_completed": len(results.get("steps", [])),
            "status": "completed" if results.get("success") else "failed",
        }
        state.task_history.append(history_entry)
        
        # Keep only recent history
        if len(state.task_history) > state.max_history:
            state.task_history = state.task_history[-state.max_history:]
        
        # Format results
        status = "✅ Success!" if results.get("success") else "⚠️ Partial Success"
        steps_completed = len(results.get("steps", []))
        sub_tasks_completed = sum(
            1 for step in results.get("steps", []) 
            if step.get("success", False)
        )
        
        status_text = f"""
{status}

📋 **Task:** {request.task}
🌐 **URL:** {request.url}
📊 **Steps Completed:** {steps_completed}/{request.max_steps}
✅ **Successful Steps:** {sub_tasks_completed}
⏱️ **Timestamp:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

**Sub-tasks:** {len(results.get("sub_tasks", []))}
"""
        
        # Close agent
        await state.agent.close()
        state.agent = None
        
        state.status = "completed"
        state.last_action = f"Completed: {request.task[:50]}..."
        state.last_update = datetime.now().isoformat()
        state.active_task = None
        
        logger.info("✅ Task completed successfully")
        
        return {
            "success": True,
            "status_text": status_text,
            "results_json": json.dumps(results, indent=2, default=str),
            "history_text": format_history(),
        }
        
    except Exception as e:
        logger.error(f"❌ Task execution failed: {str(e)}")
        
        if state.agent:
            try:
                await state.agent.close()
            except:
                pass
            state.agent = None
        
        state.status = "error"
        state.last_action = f"Failed: {str(e)}"
        state.last_update = datetime.now().isoformat()
        state.active_task = None
        
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# DECOMPOSE TASK
# ============================================================================

@app.post("/decompose")
async def decompose_task(request: DecomposeTaskRequest):
    """
    Decompose a task into sub-tasks
    """
    if not AGENT_AVAILABLE:
        return {
            "decomposition": "Agent not available (demo mode)",
            "error": "Agent components not loaded",
        }
    
    if not request.task.strip():
        raise HTTPException(status_code=400, detail="Task description cannot be empty")
    
    try:
        planner = PlannerAgent()
        sub_tasks = planner.decompose_task(request.task)
        
        result = "📝 **Task Decomposition**\n\n"
        result += f"**Original Task:** {request.task}\n\n"
        result += f"**Sub-tasks:** ({len(sub_tasks)} steps)\n\n"
        
        for i, sub_task in enumerate(sub_tasks, 1):
            result += f"{i}. {sub_task}\n"
        
        return {
            "decomposition": result,
            "error": None,
        }
    except Exception as e:
        logger.error(f"Decomposition failed: {str(e)}")
        return {
            "decomposition": f"Error: {str(e)}",
            "error": str(e),
        }


# ============================================================================
# STATUS
# ============================================================================

@app.get("/status", response_model=AgentStatusResponse)
async def get_status():
    """
    Get agent status
    """
    return {
        "status": state.status,
        "active_task": state.active_task,
        "queue_length": len([t for t in state.task_history if t.get("status") == "pending"]),
        "history_count": len(state.task_history),
        "last_action": state.last_action,
        "last_update": state.last_update,
    }


# ============================================================================
# HISTORY
# ============================================================================

@app.get("/history")
async def get_history():
    """
    Get task history
    """
    return {
        "history": state.task_history,
        "count": len(state.task_history),
    }


# ============================================================================
# UTILITIES
# ============================================================================

def format_history() -> str:
    """Format task history for display"""
    if not state.task_history:
        return "📜 No tasks executed yet"
    
    history_text = "📜 **Recent Tasks**\n\n"
    for i, task in enumerate(reversed(state.task_history), 1):
        status = "✅" if task["success"] else "⚠️"
        history_text += f"{i}. {status} {task['task']}\n"
        history_text += f"   URL: {task['url']}\n"
        history_text += f"   Steps: {task['steps_completed']}\n"
        history_text += f"   Time: {task['timestamp']}\n\n"
    
    return history_text


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)