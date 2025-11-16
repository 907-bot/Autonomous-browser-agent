
import gradio as gr
import asyncio
from typing import Dict, List, Tuple
import os
from datetime import datetime
from loguru import logger
import sys
import json

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level: <8} | {message}")

# Try importing agent components
try:
    from agent.autonomous_agent import AutonomousBrowserAgent
    from agent.planner_agent import PlannerAgent
    from mayini_integration.policy_network import MayiniPolicyNetwork
    AGENT_AVAILABLE = True
    logger.info("✅ Agent components loaded successfully")
except ImportError as e:
    AGENT_AVAILABLE = False
    logger.error(f"❌ Could not load agent: {str(e)}")


class BrowserAgentInterface:
    """Gradio interface for the autonomous browser agent."""
    
    def __init__(self):
        """Initialize the interface."""
        self.agent = None
        self.task_history: List[Dict] = []
        self.max_history = 10
        logger.info("🚀 Browser Agent Interface initialized")
    
    def execute_task_sync(
        self,
        task: str,
        url: str,
        headless: bool,
        max_steps: int
    ) -> Tuple[str, str, str]:
        """
        Synchronous wrapper for Gradio compatibility.
        
        Args:
            task: Task description
            url: Starting URL
            headless: Run headless
            max_steps: Maximum steps
            
        Returns:
            Tuple of (status, results_json, history_text)
        """
        return asyncio.run(self.execute_task_async(task, url, headless, max_steps))
    
    async def execute_task_async(
        self,
        task: str,
        url: str,
        headless: bool,
        max_steps: int
    ) -> Tuple[str, str, str]:
        """
        Execute task asynchronously.
        
        Args:
            task: Task description
            url: Starting URL
            headless: Run in headless mode
            max_steps: Maximum steps
            
        Returns:
            Tuple of (status_text, results_json, history_text)
        """
        if not AGENT_AVAILABLE:
            return (
                "❌ Demo Mode: Agent not available. This is a demo interface.",
                json.dumps({"error": "Agent components not loaded", "demo": True}, indent=2),
                "No tasks executed yet (demo mode)"
            )
        
        if not task.strip():
            return (
                "⚠️ Error: Task description cannot be empty",
                json.dumps({"error": "Empty task"}, indent=2),
                "Please enter a task description"
            )
        
        if not url.strip():
            return (
                "⚠️ Error: URL cannot be empty",
                json.dumps({"error": "Empty URL"}, indent=2),
                "Please enter a starting URL"
            )
        
        try:
            logger.info(f"📝 Executing task: {task}")
            logger.info(f"🌐 URL: {url}")
            logger.info(f"⚙️ Headless: {headless}, Max Steps: {max_steps}")
            
            # Initialize agent
            self.agent = AutonomousBrowserAgent(
                headless=headless,
                browser_type="chromium",
                embedding_dim=512,
                hidden_dim=256,
                num_actions=50
            )
            
            # Execute task
            results = await self.agent.execute_task(
                task=task,
                url=url,
                max_steps=max_steps,
                mode="autonomous"
            )
            
            # Save to history
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "task": task,
                "url": url,
                "success": results.get("success", False),
                "steps_completed": len(results.get("steps", []))
            }
            self.task_history.append(history_entry)
            
            # Keep only recent history
            if len(self.task_history) > self.max_history:
                self.task_history = self.task_history[-self.max_history:]
            
            # Format results
            status = "✅ Success!" if results.get("success") else "⚠️ Partial Success"
            steps_completed = len(results.get("steps", []))
            sub_tasks_completed = sum(
                1 for step in results.get("steps", []) 
                if step.get("success", False)
            )
            
            status_text = f"""
{status}

📋 **Task:** {task}
🌐 **URL:** {url}
📊 **Steps Completed:** {steps_completed}/{max_steps}
✅ **Successful Steps:** {sub_tasks_completed}
⏱️ **Timestamp:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

**Sub-tasks:** {len(results.get("sub_tasks", []))}
{chr(10).join(f'• {st}' for st in results.get("sub_tasks", [])[:5])}
"""
            
            # Format results as JSON
            results_json = json.dumps(results, indent=2, default=str)
            
            # Format history
            history_text = self._format_history()
            
            # Close agent
            await self.agent.close()
            
            logger.info(f"✅ Task completed successfully")
            
            return status_text, results_json, history_text
            
        except Exception as e:
            logger.error(f"❌ Task execution failed: {str(e)}")
            if self.agent:
                try:
                    await self.agent.close()
                except:
                    pass
            
            return (
                f"❌ Error: {str(e)}",
                json.dumps({"error": str(e), "type": type(e).__name__}, indent=2),
                self._format_history()
            )
    
    def decompose_task(self, task: str) -> str:
        """
        Show task decomposition.
        
        Args:
            task: Task description
            
        Returns:
            Formatted sub-tasks
        """
        if not AGENT_AVAILABLE:
            return "Agent not available (demo mode)"
        
        if not task.strip():
            return "Please enter a task description"
        
        try:
            planner = PlannerAgent()
            sub_tasks = planner.decompose_task(task)
            
            result = "📝 **Task Decomposition**\n\n"
            result += f"**Original Task:** {task}\n\n"
            result += f"**Sub-tasks:** ({len(sub_tasks)} steps)\n\n"
            
            for i, sub_task in enumerate(sub_tasks, 1):
                result += f"{i}. {sub_task}\n"
            
            return result
        except Exception as e:
            logger.error(f"Decomposition failed: {str(e)}")
            return f"Error: {str(e)}"
    
    def _format_history(self) -> str:
        """Format task history for display."""
        if not self.task_history:
            return "📜 No tasks executed yet"
        
        history_text = "📜 **Recent Tasks**\n\n"
        for i, task in enumerate(reversed(self.task_history), 1):
            status = "✅" if task["success"] else "⚠️"
            history_text += f"{i}. {status} {task['task']}\n"
            history_text += f"   URL: {task['url']}\n"
            history_text += f"   Steps: {task['steps_completed']}\n"
            history_text += f"   Time: {task['timestamp']}\n\n"
        
        return history_text


def create_interface():
    """Create Gradio interface with theme and styling."""
    interface = BrowserAgentInterface()
    
    with gr.Blocks(
        title="🤖 Autonomous Browser Agent",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("""
        # 🤖 Autonomous Browser Agent with MAYINI Framework
        
        ### Intelligent Web Automation Powered by Deep Learning
        
        This agent combines:
        - **🧠 MAYINI Framework** - Custom deep learning for decision-making
        - **👁️ Vision Transformers** - Visual page understanding
        - **🎭 Playwright** - Cross-browser automation
        - **🔄 Reinforcement Learning** - Continuous improvement
        
        ---
        """)
        
        with gr.Tab("🚀 Execute Task"):
            gr.Markdown("### Execute a web automation task")
            
            with gr.Row():
                with gr.Column(scale=3):
                    task_input = gr.Textbox(
                        label="📝 Task Description",
                        placeholder="Example: Search for flights from NYC to London on Dec 20",
                        lines=3,
                        info="Describe what you want the agent to do"
                    )
                    
                    url_input = gr.Textbox(
                        label="🌐 Starting URL",
                        placeholder="https://www.google.com/flights",
                        value="https://www.google.com",
                        info="URL where the agent will start"
                    )
                    
                    with gr.Row():
                        headless_checkbox = gr.Checkbox(
                            label="🎭 Run Headless",
                            value=True,
                            info="Run browser in background (no visible window)"
                        )
                        max_steps_slider = gr.Slider(
                            minimum=5,
                            maximum=100,
                            value=30,
                            step=5,
                            label="⏱️ Max Steps",
                            info="Maximum number of actions to attempt"
                        )
                    
                    execute_btn = gr.Button(
                        "▶️ Execute Task",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=1):
                    status_output = gr.Textbox(
                        label="📊 Status",
                        lines=12,
                        interactive=False,
                        show_label=True
                    )
            
            with gr.Row():
                results_output = gr.Textbox(
                    label="📄 Detailed Results (JSON)",
                    lines=15,
                    interactive=False,
                    max_lines=20
                )
                history_output = gr.Textbox(
                    label="📜 Task History",
                    lines=15,
                    interactive=False
                )
            
            execute_btn.click(
                fn=interface.execute_task_sync,
                inputs=[task_input, url_input, headless_checkbox, max_steps_slider],
                outputs=[status_output, results_output, history_output]
            )
        
        with gr.Tab("🔍 Task Planner"):
            gr.Markdown("### Visualize how your task will be decomposed")
            
            with gr.Row():
                planner_task_input = gr.Textbox(
                    label="📝 Task",
                    placeholder="Example: Buy a laptop on Amazon",
                    lines=2
                )
                decompose_btn = gr.Button("🔨 Decompose", variant="secondary")
            
            decomposition_output = gr.Textbox(
                label="📋 Sub-Tasks",
                lines=12,
                interactive=False
            )
            
            decompose_btn.click(
                fn=interface.decompose_task,
                inputs=[planner_task_input],
                outputs=[decomposition_output]
            )
        
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About This Project
            
            ### 🏗️ Architecture
            
            This autonomous browser agent combines cutting-edge technologies:
            
            1. **MAYINI Framework**: Custom deep learning library with neural networks
            2. **Vision Transformers**: Visual page understanding without HTML dependency
            3. **Playwright**: Cross-browser automation with auto-waiting
            4. **Reinforcement Learning**: Policy gradient methods for improvement
            
            ### 🎯 Key Features
            
            - **Hierarchical Planning**: Breaks complex tasks into sub-goals
            - **Visual Understanding**: Screenshot-based page comprehension
            - **Memory-Augmented**: LSTM networks remember past interactions
            - **Multi-Task Learning**: Trained on diverse web tasks
            - **Exploration**: Curiosity-driven discovery of new actions
            
            ### 📚 Use Cases
            
            - Form filling and submission
            - Web scraping and data extraction
            - E-commerce automation
            - Navigation and search
            - Testing and QA
            
            ### 🔗 Links
            
            - [GitHub](https://github.com/yourusername/autonomous-browser-agent)
            - [MAYINI Framework](https://pypi.org/project/mayini-framework/)
            - [Playwright](https://playwright.dev/)
            - [Documentation](https://docs.example.com)
            
            ### 📄 License
            
            MIT License - Free to use and modify!
            """)
        
        gr.Markdown("""
        ---
        <div style="text-align: center;">
            <p>Built with ❤️ using MAYINI, Playwright, and Vision Transformers</p>
            <p>© 2024 | Autonomous Browser Agent Project</p>
        </div>
        """)
    
    return demo


# Main entry point
if __name__ == "__main__":
    logger.info("🚀 Starting Autonomous Browser Agent Web Interface...")
    logger.info(f"🧠 Agent Available: {AGENT_AVAILABLE}")
    
    demo = create_interface()
    
    # Launch with Hugging Face Spaces configuration
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
