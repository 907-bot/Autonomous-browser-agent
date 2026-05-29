from setuptools import setup

setup(
    name="autonomous-browser-agent-backend",
    version="1.0.0",
    description="Minimal HTTP backend for Autonomous Browser Agent",
    python_requires=">=3.9",
    install_requires=[
        "starlette>=0.27.0",
        "uvicorn[standard]>=0.23.0",
    ],
)
