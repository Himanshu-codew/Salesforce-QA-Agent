"""
Salesforce Qwen Agent — Root Entry Point Launcher
Allows running `python app.py` directly from the root workspace directory.
"""

import os
import sys

# Set current working directory and sys.path to salesforce-qwen-agent
current_dir = os.path.dirname(os.path.abspath(__file__))
agent_dir = os.path.join(current_dir, "salesforce-qwen-agent")

if os.path.exists(agent_dir):
    os.chdir(agent_dir)
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)

if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv(override=True)

    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("APP_PORT", "8000")))

    print(f"\n============================================================")
    print(f"  Salesforce Qwen Agent Server")
    print(f"  Web UI: http://localhost:{port}")
    print(f"============================================================\n")

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=True,
        reload_excludes=["uploads/*", "*.csv", "*.html", "*.xlsx", "*.log", ".pytest_cache/*", "__pycache__/*"],
        log_level="info",
    )
