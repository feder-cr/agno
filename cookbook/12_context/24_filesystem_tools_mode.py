"""
Filesystem Context Provider — Tools Mode
=========================================

With mode=tools, the provider exposes FileTools (list_files, search_files,
read_file) directly instead of wrapping them in a query_<id> sub-agent tool.
The agent orchestrates the tools itself.

Requires: OPENAI_API_KEY
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from agno.agent import Agent
from agno.context import ContextMode
from agno.context.fs import FilesystemContextProvider
from agno.models.openai import OpenAIResponses

# ---------------------------------------------------------------------------
# Create a demo directory
# ---------------------------------------------------------------------------
DEMO_PATH = Path(__file__).resolve().parent / "demo-fs-tools"
if DEMO_PATH.exists():
    shutil.rmtree(DEMO_PATH)
DEMO_PATH.mkdir()
(DEMO_PATH / "README.md").write_text("# Demo Project\n\nA sample project for testing.\n")
(DEMO_PATH / "config.json").write_text('{"debug": true, "port": 8080}\n')
(DEMO_PATH / "src").mkdir()
(DEMO_PATH / "src" / "main.py").write_text("print('Hello, world!')\n")

# ---------------------------------------------------------------------------
# Create the provider with mode=tools
# ---------------------------------------------------------------------------
fs = FilesystemContextProvider(
    id="project",
    root=DEMO_PATH,
    mode=ContextMode.tools,
)

# ---------------------------------------------------------------------------
# Create the Agent
# ---------------------------------------------------------------------------
agent = Agent(
    model=OpenAIResponses(id="gpt-5.4"),
    tools=fs.get_tools(),
    instructions=fs.instructions(),
    markdown=True,
)

# ---------------------------------------------------------------------------
# Run the Agent
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\nfs.status() =", fs.status())
    print("fs.mode =", fs.mode)
    tools = fs.get_tools()
    print("fs.get_tools() =", [t.name if hasattr(t, "name") else type(t).__name__ for t in tools])
    print()

    prompt = "List all files in the project and show me the contents of config.json"
    print(">", prompt)
    print()
    asyncio.run(agent.aprint_response(prompt, stream=True))
