"""
Context Provider Streaming on AgentOS
=====================================

Tests sub-agent event streaming through os.agno.com. When the parent agent
calls a context provider's query tool, the sub-agent's events (tool calls,
content) are streamed back in real-time.

Run locally:
    python cookbook/12_context/25_agentos_streaming.py

Then open os.agno.com and connect to http://localhost:7777

Requires: OPENAI_API_KEY, PostgreSQL running on 5532
"""

from __future__ import annotations

import shutil
from pathlib import Path

from agno.agent import Agent
from agno.context.wiki import FileSystemBackend, WikiContextProvider
from agno.db.postgres import PostgresDb
from agno.models.openai import OpenAIResponses
from agno.os import AgentOS

# ---------------------------------------------------------------------------
# Setup demo wiki
# ---------------------------------------------------------------------------
WIKI_PATH = Path(__file__).resolve().parent / "demo-wiki-os"
if WIKI_PATH.exists():
    shutil.rmtree(WIKI_PATH)
WIKI_PATH.mkdir()
(WIKI_PATH / "README.md").write_text(
    "# Demo Wiki\n\nA tiny wiki for testing sub-agent streaming on AgentOS.\n"
)
(WIKI_PATH / "architecture.md").write_text(
    "# Architecture\n\n"
    "The system uses a three-tier architecture:\n"
    "1. Frontend (React)\n"
    "2. API (FastAPI)\n"
    "3. Database (PostgreSQL)\n"
)
(WIKI_PATH / "deployment.md").write_text(
    "# Deployment\n\n"
    "We deploy to Kubernetes using Helm charts.\n"
    "The CI/CD pipeline runs on GitHub Actions.\n"
)

# ---------------------------------------------------------------------------
# Setup context provider
# ---------------------------------------------------------------------------
wiki = WikiContextProvider(
    id="wiki",
    backend=FileSystemBackend(path=WIKI_PATH),
    model=OpenAIResponses(id="gpt-5.4-mini"),
)

# ---------------------------------------------------------------------------
# Setup database and agent
# ---------------------------------------------------------------------------
db = PostgresDb(
    id="context-streaming-db",
    db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
)

agent = Agent(
    name="Wiki Assistant",
    db=db,
    model=OpenAIResponses(id="gpt-5.4"),
    tools=wiki.get_tools(),
    instructions=[
        wiki.instructions(),
        "You help users explore a wiki. Use the query_wiki tool to find information.",
    ],
    markdown=True,
)

# ---------------------------------------------------------------------------
# Setup AgentOS
# ---------------------------------------------------------------------------
agent_os = AgentOS(
    description="Context provider streaming demo",
    agents=[agent],
)
app = agent_os.get_app()

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\nWiki files:")
    for f in WIKI_PATH.iterdir():
        print(f"  - {f.name}")
    print()
    print("Starting AgentOS on http://localhost:7777")
    print("Connect via os.agno.com and ask: 'What is our system architecture?'")
    print()
    agent_os.serve(app="25_agentos_streaming:app", reload=True)
