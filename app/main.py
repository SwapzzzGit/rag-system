import logfire
import os
from dotenv import load_dotenv


load_dotenv()
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

# now safe to import app models - because logfire is active

from fastapi import FastAPI, Response
from app.agents.graph import rag_agent

from pydantic import BaseModel
from typing import Optional

# Initialize FASTAPI

app = FastAPI(title="Enterprise Agentic RAG API")


class QueryRequest(BaseModel):
    q: str
    thread_id: Optional[str] = "default_user"


@app.get("/")
def home():
    return {"message": "Enterprise LangGraph RAG API is live."}


@app.get("/graph")
def get_graph_imge():
    """
    Returns image of agent's workflow.
    """
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"Error": f"Could not generate graph image {e}"}


@app.get("/query")
def query(request: QueryRequest):
    """
    Executes Langgraph RAF flow with memory using POST Request.
    """
    q = request.q
    thread_id = request.thread_id

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph...",
    }

    # Configuration for Memory (Thread ID)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Run the graph synchronously to preserve Logfire context variables
        final_output = rag_agent.invoke(initial_state, config=config)

        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", []),
        }
    except Exception as e:
        logfire.error(f"❌ Backend Execution Failed: {e}")
        return {
            "question": q,
            "answer": "I apologize, but I encountered an internal error while processing your request. Please try again later.",
            "thought_process": ["Error encountered during execution."],
            "status": "error",
            "sources": [],
        }
