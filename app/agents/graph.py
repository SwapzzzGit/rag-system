from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.agents.state import agentState
from app.agents.nodes.planner import planner_node
from app.agents.nodes.responder import generate_node
from app.agents.nodes.retriver import retrieve_node

# Initializing the state graph
workflow = StateGraph(agentState)

# defining the nodes
workflow.add_node("planner", planner_node)
workflow.add_node("retriver", retrieve_node)
workflow.add_node("responder", generate_node)


# defining the edges and routing logic
def route_planner(state: agentState):
    """
    Routes the workflow based on the decision
    """
    if state["current_query"] == "CONVERSATIONAL":
        return "responder"
    return "retriver"


# The workflow will always start from planner.py
workflow.set_entry_point("planner")

# adding condition where | Conditional edge : Planner -> Routes -> Decides (Responder or Retriver)
workflow.add_conditional_edges(
    "planner", route_planner, {"retriver": "retriver", "responder": "responder"}
)

workflow.add_edge("retriver", "responder")
workflow.add_edge("responder", END)

# ---- MEMORY ----
# MemorySaver helps the agent to remember convo based on thread_id (convo history)

checkpointer = MemorySaver()

# compile Graph with memory

rag_agent = workflow.compile(checkpointer=checkpointer)
