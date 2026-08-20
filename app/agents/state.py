import operator
from typing import Annotated, TypedDict


class agentState(TypedDict):
    messages: Annotated[list[dict], operator.add]
    current_query: str
    documents: list[str]
    plan: list[str]
    status: str
    final_answer: str


"""
class agentState:
    messages: []
    current_query = "what is rag ?"
    documents = []
    plan = []
    status = "starting"
    final_answer = ""
"""
