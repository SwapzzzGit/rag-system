import operator
from typing import Annotated, TypedDict


class agentState(TypedDict):
    messages: Annotated[list[dict], operator.add]
    current_query: str
    documents: list[str]
    plan: list[str]
    status: str
    final_answer: str


# 1. ai gateway portal
# 2. ai gateway features :
# check if governance is already available in azure dashboard
# in azure gateway = each team its on limit -> token utilization ( team level limit and user level limit)
# (remove skills feature) and add gems agent feature-> user should be able to create agent on own
# how agent is made in code ?
# push frontend and backend to github
# token data in observability wfrom where we are getting it ?

"""
class agentState:
    messages: []
    current_query = "what is rag ?"
    documents = []
    plan = []
    status = "starting"
    final_answer = ""
"""
