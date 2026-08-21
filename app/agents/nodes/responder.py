import logfire
from langchain_groq import ChatGroq

from app.agents.state import agentState
from app.config import Settings

llm = ChatGroq(
    api_key=Settings.GROQ_API_KEY, model=Settings.GROQ_MODEL, temperature=0.1
)


def generate_node(state: agentState):
    """
    Synthesize using both documentation context and conversational history
    """
    # takes current query stored in state.py which we updated in planner.py
    query = state["current_query"]

    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else ""
        history_str += f"role : {role} {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if query == "CONVERSATIONAL":
        logfire.info("Generating Conversational response using memory")
        prompt = f"""
        You are friendly and helpful Enterprise AI Assistent.
        Answer the user's latest message using CONVERSATIONAL HISTORY below.  

        CONVERSATION HISTORY :
        {history_str}

        LATEST MESSAGE:
        {user_msg}
        """
        with logfire.span("💬 LLM Conversational Response"):
            try:
                content = llm.invoke(prompt).content
                logfire.info("✅ Conversational response generated.")
                return {
                    "final_answer": content,
                    "status": "Response generated.",
                    "plan": state["plan"],
                    "messages": [{"role": "assistant", "content": content}],
                }
            except Exception as e:
                logfire.error(f"LLM Conversational Generation failed: {e}")
                raise e
    else:
        logfire.info("Generating Technical RAG response")
        max_context_chars = 25000
        full_context = ""

        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break

        prompt = f"""
        You are a Senior Technical Architect.
        Answer the question using the TECHNICAL CONTEXT provided.

        TECHNICAL CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """
        with logfire.span(" & LLM Synthesis"):
            try:
                content = llm.invoke(prompt).content
                logfire.info(" ¥ Response synthesised via LLM.")
                return {
                    "final_answer": content,
                    "status": "Response generated.",
                    "plan": state["plan"],
                    "messages": [{"role": "assistant", "content": content}],
                }
            except Exception as e:
                logfire.error(f"LLM Generation failed: {e}")
                raise e
