# Objectives :
# 1. Define state structure with a list of humanMessage Objects
# 2. Initialize a GPT-4o model using Langchain's ChatOpenAI
# 3. Sending and handling different types of message.
# 4. Building and compiling the graph of the Agent

# Main-Goal: How to integrate LLM in our state in our Graph.

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from typing import TypedDict, List, Dict

load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]
    
llm = ChatOpenAI(model="gpt-4o", temperature=0)


def process_messages(state: AgentState) -> AgentState:
    """Process the list of messages using the LLM and append the response."""
    response = llm.invoke(state["messages"])
    print(f"\nAI Response: {response.content}\n")
    return state

graph = StateGraph(AgentState)
graph.add_node("process_messages", process_messages)
graph.add_edge(START, "process_messages")
graph.add_edge("process_messages", END) 

agent=graph.compile()

user_input = input("Enter your message: ")

agent.invoke({"messages": [HumanMessage(content=user_input)]})
