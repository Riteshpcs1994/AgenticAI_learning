# Objectives :
# 1. Define state structure with a list of humanMessage Objects
# 2. Initialize a GPT-4o model using Langchain's ChatOpenAI
# 3. Sending and handling different types of message.
# 4. Building and compiling the graph of the Agent

# Main-Goal: How to integrate LLM in our state in our Graph.

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Union

from pathlib import Path



load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    
llm = ChatOpenAI(model="gpt-4o", temperature=0)


def process_messages(state: AgentState) -> AgentState:
    """Process the list of messages using the LLM and append the response."""
    #print(state["messages"])
    response = llm.invoke(state["messages"])
    print(f"\nAI Response: {response.content}\n")
    state["messages"].append(AIMessage(content=response.content))
    return state

graph = StateGraph(AgentState)
graph.add_node("process_messages", process_messages)
graph.add_edge(START, "process_messages")
graph.add_edge("process_messages", END) 

agent=graph.compile()

convesation_history = []

user_input = input("Enter your message: ")

agent.invoke({"messages": [HumanMessage(content=user_input)]})

while user_input != "exit":
    convesation_history.append(HumanMessage(content=user_input))
    state = {"messages": convesation_history}
    state = agent.invoke(state)
    convesation_history = state["messages"]
    user_input = input("Enter your message: ")

data_path = Path.cwd() / "data_files"
file_path = data_path / "conversation_history.txt"

file_path.touch(exist_ok=True)

with open(file_path, "w") as f:
    for message in convesation_history:
        if isinstance(message, HumanMessage):
            f.write(f"Human: {message.content}\n")        
        elif isinstance(message, AIMessage):  
            f.write(f"AI: {message.content}\n\n") 

    f.write("\n--- End of Conversation ---\n")
    f.close()

print(f"Conversation history saved to {file_path}.txt")



