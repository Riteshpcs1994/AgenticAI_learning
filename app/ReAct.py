from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage # The foundation class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Message type for tool interactions
from langchain_core.messages import SystemMessage # Message type for system-level instructions
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain.tools import tool

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages()]  # Using Annotated to specify message handling

@tool 
def add(a: int, b :int) -> int:
    """Example tool that adds two integers."""
    return a + b

tools = [add]

model = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    """Invoke the LLM with the current messages and append the response."""
    system_prompt = SystemMessage(content="You are my AI assistant, please answer the query to the best of your ability.")
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState) -> bool:
    """Decide whether to continue the conversation based on the last message."""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "end"  # Continue if the last message was a tool interaction
    else :
        return "continue"  # Stop otherwise

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node =ToolNode(tools=tools)
graph.add_node("tools", tool_node)
graph.add_edge(START, "our_agent")

graph.add_conditional_edges("our_agent", should_continue, {
    "continue": tool_node,
    "end": END
}   )

graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    """Utility function to print streaming responses."""
    for s in stream:
        message = s["message"][-1]
        if isinstance(message, tuple):
            print(message.content)

        else :
            message.pretty_print()


input_query = {"messages": [("What is 2 + 2?")]}
print_stream(app.stream(input_query, stream_mode="values"))