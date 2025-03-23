from typing import List
from pydantic import BaseModel
from langchain.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import BaseMessage
from mock_actions import mock_restart_server, mock_get_server_status


# Mock action suggestions (can be expanded dynamically)
actions = [
    {"name": "Restart Service"},
    {"name": "Generate Incident Report"},
    {"name": "Check System Logs"},
    {"name": "Send Email"}
]

# ✅ Define State Schema
class AgentState(BaseModel):
    messages: List[BaseMessage]
    next: str  # Stores the next action to perform

# ✅ Define tool input schemas
class RestartServerInput(BaseModel):
    server_name: str

class GetServerStatusInput(BaseModel):
    server_name: str

# ✅ Define structured tools
restart_server_tool = StructuredTool.from_function(
    func=mock_restart_server,
    name="Restart Server",
    description="Use this tool to restart a given server by providing its name.",
    args_schema=RestartServerInput
)

get_server_status_tool = StructuredTool.from_function(
    func=mock_get_server_status,
    name="Get Server Status",
    description="Use this tool to get the status of a given server.",
    args_schema=GetServerStatusInput
)

# ✅ Define the system prompt
system_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that can manage servers."),
    ("human", "{input}")
])

# ✅ Function to determine next action
def decide_action(state: AgentState) -> AgentState:
    latest_message = state.messages[-1].content.lower()
    for action in actions:
        if action["name"] == "Restart Service":
            return AgentState(messages=state.messages, next="restart_server")
        
        elif action["name"] == "Generate Incident Report":
            return AgentState(messages=state.messages, next="get_server_status")
        elif action["name"] == "Check System Logs":
            return AgentState(messages=state.messages, next="get_server_status")#TODO -
        elif action["name"] == "Send Email":
            return AgentState(messages=state.messages, next="get_server_status")#TODO -
        else:
            return AgentState(messages=state.messages, next="respond")  # ✅ Default behavior when no task is found

# ✅ Function to handle normal responses
def respond(state: AgentState):
    ai_response = AIMessage(content="No Actions executed! You can ask me to restart a server or check its status.")
    return AgentState(messages=state.messages + [ai_response],next="decide")

# ✅ Function to restart server
def restart_server(state: AgentState):
    response = "Server is restarting..."  
    return AgentState(messages=state.messages + [AIMessage(content=response)], next="decide")

# ✅ Function to get server status
def get_server_status(state: AgentState):
    response = "Fetching server status..."  
    return AgentState(messages=state.messages + [AIMessage(content=response)], next="decide")


# ✅ Create the LangGraph workflow
workflow = StateGraph(AgentState)

# ✅ Define nodes
workflow.add_node("decide", RunnableLambda(decide_action))
workflow.add_node("respond", RunnableLambda(respond))
workflow.add_node("restart_server", RunnableLambda(restart_server))
workflow.add_node("get_server_status", RunnableLambda(get_server_status))

# ✅ Define transitions
workflow.set_entry_point("decide")
workflow.add_conditional_edges("decide", lambda state: state.next)

# ✅ Compile the graph
graph = workflow.compile()

# ✅ Function to invoke the agent
def run_agent(user_input: str):
    initial_state = AgentState(messages=[HumanMessage(content=user_input)], next="decide")
    response = graph.invoke(initial_state)

    # Convert AddableValuesDict to AgentState before accessing messages
    response_state = AgentState(**response)
    return response_state


# ✅ Running the agent
if __name__ == "__main__":
    user_inputs = [
        "Restart the UAT server",
        "What is the status of the UAT server?",
        "Hello, what can you do?"
    ]
    
    for user_input in user_inputs:
        print(f"\n🟢 User: {user_input}")
        response_state = run_agent(user_input)

        for msg in response_state.messages:
            print(f"🤖 AI: {msg.content}")
