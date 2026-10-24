from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import requests
import asyncio

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b")

# ---------------------------------------------------Tools--------------------------------------------------


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations : add , sub , mul , div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Divison by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


# make tool list
tools = [calculator]

# make the llm tool aware
llm_with_tools = llm.bind_tools(tools)


# ---------------------------STATE------------------------------------------------------------------------


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph():
    async def chat_node(state: ChatState):
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    chatbot = graph.compile()
    return chatbot


async def main():
    chatbot = build_graph()

    result = await chatbot.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="Find the modulus of 132354 and give answer like a cricket commentator."
                )
            ]
        }
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())

# tool_node = ToolNode(
#     tools, handle_tool_errors="Search failed. Please try again later."
# )  # executes tool calls


# def chat_node(state: ChatState):
#     messages = state["messages"]
#     response = llm_with_tools.invoke(messages)
#     return {"messages": [response]}


# connectionObject = sqlite3.connect(database="chatbot.db", check_same_thread=False)

# checkpointer = SqliteSaver(conn=connectionObject)

# graph = StateGraph(ChatState)

# graph.add_node("chat_node", chat_node)
# graph.add_node("tools", tool_node)

# graph.add_edge(START, "chat_node")
# graph.add_conditional_edges("chat_node", tools_condition)

# graph.add_edge("tools", "chat_node")

# chatbot = graph.compile(checkpointer=checkpointer)


# def retrieve_all_threads():
#     all_threads = set()
#     for checkpoint in checkpointer.list(None):
#         all_threads.add(checkpoint.config["configurable"]["thread_id"])

#     return list(all_threads)
