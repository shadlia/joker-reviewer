from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import datetime
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()


def execute_toolcall(response):
    if response.tool_calls:
        selected_tool = response.tool_calls[0]
        tool_args = selected_tool.get("args")

        return eval(f"""{selected_tool["name"]}(**{tool_args})""")


def get_timestamp():
    """
    Returns current timestamp in ISO 8601 format.
    """
    return datetime.datetime.now().isoformat()


def search_web_tool(query):
    """
    browse the web for the answer to the user's question.
    """
    search = DuckDuckGoSearchRun()
    response = search.invoke(query)
    return response


# duckduckgo library for seaching the web : install the right package

model = ChatOpenAI(model="gpt-4o-mini").bind_tools([get_timestamp, search_web_tool])
prompt = ChatPromptTemplate.from_template(
    """
answer the user question: {question}
"""
)
chain = prompt | model
response = chain.invoke({"question": "who is the best singer in the world ?"})
print(execute_toolcall(response))
