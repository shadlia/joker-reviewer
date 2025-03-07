from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import datetime

load_dotenv()


def get_timestamp():
    """
    Returns current timestamp in ISO 8601 format.
    """
    return datetime.datetime.now().isoformat()


def execute_toolcall(response):
    if response.tool_calls:
        selected_tool = response.tool_calls[0]
        tool_args = selected_tool.get("args")

        return eval(f"""{selected_tool["name"]}(**{tool_args})""")


def search_web_tool(query):
    """
    browse the web for the answer to the user's question.
    """
    return "liverpool against psg"


# duckduckgo library for seaching the web : install the right package

model = ChatOpenAI(model="gpt-4o-mini").bind_tools([get_timestamp, search_web_tool])
prompt = ChatPromptTemplate.from_template(
    """
answer the user question: {question}
"""
)
chain = prompt | model
response = chain.invoke({"question": "who won between liverpool and psg ?"})
print(response)
print(execute_toolcall(response))
