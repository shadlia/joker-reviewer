"""Define a simple chatbot agent.

This agent returns a predefined response without using an actual LLM.
"""

from typing import Any, Dict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from agent.configuration import Configuration
from agent.state import State
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field
import os

load_dotenv()


class Review(BaseModel):
    funny_or_not: bool = Field(description="Whether the joke is funny or not")
    reason: str = Field(description="The reason why the joke is funny or not")


def should_continue(state: State):
    if state["funny_or_not"]:
        return END
    return "joker_agent"


async def joker_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    configuration = Configuration.from_runnable_config(config)
    prompt = ChatPromptTemplate.from_template(
        """
       - You are a joker. You are in a room with a reviewer. You are trying to make the reviewer laugh.
       - Try to give dull and boring jokes sometimes .
       - Make sure its just a single joke.
       - This is the context of the joke : {context}"""
    )
    model = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | model
    response = chain.invoke({"context": state["context"]})
    return {"joke": response.content, "reviews_num": state["reviews_num"] + 1}


async def reviewer_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    configuration = Configuration.from_runnable_config(config)

    prompt = ChatPromptTemplate.from_template(
        """
         - You are a reviewer. You are in a room with a joker. You are trying to not laugh at the joker's joke.
         - This is the context of the joke : {context} and this is the given joke : {joke}
         - You have 10% chance of laughing at the joke and 90% chance of not laughing at the joke.
         - if the joke is funny return True otherwise return False.
         - Provide a reason for your answer and why you think the joke is funny or not.
     
         
    """
    )
    model = ChatOpenAI(model="gpt-4o-mini").with_structured_output(Review)
    chain = prompt | model
    response = chain.invoke({"context": state["context"], "joke": state["joke"]})
    return {"funny_or_not": response.funny_or_not, "reason": response.reason}


# Define a new graph
workflow = StateGraph(State, config_schema=Configuration)

# Add the node to the graph
workflow.add_node("joker_agent", joker_agent)


# Set the entrypoint as `call_model`
workflow.set_entry_point("joker_agent")
workflow.add_node("reviewer_agent", reviewer_agent)

workflow.add_edge("joker_agent", "reviewer_agent")
workflow.add_conditional_edges("reviewer_agent", should_continue)
workflow.add_edge("reviewer_agent", END)

# Compile the workflow into an executable graph
graph = workflow.compile()
graph.name = "graph "
