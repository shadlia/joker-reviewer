from langgraph.graph import StateGraph, END
from typing import TypedDict
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


class State(TypedDict):
    context: str
    joke: str
    funny_or_not: bool
    reason: str


class Review(BaseModel):
    funny_or_not: bool = Field(description="Whether the joke is funny or not")
    reason: str = Field(description="The reason why the joke is funny or not")


# Agent 1 : the joker
def joker_agent(state: State):
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
    print(response.content)
    return {"joke": response.content}


# Agent 2 : the reviewer
def reviewer_agent(state: State):
    prompt = ChatPromptTemplate.from_template(
        """
         - You are a reviewer. You are in a room with a joker. You are trying to not laugh at the joker's joke.
         - This is the context of the joke : {context} and this is the given joke : {joke}
         - Be strict and hard to laugh at the joker's joke.
         - if the joke is funny return True otherwise return False.
         - Provide a reason for your answer and why you think the joke is funny or not.
     
         
    """
    )
    model = ChatOpenAI(model="gpt-4o-mini").with_structured_output(Review)
    chain = prompt | model
    response = chain.invoke({"context": state["context"], "joke": state["joke"]})
    return {"funny_or_not": response.funny_or_not, "reason": response.reason}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("joker_agent", joker_agent)
    graph.set_entry_point("joker_agent")
    graph.add_node("reviewer_agent", reviewer_agent)
    graph.add_edge("joker_agent", "reviewer_agent")
    graph.add_edge("reviewer_agent", END)
    return graph.compile()


if __name__ == "__main__":
    agent = build_graph()
    reponse = agent.invoke({"context": "movies"})
    print(reponse)
