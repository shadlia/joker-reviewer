from langgraph.graph import StateGraph, END
from typing import TypedDict
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


class State(TypedDict):
    context: str
    joke: str


# Agent 1 : the joker
def joker_agent(state: State):
    prompt = ChatPromptTemplate.from_template(
        """
       - You are a joker. You are in a room with a reviewer. You are trying to make the reviewer laugh.
       - Make sure its just a single joke.
       - This is the context of the joke : {context}"""
    )
    model = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | model
    response = chain.invoke({"context": state["context"]})
    print(response.content)
    return {"joke": response.content}


# Agent 2 : the reviewer
def reviewer_agent():
    return


def build_graph():
    graph = StateGraph(State)
    graph.add_node("joker_agent", joker_agent)
    graph.set_entry_point("joker_agent")
    graph.add_edge("joker_agent", END)
    return graph.compile()


if __name__ == "__main__":
    agent = build_graph()
    reponse = agent.invoke({"context": "give me a joke about Coding"})
    print(reponse)
