"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Optional, TypedDict


@dataclass
class State(TypedDict):
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    for more information.
    """

    context: str
    joke: str
    funny_or_not: bool
    reason: str
    reviews_num: Optional[int] = 0
