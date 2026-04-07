from collections import deque
from typing import Deque, List, TypedDict

from langgraph import StateGraph

from agents import DemandForecaster, HybridController, SolarForecaster


class Observation(TypedDict):
    solar: float
    demand: float
    battery: float
    price: float
    carbon: float
    hour: float


class State(TypedDict, total=False):
    observation: Observation
    history: List[Observation]
    solar_forecast: float
    demand_forecast: float
    carbon_index: str
    action: int
    rationale: str


solar_agent = SolarForecaster(window=6)
demand_agent_impl = DemandForecaster(window=12)
controller = HybridController()


def _history(state: State) -> Deque[Observation]:
    return deque(state.get("history", []), maxlen=48)


def solar_agent_node(state: State) -> State:
    obs = state["observation"]
    state["solar_forecast"] = solar_agent(_history(state), obs["hour"])
    return state


def demand_agent_node(state: State) -> State:
    obs = state["observation"]
    state["demand_forecast"] = demand_agent_impl(_history(state), obs["hour"])
    return state


def carbon_node(state: State) -> State:
    carbon = state["observation"]["carbon"]
    if carbon < 80:
        index = "very low"
    elif carbon < 120:
        index = "low"
    elif carbon < 200:
        index = "moderate"
    else:
        index = "high"
    state["carbon_index"] = index
    return state


def control_agent(state: State) -> State:
    obs = state["observation"]
    action, rationale = controller.decide(
        obs,
        state.get("solar_forecast", obs["solar"]),
        state.get("demand_forecast", obs["demand"]),
        state.get("carbon_index", "moderate"),
    )
    state["action"] = action
    state["rationale"] = rationale
    return state


graph = StateGraph(State)
graph.add_node("solar", solar_agent_node)
graph.add_node("demand", demand_agent_node)
graph.add_node("carbon", carbon_node)
graph.add_node("control", control_agent)

graph.add_edge("solar", "demand")
graph.add_edge("demand", "carbon")
graph.add_edge("carbon", "control")

graph.set_entry_point("solar")

compiled_graph = graph.compile()
