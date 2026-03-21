from langgraph import StateGraph
from typing import TypedDict

class State(TypedDict):
    solar_forecast: float
    demand_forecast: float
    battery: float
    price: float
    carbon: float
    action: int

def forecast_agent(state):
    # Simple forecast: use current solar
    state['solar_forecast'] = state.get('solar', 0)
    return state

def demand_agent(state):
    # Simple: use current demand
    state['demand_forecast'] = state.get('demand', 0)
    return state

def control_agent(state):
    # For now, simple rule
    net = state['solar_forecast'] - state['demand_forecast']
    if net > 0:
        state['action'] = 1  # charge
    else:
        state['action'] = 2  # discharge
    return state

def carbon_agent(state):
    # Compute emissions based on action
    # Placeholder
    state['emissions'] = state.get('carbon', 0) * 1  # assume import
    return state

# Build graph
graph = StateGraph(State)
graph.add_node("forecast", forecast_agent)
graph.add_node("demand", demand_agent)
graph.add_node("control", control_agent)
graph.add_node("carbon", carbon_agent)

graph.add_edge("forecast", "demand")
graph.add_edge("demand", "control")
graph.add_edge("control", "carbon")

graph.set_entry_point("forecast")

compiled_graph = graph.compile()

# Example run
initial_state = {"solar": 5.0, "demand": 3.0, "battery": 5.0, "price": 0.1, "carbon": 200}
result = compiled_graph.invoke(initial_state)
print(result)