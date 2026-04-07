from graph.langgraph_flow import compiled_graph

state = {
    "observation": {
        "solar": 6.0,
        "demand": 4.5,
        "battery": 5.0,
        "price": 0.12,
        "carbon": 150.0,
        "hour": 10.0,
    },
    "history": [
        {"solar": 5.5, "demand": 4.0, "battery": 5.0, "price": 0.11, "carbon": 140.0, "hour": 9.0},
        {"solar": 4.0, "demand": 4.2, "battery": 5.1, "price": 0.10, "carbon": 130.0, "hour": 8.0},
    ],
}
result = compiled_graph.invoke(state)
print("LangGraph result:", result)
assert "action" in result and "rationale" in result
