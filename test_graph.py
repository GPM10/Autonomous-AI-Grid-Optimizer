from graph.langgraph_flow import compiled_graph

initial_state = {"solar": 5.0, "demand": 3.0, "battery": 5.0, "price": 0.1, "carbon": 200}
result = compiled_graph.invoke(initial_state)
print("LangGraph result:", result)