# import psycopg
# from langgraph.checkpoint.postgres import PostgresSaver
# from langgraph.graph import END, START, StateGraph

# from agents import (
# budget_agent,
# final_response_agent,
# flight_agent,
# hotel_agent,
# human_approval_agent,
# itinerary_agent,
# supervisor_agent,
# weather_agent,
# )
# from config import DATABASE_URL
# from state import TravelState

# AGENT_ORDER = [
#     "flight_agent",
#     "hotel_agent",
#     "weather_agent",
#     "budget_agent",
#     "itinerary_agent",
# ]

# ROUTE_MAP = {
#     "flight_agent": "flight_agent",
#     "hotel_agent": "hotel_agent",
#     "weather_agent": "weather_agent",
#     "budget_agent": "budget_agent", 
#     "itinerary_agent": "itinerary_agent",
# }


# def _selected_agents(state: TravelState) -> list[str]:
#     selected = state.get("selected_agents")
#     return [agent for agent in AGENT_ORDER if agent in selected]  


# def route_from_supervisor(state: TravelState) -> str:
#     # Guardrail blocked the request â†’ skip all agents, go straight to the end
#     if not state.get("selected_agents"):
#         return "blocked"
#     selected = _selected_agents(state)
#     return selected[0] if selected else "itinerary_agent"



# def route_after_agent(current_agent: str):
#     def route(state: TravelState) -> str:
#         selected = _selected_agents(state)
#         current_index = AGENT_ORDER.index(current_agent)

#         for next_agent in AGENT_ORDER[current_index + 1:]:
#             if next_agent in selected:
#                 return next_agent

#         return "itinerary_agent"

#     return route




# def build_graph():
#     graph = StateGraph(TravelState)

#     graph.add_node("supervisor", supervisor_agent)
#     graph.add_node("flight_agent", flight_agent)
#     graph.add_node("hotel_agent", hotel_agent)
#     graph.add_node("weather_agent", weather_agent)
#     graph.add_node("budget_agent", budget_agent)
#     graph.add_node("itinerary_agent", itinerary_agent)
#     graph.add_node("human_approval", human_approval_agent)
#     graph.add_node("final_response", final_response_agent)

#     graph.add_edge(START, "supervisor")
#     graph.add_conditional_edges("supervisor",route_from_supervisor, {**ROUTE_MAP, "blocked": "final_response"},)
#     graph.add_conditional_edges("flight_agent", route_after_agent("flight_agent"), ROUTE_MAP)
#     graph.add_conditional_edges("hotel_agent", route_after_agent("hotel_agent"), ROUTE_MAP)
#     graph.add_conditional_edges("weather_agent", route_after_agent("weather_agent"), ROUTE_MAP)
#     graph.add_conditional_edges("budget_agent", route_after_agent("budget_agent"), ROUTE_MAP)
#     graph.add_edge("itinerary_agent", "human_approval")
#     graph.add_edge("human_approval", "final_response")
#     graph.add_edge("final_response", END)

#     if DATABASE_URL:
#         conn = psycopg.connect(DATABASE_URL, auto_commit=True)
#         checkpointer = PostgresSaver(conn)
#         checkpointer.setup()
#         return graph.compile(checkpointer=checkpointer)

#     return graph.compile()


# app = build_graph()



#========================================#Fan-in is automatic â€” when all three data agents#
# write to budget_agent in the same super-step, LangGraph schedules it once, after all have finished.


import psycopg
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph

from agents import (
    budget_agent,
    final_response_agent,
    flight_agent,
    hotel_agent,
    human_approval_agent,
    itinerary_agent,
    supervisor_agent,
    weather_agent,
)
from config import DATABASE_URL
from state import TravelState

# agents that are independent of each other -> run concurrently
DATA_AGENTS = ["flight_agent", "hotel_agent", "weather_agent"]

ROUTE_MAP = {
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}


def route_from_supervisor(state: TravelState):
    """Return a LIST of data agents -> LangGraph dispatches them in parallel."""
    selected = state.get("selected_agents") or []

    # guardrail blocked the request
    if not selected:
        return "blocked"

    parallel = [a for a in DATA_AGENTS if a in selected]
    if parallel:
        return parallel                     # <- fan-out happens here

    # no data agents needed at all
    if "budget_agent" in selected:
        return "budget_agent"
    return "itinerary_agent"


def route_after_data(state: TravelState) -> str:
    """Every data agent converges here -> fan-in."""
    selected = state.get("selected_agents") or []
    return "budget_agent" if "budget_agent" in selected else "itinerary_agent"


def build_graph():
    graph = StateGraph(TravelState)

    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("flight_agent", flight_agent)
    graph.add_node("hotel_agent", hotel_agent)
    graph.add_node("weather_agent", weather_agent)
    graph.add_node("budget_agent", budget_agent)
    graph.add_node("itinerary_agent", itinerary_agent)
    graph.add_node("human_approval", human_approval_agent)
    graph.add_node("final_response", final_response_agent)

    graph.add_edge(START, "supervisor")

    # FAN-OUT: path function may return a list -> those nodes run in one super-step
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {**ROUTE_MAP, "blocked": "final_response"},
    )

    # FAN-IN: all data agents converge; LangGraph waits for all of them
    for agent in DATA_AGENTS:
        graph.add_conditional_edges(agent, route_after_data, ROUTE_MAP)

    # sequential tail (real data dependencies)
    graph.add_edge("budget_agent", "itinerary_agent")
    graph.add_edge("itinerary_agent", "human_approval")
    graph.add_edge("human_approval", "final_response")
    graph.add_edge("final_response", END)

    if DATABASE_URL:
        pool = ConnectionPool(
            conninfo=DATABASE_URL,
            max_size=10,
            kwargs={
                "autocommit": True,
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            },
        )
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


app = build_graph()
