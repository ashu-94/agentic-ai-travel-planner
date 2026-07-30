import os

from langchain_mcp_adapters.client import MultiServerMCPClient

from config import (
    AVIATION_STACK_API_KEY,
    OPENWEATHER_API_KEY,
    TAVILY_API_KEY,
)
# Create MCP Client

client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
        },

        "aviationstack": {
            "transport": "stdio",
            "command": r"C:\Users\Ashutosh\multi_agent_system_with_guardrails\aviationstack-mcp\.venv\Scripts\python.exe",
            "args": [
                "-m",
                "aviationstack_mcp",
                "mcp",
                "run"
            ],
            "env": {
                "AVIATION_STACK_API_KEY": AVIATION_STACK_API_KEY
            }
        }   ,
        "weather": {
            "transport": "stdio",
            "command": r"C:\Users\Ashutosh\multi_agent_system_with_guardrails\langgraph_env3\Scripts\python.exe",
            "args": [
                r"C:\Users\Ashutosh\multi_agent_system_with_guardrails\weather_mcp_server.py"
            ],
            "env": {
                "OPENWEATHER_API_KEY": OPENWEATHER_API_KEY
            }
        }

 
    }
)


# Cache tools so we don't load them repeatedly
_tools_cache = None




import asyncio
import random
import threading

# ------------------------
# Reliability layer
# ------------------------

TOOL_TIMEOUT = 45          # seconds per tool call
TOOL_RETRIES = 3

_RETRYABLE = (
    "timeout", "timed out", "connect", "connection", "read error",
    "reset", "temporarily", "overloaded", "502", "503", "504",
)


def _error_signature(e) -> str:
    """Flatten an exception (and any ExceptionGroup children) into one string."""
    parts = [f"{type(e).__name__} {e}"]
    for sub in getattr(e, "exceptions", None) or []:
        parts.append(f"{type(sub).__name__} {sub}")
    return " ".join(parts).lower()


# Cache tools so we don't load them repeatedly.
# threading.Lock (not asyncio.Lock) because agents are sync functions calling
# asyncio.run(), so each runs in its own thread with its own event loop.
_tools_cache = None
_tools_lock = threading.Lock()


async def get_tools():
    global _tools_cache

    if _tools_cache is not None:
        return _tools_cache

    with _tools_lock:
        if _tools_cache is not None:        # another thread won the race
            return _tools_cache
        try:
            _tools_cache = await client.get_tools()
        except Exception as e:
            print("\n========== MCP TOOL LOAD FAILED ==========")
            print(type(e))
            print(repr(e))
            for i, sub in enumerate(getattr(e, "exceptions", None) or []):
                print(f"--- sub {i + 1} --- {type(sub)} {sub!r}")
            print("==========================================\n")
            raise

    return _tools_cache


async def call_tool(tool_name: str, args: dict = None,
                    retries: int = TOOL_RETRIES, timeout: int = TOOL_TIMEOUT):
    """Invoke an MCP tool with a hard timeout and bounded retries.

    Flight and hotel hit Tavily concurrently in the same LangGraph super-step,
    so a single cold-handshake timeout would otherwise return empty results
    for both.
    """
    tools = await get_tools()

    tool = next((t for t in tools if t.name == tool_name), None)
    if tool is None:
        raise ValueError(f"Tool '{tool_name}' not found")

    last_error = None

    for attempt in range(retries):
        try:
            return await asyncio.wait_for(tool.ainvoke(args or {}), timeout=timeout)

        except Exception as e:
            last_error = e
            signature = _error_signature(e)
            retryable = (
                isinstance(e, asyncio.TimeoutError)
                or any(s in signature for s in _RETRYABLE)
            )

            if not retryable:
                raise                        # a real bug — fail loudly

            if attempt == retries - 1:
                break

            wait = (2 ** attempt) + random.uniform(0, 0.4)
            print(f"[call_tool] {tool_name}: {type(e).__name__} on attempt "
                  f"{attempt + 1}/{retries}, retrying in {wait:.1f}s")
            await asyncio.sleep(wait)

    print(f"[call_tool] {tool_name} failed after {retries} attempts: {last_error!r}")
    raise last_error

# ------------------------
# Tavily MCP Tools
# ------------------------



async def tavily_search(query: str, max_results: int = 10):
    return await call_tool(
        "tavily_search",
        {"query": query, "max_results": max_results},
    )

async def list_airports(search: str = "", limit: int = 10):
    return await call_tool("list_airports", {"search": search, "limit": limit, "offset": 0})


async def list_airlines(search: str = "", limit: int = 10):
    return await call_tool("list_airlines", {"search": search, "limit": limit, "offset": 0})


async def current_weather(city: str):
    return await call_tool("get_current_weather", {"city": city})

async def forecast(city: str):
    return await call_tool("get_forecast", {"city": city})


async def flight_schedule(airport_iata: str, schedule_type: str = "departure",
                          number_of_flights: int = 40):
    return await call_tool("flight_arrival_departure_schedule", {
        "airport_iata_code": airport_iata,
        "schedule_type": schedule_type,
        "number_of_flights": number_of_flights,
    })