import asyncio
import os
import random
import sys
import threading
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

_HERE = Path(__file__).parent

from config import (
    AVIATION_STACK_API_KEY,
    OPENWEATHER_API_KEY,
    TAVILY_API_KEY,
)

# ------------------------
# MCP server configuration
# ------------------------

SERVERS = {
    "tavily": {
        "transport": "streamable_http",
        "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}",
    },
    "aviationstack": {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", "aviationstack_mcp", "mcp", "run"],
        "env": {"AVIATION_STACK_API_KEY": AVIATION_STACK_API_KEY},
    },
    "weather": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(_HERE / "weather_mcp_server.py")],
        "env": {"OPENWEATHER_API_KEY": OPENWEATHER_API_KEY},
    },
}

# Kept for any code that still imports `client` directly.
client = MultiServerMCPClient(SERVERS)

# One isolated client per server. Loading them separately means a server that
# fails to start degrades only its own capability, instead of aborting the
# whole tool-loading step and leaving every agent with nothing.
_server_clients = {
    name: MultiServerMCPClient({name: cfg}) for name, cfg in SERVERS.items()
}


# ------------------------
# Reliability layer
# ------------------------

TOOL_TIMEOUT = 45           # seconds per tool call
TOOL_RETRIES = 3
SERVER_LOAD_TIMEOUT = 30    # seconds to wait for one server's tool list

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


# server_name -> list of tools. Servers that fail are simply absent from the
# cache and get retried on the next call, rather than poisoning it permanently.
# threading.Lock (not asyncio.Lock) because agents are sync functions calling
# asyncio.run(), so each runs in its own thread with its own event loop.
_tools_cache: dict = {}
_tools_lock = threading.Lock()


async def get_tools():
    """Load tools from every MCP server independently.

    Each server is loaded through its own client so that one unreachable or
    misconfigured server cannot abort the others. Returns the union of all
    tools that loaded successfully.
    """
    with _tools_lock:
        pending = [name for name in _server_clients if name not in _tools_cache]

    for name in pending:
        try:
            tools = await asyncio.wait_for(
                _server_clients[name].get_tools(),
                timeout=SERVER_LOAD_TIMEOUT,
            )
            with _tools_lock:
                _tools_cache[name] = tools
            print(f"[mcp] '{name}' loaded {len(tools)} tool(s)")

        except Exception as e:
            print(
                f"[mcp] '{name}' unavailable, continuing without it: "
                f"{type(e).__name__}: {_error_signature(e)[:200]}"
            )

    with _tools_lock:
        if not _tools_cache:
            raise RuntimeError(
                "No MCP server could be reached — check network access and "
                "API keys for: " + ", ".join(SERVERS)
            )
        return [tool for tools in _tools_cache.values() for tool in tools]


def available_servers() -> list:
    """Names of MCP servers whose tools loaded successfully."""
    with _tools_lock:
        return sorted(_tools_cache)


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
        up = available_servers()
        raise ValueError(
            f"Tool '{tool_name}' unavailable — its MCP server failed to load. "
            f"Servers currently up: {', '.join(up) if up else 'none'}"
        )

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
# Tavily MCP tools
# ------------------------

async def tavily_search(query: str, max_results: int = 10):
    return await call_tool(
        "tavily_search",
        {"query": query, "max_results": max_results},
    )


# ------------------------
# AviationStack MCP tools
# ------------------------

async def list_airports(search: str = "", limit: int = 10):
    return await call_tool("list_airports", {"search": search, "limit": limit, "offset": 0})


async def list_airlines(search: str = "", limit: int = 10):
    return await call_tool("list_airlines", {"search": search, "limit": limit, "offset": 0})


async def flight_schedule(airport_iata: str, schedule_type: str = "departure",
                          number_of_flights: int = 40):
    return await call_tool("flight_arrival_departure_schedule", {
        "airport_iata_code": airport_iata,
        "schedule_type": schedule_type,
        "number_of_flights": number_of_flights,
    })


# ------------------------
# OpenWeather MCP tools (custom server)
# ------------------------

async def current_weather(city: str):
    return await call_tool("get_current_weather", {"city": city})


async def forecast(city: str):
    return await call_tool("get_forecast", {"city": city})