# Multi-Agent Travel Planner

An agentic travel planning system built on **LangGraph**. An LLM supervisor screens
each request, selects the minimum set of specialist agents, and runs the
independent ones **in parallel**. Every fact in the output traces back to a live
tool call or a retrieved web source — and when a source has nothing, the system
says so rather than inventing an answer.

Built as a study in two things that are usually skipped in agent demos:
**concurrent orchestration** and **provenance discipline**.

---

## Architecture

```mermaid
flowchart TD
    UI[Streamlit UI] --> SUP[Supervisor<br/>guardrail + agent selection]
    SUP -->|rejected| FIN
    SUP --> FL[Flight agent]
    SUP --> HO[Hotel agent]
    SUP --> WE[Weather agent]
    FL --> BU[Budget agent]
    HO --> BU
    WE --> BU
    BU --> IT[Itinerary agent]
    IT --> HA[Human approval<br/>graph interrupt]
    HA --> FIN[Final response]
    FIN --> OUT[PDF / Markdown / HTML]
```

The supervisor's routing function returns a **list** of agents, so LangGraph
dispatches flight, hotel and weather concurrently in a single super-step. All
three converge on the budget agent, which LangGraph schedules exactly once with
every upstream result populated.

### Node responsibilities

| Node | What it does | Data source |
|---|---|---|
| `supervisor_agent` | LLM guardrail returns a JSON verdict; rejects out-of-scope requests before any paid call. Then selects agents and resolves the destination to a single city. | LLM only |
| `flight_agent` | Resolves city pair to IATA codes, samples live departures, grounds the route against web search. | AviationStack (MCP) + Tavily (MCP) |
| `hotel_agent` | Resolves the destination city, searches, filters low-quality sources, lists only hotels found in results. | Tavily (MCP) |
| `weather_agent` | Resolves the city, fetches current conditions and forecast. | Custom OpenWeather MCP server |
| `budget_agent` | Costs the trip strictly from figures the upstream agents returned; labels every estimate; shows its arithmetic. | Upstream state only |
| `itinerary_agent` | Produces the day-by-day draft, preserving source URLs. | Upstream state only |
| `human_approval_agent` | Interrupts the graph for review. Approval or free-text revision resumes the same run. | Human |
| `final_response_agent` | Polishes the plan under figure and section rules; cannot introduce numbers the budget agent didn't give. | Upstream state only |

---

## MCP integration

Three servers, three transports, behind one `MultiServerMCPClient`:

| Server | Transport | Purpose |
|---|---|---|
| **Tavily** | Remote, streamable HTTP | Hotel listings and flight route grounding |
| **AviationStack** | Local, stdio subprocess | Live scheduled departures |
| **OpenWeather** | Local, stdio — **written for this project** | Current conditions and forecast |

Tools are sessionless, so each concurrent branch opens its own connection. Tool
discovery is cached behind a `threading.Lock` — the agents are sync functions
calling `asyncio.run()`, so LangGraph runs each in its own thread with its own
event loop, and an `asyncio.Lock` would not guard across them.

---

## Design decisions worth explaining

### Parallel fan-out with additive reducers

`state.py` annotates the shared counters:

```python
messages: Annotated[list[AnyMessage], operator.add]
llm_calls: Annotated[int, operator.add]
```

Without a reducer, two nodes writing the same key in one super-step raise
`InvalidUpdateError: can receive only one value per step`. With `operator.add`,
concurrent writes merge. The three data agents otherwise write **disjoint**
keys (`flight_results`, `hotel_results`, `weather_results`), so no other
channel needs one.

The fan-in is safe because all three branches are single-node and therefore
equal depth. Adding a second node to one branch would make the budget agent
fire early — that case needs `defer=True` on the join.

### Grounding rules, not grounding hopes

Each agent's prompt ends with an explicit rule block. These were written in
response to observed failures, not in the abstract:

- **Provenance labelling** — outputs distinguish live tool data, web sources,
  and model knowledge.
- **No invented URLs** — cite only URLs returned by the search tool.
- **Source filtering** — video, social and forum results are dropped before the
  model sees them. A YouTube listicle once produced a hotel that doesn't exist,
  transcribed from captions; a travel forum produced a chain name instead of a
  bookable property.
- **Currency preservation** — a bare `170` in a Paris snippet is not `₹170`.
  Quote the original currency or omit the price.
- **Cross-agent consistency** — the budget agent must cost from upstream
  figures, show its multiplication, and label every estimate. This closed a
  case where the sourced fare and the budget's own estimate diverged 3×.
- **No fabricated sections** — if an agent didn't run, its section is omitted
  rather than written from training data.
- **Refuse over improvise** — when search returns nothing, the agent says so.

### Failure containment

- `_llm_text` retries transient failures with exponential backoff, but fails
  **immediately** on daily-quota errors, where the reset window is hours away.
- `call_tool` wraps every MCP call in a 45-second timeout plus bounded retries.
  Error signatures are flattened through `ExceptionGroup` children, because MCP
  buries the real `ConnectTimeout` inside a generic TaskGroup wrapper.
- Each agent contains its own exceptions and returns a typed sentinel, so one
  failed branch degrades the plan instead of aborting the concurrent super-step.
- Postgres checkpointing makes runs resumable across a process restart.

---

## Setup

### Prerequisites

- Python 3.11 or 3.12
- PostgreSQL (local, or a hosted instance such as Neon or Supabase)
- API keys: Groq, Tavily, AviationStack, OpenWeather

### Install

```bash
git clone https://github.com/ashu-94/agentic-ai-travel-planner.git
cd <repo>

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Environment

Create a `.env` file (it is gitignored — never commit it):

```env
GROQ_API_KEY=...
TAVILY_API_KEY=...
AVIATION_STACK_API_KEY=...
OPENWEATHER_API_KEY=...
DATABASE_URL=postgresql://user:pass@host:5432/dbname?sslmode=require
```

### Run

```bash
streamlit run frontend.py
```

The two stdio MCP servers are spawned automatically as subprocesses of the app.

### Optional

- `pip install reportlab` enables PDF export
- `pip install markdown` improves the HTML export
- Put `flight.png`, `hotel.png`, `weather.png` in an `assets/` folder for the
  hero and card imagery; drawn SVG illustrations are used when they're absent

---

## Project structure

```
├── frontend.py             Streamlit UI, PDF/MD/HTML export
├── graph.py                LangGraph wiring, fan-out/fan-in, checkpointing
├── agents.py               All eight agents and their grounding rules
├── state.py                Typed shared state with concurrent reducers
├── mcp_client.py           MultiServerMCPClient, retries, timeouts, tool cache
├── config.py               Environment and LLM configuration
├── weather_mcp_server.py   Custom MCP server for OpenWeather
├── aviationstack-mcp/      Local MCP server for flight schedules
└── requirements.txt
```

---

## Limitations

These are real constraints, documented rather than hidden.

**AviationStack free tier.** 100 requests per month, and only the schedule and
airline endpoints are available — `list_airports`, `list_routes` and historical
data are restricted. The flight agent samples 40 live departures and filters by
arrival airport, so a route can genuinely exist and not appear in the sample.
The output says exactly that rather than claiming the route has no flights.

**Weather is near-term only.** OpenWeather's free tier gives current conditions
and a short forecast. For a trip six weeks out, "31°C in Lucknow right now"
isn't decision-useful. Seasonal guidance would need a climate-normals source.

**One destination city per trip.** The supervisor resolves countries and regions
to a single gateway city so all three agents work from the same value. A genuine
multi-city itinerary would need per-leg constraints.

**Hotel prices are often unavailable.** Aggregator listing pages rarely include
nightly rates in their search snippets. The system reports the rate as missing
rather than estimating one — correct, but it means the budget agent frequently
can't total accommodation.

**Output quality tracks search quality.** With a specific city pair, Tavily
returns deep route pages and the output is strong. With a vague destination it
returns homepages, and the flight agent has less to work with.

**Groq free tier limits throughput.** Roughly 100k tokens per day, and a full
five-agent run costs 25–30k, so about three or four complete plans per day.

**No automated evaluation yet.** Correctness has been verified by reading
outputs, not by a test suite or an eval harness. This is the next thing worth
building.

---

## Tech stack

**Orchestration** LangGraph · supervisor/router · parallel fan-out/fan-in ·
conditional edges · interrupts · PostgresSaver checkpointing

**LLM** Groq LLaMA 3.3 70B via LangChain

**Tools** MCP — remote streamable-HTTP and local stdio · Tavily · AviationStack ·
custom OpenWeather server

**Data** PostgreSQL

**Interface** Streamlit · reportlab PDF export

---

## License

MIT — see [LICENSE](LICENSE).