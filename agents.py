import asyncio
import json
import random
import time
import re

from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langgraph.types import interrupt

from config import get_llm
from mcp_client import current_weather, forecast, flight_schedule, tavily_search
from state import TravelState

llm = get_llm()


_RETRYABLE = (
    "rate limit", "429", "too many requests", "timeout", "timed out",
    "connection", "overloaded", "service unavailable", "503", "temporarily")

# Daily/monthly quota exhaustion. Retrying is pointless — the window is hours
# away, not seconds. Fail immediately instead of burning three attempts.
_QUOTA_EXHAUSTED = (
    "tokens per day", "requests per day", "tpd)", "rpd)",
    "quota exceeded", "insufficient_quota",
)



def _llm_text(system: str, prompt: str, retries: int = 3) -> str:
    """Call the LLM, retrying transient failures.

    Five LLM calls now fire in a single superstep once flight, hotel and
    weather run in parallel, so one rate-limit response would otherwise
    abort the whole run.
    """
    last_error = None

    for attempt in range(retries):
        try:
            response = llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=prompt),
            ])
            return response.content

        except Exception as e:
            last_error = e
            signature = f"{type(e).__name__} {e}".lower()

            if any(sig in signature for sig in _QUOTA_EXHAUSTED):
                print("[_llm_text] daily quota exhausted — not retrying")
                raise

            if not any(sig in signature for sig in _RETRYABLE):
                raise                       # a real bug — fail loudly


            if attempt == retries - 1:
                break

            wait = (2 ** attempt) + random.uniform(0, 0.5)
            print(f"[_llm_text] {type(e).__name__} on attempt {attempt + 1}, "
                  f"retrying in {wait:.1f}s")
            time.sleep(wait)

    raise last_error

   
def _resolve_city(query: str) -> str:
    """Resolve a travel query to ONE geocodable city."""
    city = _llm_text(
        "You extract a single city name. Return ONLY the city, nothing else.",
        f"""Extract ONE destination city from this travel query.
- If multiple places are mentioned, pick the main ARRIVAL destination.
- If a region or state is mentioned (Kashmir, Ladakh, Goa-the-state, Maharashtra),
  return its main city (Kashmir->Srinagar, Ladakh->Leh, Goa->Panaji, Maharashtra->Mumbai).
- If a COUNTRY or CONTINENT is mentioned, return its main gateway city
  (France->Paris, Germany->Berlin, Japan->Tokyo, Europe->Paris).
- Never answer "None" or "unknown". Always return one real city name.

Query:
{query}""",
    ).strip()

    if not city or city.lower() in {"none", "null", "n/a", "unknown", ""}:
        return ""
    return city

def _json_from_llm(text:str)->dict:
    print("\n=======RAW LLM RESPONSE=======")
    print(text)
    print("================================\n")

    start=text.index("{")
    end=text.rindex("}")+1

    json_text=text[start:end]


    print("\n=======EXTRACTED JSON=======")
    print(json_text)
    print("================================\n")

    return json.loads(json_text)

#with Guardrails


def supervisor_agent(state: TravelState):
    query = state["user_query"]

    # INPUT GUARDRAIL
    guardrail_prompt = f"""
You are a strict input validator for a TRAVEL PLANNING system.

ALLOW a request if it asks to plan, book, or get help with a trip, vacation,
flights, hotels, an itinerary, travel to a place, OR asks about weather,
climate,season, best time to visit, or what to pack for a destination.

REJECT anything else: general knowledge questions, definitions, coding
questions, or anything not related to travel or a destination.

Examples:
- "Plan a 5 day trip to Goa under 50k" -> {{"allowed": true, "reason": "travel request"}}
- "Cheap hotels in Paris" -> {{"allowed": true, "reason": "travel request"}}
- "weather in Lucknow" -> {{"allowed": true, "reason": "destination weather request"}}
- "best time to visit Bali" -> {{"allowed": true, "reason": "travel timing request"}}
- "What is Generative AI" -> {{"allowed": false, "reason": "not a travel request"}}
- "What is Agentic AI" -> {{"allowed": false, "reason": "not a travel request"}}
- "Write me Python code" -> {{"allowed": false, "reason": "not a travel request"}}

Return ONLY JSON in this exact format:
{{
    "allowed": true or false,
    "reason": "one short sentence"
}}

User request:
{query}
"""
    guardrail_raw = _llm_text(
        "You are an input validation guardrail. Return strict JSON only.",
        guardrail_prompt,
    )

    print("\n========== GUARDRAIL RAW RESPONSE ==========")
    print(guardrail_raw)
    print("============================================\n")

    guardrail_result = _json_from_llm(guardrail_raw)

    print("\n========== GUARDRAIL PARSED RESPONSE ==========")
    print(json.dumps(guardrail_result, indent=2))
    print("================================================\n")

    if not guardrail_result.get("allowed", False):
        reason = guardrail_result.get(
            "reason",
            "Request rejected by input guardrail."
        )
        return {
            "selected_agents": [],
            "trip_constraints": {},
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [
                AIMessage(content=f"Guardrail blocked request: {reason}")
            ],
            "llm_calls": 1,
        }

    # supervisor logic starts here (valid travel request)
    prompt = f"""
You are the supervisor of a real-world multi-agent travel planning system.

Decide which specialist agents are needed for this user request.

Available agents:
- flight_agent: use when flights, airports, airlines, routes, or airfare guidance are needed
- hotel_agent: use when hotels, stays, neighborhoods, or accommodation are needed
- weather_agent: use when weather, climate, season, packing, or forecast is useful
- budget_agent: use when budget, affordability, cost, or price constraints are mentioned
- itinerary_agent: almost always needed to produce the travel plan

Return only JSON with this schema:
{{
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""
}}

DESTINATION RULE (follow strictly):
- "destination" must be a single specific city. Never a country, region or
  continent.
- If the user names a broader area, resolve it to that area's main gateway
  city and put the city in "destination" (Europe->Paris, France->Paris,
  Germany->Berlin, Uttar Pradesh->Lucknow, Kashmir->Srinagar, Goa->Panaji).
- Keep the user's original wording in "special_preferences" if it matters.

User request:
{query}
"""
    raw = _llm_text(
        "You route work to specialist agents. Return strict JSON only.",
        prompt,
    )

    print("\n========== RAW LLM RESPONSE ==========")
    print(raw)
    print("======================================\n")

    parsed = _json_from_llm(raw)

    print("\n========== PARSED JSON ==========")
    print(json.dumps(parsed, indent=2))
    print("=================================\n")

    selected = parsed["selected_agents"]

    return {
        "selected_agents": selected,
        "trip_constraints": parsed["trip_constraints"],
        "supervisor_reasoning": parsed["reasoning"],
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": 1,
    }

def _extract_tavily_results(raw):
    """Pull a list of {title, url, content} out of a Tavily MCP response."""
    data = raw
    if isinstance(data, list) and data and isinstance(data[0], dict) and "text" in data[0]:
        data = data[0]["text"]
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return []
    results = data.get("results", []) if isinstance(data, dict) else []
    return [
        {
            "title": r.get("title", "Unknown"),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        }
        for r in results if isinstance(r, dict)
    ]
# ==============================================================================================================================
# Sources that appear in search results but aren't usable as booking references.
# Video and social pages produce mangled hotel names transcribed from captions.
BLOCKED_DOMAINS = (
    "youtube.com", "youtu.be", "instagram.com", "tiktok.com",
    "facebook.com", "reddit.com", "pinterest.com", "quora.com",
    "twitter.com", "x.com",
    "community.ricksteves.com", "/ShowTopic-", "/travel-forum/",
)

def _drop_low_quality(results):
    """Remove results with no URL or from video/social domains."""
    kept = [
        r for r in results
        if r.get("url") and not any(d in r["url"] for d in BLOCKED_DOMAINS)
    ]
    dropped = len(results) - len(kept)
    if dropped:
        print(f"[filter] dropped {dropped} social/video/no-url results")
    return kept

# General travel wikis never carry route-specific fare or schedule data.
FLIGHT_BLOCKED_DOMAINS = ("wikivoyage.org", "wikipedia.org", "wikitravel.org")

_FLIGHT_TERMS = ("flight", "airline", "airfare", "fare", "nonstop",
                 "non-stop", "direct", "airport")


# The supervisor rewrites regions to gateway cities (Goa -> Panaji), but web
# sources name the region. Both spellings must count as the same endpoint.
_CITY_ALIASES = {
    "panaji": ("goa",), "srinagar": ("kashmir",), "leh": ("ladakh",),
    "bengaluru": ("bangalore",), "bangalore": ("bengaluru",),
    "mumbai": ("bombay", "maharashtra"), "chennai": ("madras",),
    "kolkata": ("calcutta",), "puducherry": ("pondicherry",),
    "thiruvananthapuram": ("trivandrum",),
}


def _route_relevant(results, origin, destination, dep_iata, arr_iata):
    """Keep results about this specific route and about flying."""

    def _aliases(city, iata):
        out = set()
        if city:
            c = city.strip().lower()
            out.add(c)
            out.update(_CITY_ALIASES.get(c, ()))
        if iata:
            out.add(iata.strip().lower())
        return out

    def _mentions(blob, aliases):
        for a in aliases:
            # 3-letter tokens are IATA codes — need boundaries so GOI != going
            if len(a) == 3 and re.search(rf"\b{re.escape(a)}\b", blob):
                return True
            if len(a) > 3 and a in blob:
                return True
        return False

    dest_aliases = _aliases(destination, arr_iata)
    orig_aliases = _aliases(origin, dep_iata)

    kept = []
    for r in results:
        if any(d in r["url"] for d in FLIGHT_BLOCKED_DOMAINS):
            continue
        blob = f"{r['title']} {r['content']}".lower()
        if not any(t in blob for t in _FLIGHT_TERMS):
            continue
        if dest_aliases and not _mentions(blob, dest_aliases):
            continue
        # Origin unknown -> destination + flight terms is the strongest test
        # available. Don't drop everything just because we can't name a departure.
        if orig_aliases and not _mentions(blob, orig_aliases):
            continue
        kept.append(r)

    dropped = len(results) - len(kept)
    if dropped:
        print(f"[flight_agent] dropped {dropped} results not about "
              f"{dep_iata or origin or '?'}->{arr_iata or destination or '?'}")
    if results and not kept:
        print("[flight_agent] WARNING: route filter removed ALL results — "
              f"titles were: {[r['title'][:60] for r in results]}")
    return kept


def _resolve_iata(origin: str, destination: str) -> dict:
    """Resolve city/region names to airport IATA codes. One LLM call."""
    raw = _llm_text(
        "You return only JSON. No prose, no markdown fences.",
        f"""Give the primary commercial airport IATA code for each place.
If a region or state is given, use its main airport
(Kashmir->SXR, Goa->GOI, Maharashtra->BOM, Ladakh->IXL).
If a place is empty or unknown, use "".

Origin: {origin or "(none given)"}
Destination: {destination}

Return exactly: {{"origin_iata": "XXX", "destination_iata": "YYY"}}""",
    )
    try:
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start:end + 1])
        return {
            "origin_iata": (data.get("origin_iata") or "").strip().upper(),
            "destination_iata": (data.get("destination_iata") or "").strip().upper(),
        }
    except Exception:
        return {"origin_iata": "", "destination_iata": ""}


def _live_flights(dep_iata: str, arr_iata: str, sample: int = 40):
    """Real scheduled departures from dep_iata; returns (sample_size, matches)."""
    if not dep_iata:
        return 0, []
    try:
        raw = asyncio.run(flight_schedule(dep_iata, "departure", sample))
        data = raw[0]["text"] if isinstance(raw, list) and raw and "text" in raw[0] else raw
        flights = json.loads(data) if isinstance(data, str) else data
        if not isinstance(flights, list):
            return 0, []
        matches = [
            f for f in flights
            if isinstance(f, dict)
            and (f.get("arrival_airport_code") or "").upper() == arr_iata.upper()
        ]
        return len(flights), matches
    except Exception as e:
        print(f"[flight_agent] live schedule failed: {e!r}")
        return 0, []

    
# flight code

def flight_agent(state: TravelState):
    query = state["user_query"]
    constraints = state.get("trip_constraints", {}) or {}
    origin = constraints.get("origin", "")
    destination = constraints.get("destination", "")

    print("\n========== FLIGHT AGENT INPUT ==========")
    print("Query:", query)
    print("Constraints:", constraints)
    print("========================================\n")

    # 1. resolve to airport codes
    codes = _resolve_iata(origin, destination)
    dep_iata = codes["origin_iata"]
    arr_iata = codes["destination_iata"]
    print(f"[flight_agent] resolved {origin!r} -> {dep_iata}, {destination!r} -> {arr_iata}")

    # 2. LIVE scheduled flights from AviationStack
    sample_size, matches = _live_flights(dep_iata, arr_iata, sample=40)
    print(f"[flight_agent] sampled {sample_size} live departures from {dep_iata}, "
          f"{len(matches)} bound for {arr_iata}")

    if matches:
        live_block = "\n".join(
            f"- {f.get('airline','?')} {f.get('flight_number','?')} | "
            f"dep {f.get('departure_scheduled_time','?')} -> arr {f.get('arrival_scheduled_time','?')} | "
            f"aircraft {f.get('aircraft','?')} | terminal {f.get('arrival_terminal') or '-'}"
            for f in matches
        )
        live_note = (f"{len(matches)} of {sample_size} sampled live departures from "
                     f"{dep_iata} are bound for {arr_iata}:")
    else:
        live_block = "(none)"
        live_note = (f"No {arr_iata}-bound flights appeared in a random sample of "
                     f"{sample_size} live departures from {dep_iata}. This does NOT mean "
                     f"no such flights exist — the sample is partial.")

    # 3. web grounding for route context
    if dep_iata and arr_iata:
        route_query = (f"{dep_iata} to {arr_iata} flights {origin} to {destination} "
                       f"airlines nonstop fare price")
    else:
        route_query = (f"{origin} to {destination} flight route "
                       f"which airlines fly fare price")

    try:
        route_raw = asyncio.run(tavily_search(route_query, max_results=8))
        route_results = _extract_tavily_results(route_raw)
    except Exception as e:
        print(f"[flight_agent] web search failed: {e!r}")
        route_results = []
    route_results = _drop_low_quality(route_results)
    route_results = _route_relevant(route_results, origin, destination, dep_iata, arr_iata)

    route_sources = "\n".join(
        f"- {r['title']} | {r['url']}\n  {r['content'][:1200]}"
        for r in route_results
    ) or "(no web results specific to this route)"
    print(f"[flight_agent] parsed {len(route_results)} web route results")

    prompt = f"""
Create detailed flight guidance for this trip.

User request:
{query}

Trip constraints:
{constraints}

--- SOURCE A: AviationStack live schedule (real scheduled flights) ---
Resolved airports: {origin or '(origin not given)'} = {dep_iata or 'unknown'},
{destination} = {arr_iata or 'unknown'}
{live_note}
{live_block}

--- SOURCE B: Web search results ---
{route_sources}

Produce:
1. Departure and arrival airports with IATA codes
2. Airlines serving this route
3. Direct vs layover, and typical layover points
4. Typical flight duration
5. Fare range by cabin class
6. Best booking window and peak season warning
7. One booking tip

PROVENANCE RULES (follow strictly):
- Flights from SOURCE A are REAL live scheduled flights — present them as
  "live scheduled flights (AviationStack)" and list flight numbers and times.
- If SOURCE A had no matches, say plainly that no matching flights appeared in
  the sampled live data, and do NOT claim the route has no flights.
- Note that some listed carriers may be codeshare marketing airlines rather
  than the operating carrier.
- Facts from SOURCE B must be attributed to web sources; cite only URLs that
  appear above. Never invent a URL.
- Do NOT infer that an airline serves this route because its website appears
  in the results. Name an airline only if a source explicitly connects it to
  this specific city pair. An airline homepage listing a city as a destination
  is not evidence that it operates this route.
- If a source gives a duration range whose upper bound is more than double
  the lower bound, report the shortest and typical figures instead, and note
  that longer times reflect multi-stop itineraries.
- If SOURCE B is empty, cite no URLs at all. Do not list a source in order to
  note that it was off-topic.
- Sanity-check every fare's currency against the route. The origin is {origin or dep_iata}
  and the destination is {destination}. A fare is only plausible in a currency tied to the
  origin country, the destination country, or a major international currency (USD, EUR, GBP).
  If a fare appears in a currency unrelated to this route (for example an India-to-Europe
  fare quoted in Brazilian Real, Thai Baht, or similar), treat it as a scraping error:
  do NOT report that figure. Instead say the fare could not be reliably determined from the
  sources and point to the source link. Never repeat a fare whose currency does not fit the route.
- Anything supported by neither source must be labelled "general knowledge".

SCOPE (follow strictly):
- Cover flights only. Hotels, weather, and budget are handled by other agents.
- Do not discuss them, even to note that they're missing from your sources.
- Stop after point 7. Do not add a summary, a cost breakdown, or a general
  overview of the destination.
"""

    result = _llm_text("You are a flight planning specialist.", prompt)

    print("\n========== FLIGHT AGENT OUTPUT ==========")
    print(result)
    print("=========================================\n")

    return {
        "flight_results": result,
        "messages": [AIMessage(content="Flight agent completed.")],
        "llm_calls": 2,
    }



# Nightly-rate floors, symbol-aware. Below these, a "nightly rate" scraped from
# an aggregator page is a fragment (a tax line, a per-hour rate, a truncated
# figure), not a room price.
_RATE_FLOORS = {
    "$": 15, "USD": 15,
    "€": 15, "EUR": 15,
    "£": 12, "GBP": 12,
    "₹": 800, "INR": 800, "RS": 800,
}

_RATE_RE = re.compile(
    r"(?:from\s+)?"
    r"(?P<sym>₹|\$|€|£|INR|USD|EUR|GBP|Rs\.?)\s?"
    r"(?P<amt>\d[\d,]*(?:\.\d+)?)"
    r"(?P<suffix>\s*(?:per\s+night|/\s*night|a\s+night|per\s+room\s+per\s+night))?",
    re.IGNORECASE,
)


def _sanitize_hotel_prices(text: str) -> tuple[str, int]:
    """Replace implausibly low nightly rates with an explicit uncertainty marker.

    Runs on hotel_agent output before it reaches state, so the budget agent
    never sees a number the hotel agent couldn't stand behind.
    """
    suppressed = []

    def _repl(m):
        sym = m.group("sym").upper().rstrip(".")
        floor = _RATE_FLOORS.get(sym if len(sym) > 1 else m.group("sym"))
        if floor is None:
            return m.group(0)
        try:
            amount = float(m.group("amt").replace(",", ""))
        except ValueError:
            return m.group(0)
        if amount >= floor:
            return m.group(0)
        suppressed.append(m.group(0).strip())
        return "price not reliably determined"

    cleaned = _RATE_RE.sub(_repl, text)
    if suppressed:
        print(f"[hotel_agent] suppressed {len(suppressed)} implausible rates: {suppressed}")
    return cleaned, len(suppressed)

def hotel_agent(state: TravelState):
    constraints = state.get("trip_constraints", {}) or {}
    budget = constraints.get("budget") or "not specified"
    style = constraints.get("travel_style") or "not specified"

    city = _resolve_city(state["user_query"])

    if not city:
        print("[hotel_agent] skipped: no city could be resolved")
        return {
            "hotel_results": (
                "Hotel search was skipped because no destination city could be "
                "resolved from the request."
            ),
            "messages": [AIMessage(content="Hotel agent skipped; no city resolved.")],
            "llm_calls": 1,
        }

    style_hint = "" if style == "not specified" else f" {style}"
    query = f"best hotels in {city}{style_hint} where to stay areas price"

    try:
        raw = asyncio.run(tavily_search(query, max_results=10))

        print("\n========== HOTEL SEARCH RESULT ==========")
        print(raw)
        print("=========================================\n")

        results = _extract_tavily_results(raw)
    except Exception as e:
        print(f"[hotel_agent] web search failed: {e!r}")
        results = []

    results = _drop_low_quality(results)
    print(f"[hotel_agent] {len(results)} usable results for {city}")

    sources = "\n".join(
        f"- {r['title']} | {r['url']}\n  {r['content'][:1500]}"
        for r in results
    )

    summary = _llm_text(
        "You are a hotel recommendation specialist.",
        f"""If the search results section below is empty, do NOT list hotels from your
own knowledge. Reply exactly: "Hotel search data was unavailable for this request."

Otherwise, using ONLY the search results below, list up to 8 hotels.

User request: {state['user_query']}
Destination city: {city}
Stated budget: {budget}
Travel style: {style}

Search results (title | url | snippet):
{sources}

SELECTION RULES:
- Only list hotels located in {city} or its immediate surroundings. If a result
  is for a different city, skip it and say how many you dropped.
- Order the list to match the stated budget and travel style. If the budget is
  modest, lead with the most affordable options in the results and do not open
  with luxury or palace hotels.
- If the results contain nothing that fits the stated budget, say so plainly
  rather than presenting expensive hotels as though they fit.
- Sanity-check every rate against the destination. A nightly rate below about
  $15 / €15 / £12 / ₹800 is a scraping artefact, not a room price. Omit the
  price line for that hotel and write "price not reliably determined" instead.

For each hotel give:
- Hotel name (in bold)
- Source URL — the URL of the search result where the hotel was mentioned.
  Several hotels may share the same source URL if they came from the same
  listing page; that is correct and expected, not a problem. Only write
  "URL not available" if the hotel appears in no result that has a url.
  Never invent a url.
- Price only if the snippet states a nightly rate WITH an explicit currency
  symbol or code (₹, $, €, £, INR, USD, EUR, GBP). Quote it in that original
  currency, exactly as written. Never convert it, and never add a currency
  symbol the snippet did not contain. If the snippet shows a bare number with
  no currency, omit the price line entirely.
- One short line on area or hotel type if the snippet mentions it.

End with one booking tip. If fewer than 8 hotels appear, list all you found
and say so. Clean markdown only. No raw JSON, no relevance scores.""",
    )

    summary, suppressed = _sanitize_hotel_prices(summary)

    if suppressed:
        summary += (
            f"\n\n_Note: {suppressed} nightly rate(s) in the source snippets were "
            f"below a plausible floor for this city and have been suppressed as "
            f"unreliable. Do not treat their absence as a low price._"
        )

    return {
        "hotel_results": summary,
        "messages": [AIMessage(content="Hotel agent completed.")],
        "llm_calls": 2,
    }


def weather_agent(state: TravelState):
    city = "the destination"
    calls = 0

    try:
        city = _resolve_city(state["user_query"])
        calls += 1
        if not city:
            raise ValueError("could not resolve a city from the query")

        weather_data = asyncio.run(current_weather(city))
        forecast_data = asyncio.run(forecast(city))

        print("\n========== CURRENT WEATHER ==========")
        print(weather_data)
        print("========== FORECAST ==========")
        print(forecast_data)
        print("==============================\n")

        summary = _llm_text(
            "You are a travel weather specialist.",
            f"""Summarize the weather for {city} into clean guidance.

Current weather data:
{str(weather_data)[:2000]}

Forecast data:
{str(forecast_data)[:3000]}

Give: current conditions (temp, feels-like, humidity, sky), a short forecast
trend, and one packing tip. Clean prose/markdown only.
Do NOT output raw JSON, ids, or escaped characters.""",
        )
        calls += 1

        return {
            "weather_results": summary,
            "messages": [AIMessage(content="Weather agent completed.")],
            "llm_calls": calls,
        }

    except Exception as e:
        print(f"[weather_agent] failed for {city!r}: {e!r}")
        return {
            "weather_results": (
                f"Live weather was unavailable for {city} "
                f"({type(e).__name__}). Plan without a forecast and check "
                "conditions closer to the travel date."
            ),
            "messages": [AIMessage(
                content="Weather agent failed; continuing without a forecast."
            )],
            "llm_calls": calls,
        }

  
    

def budget_agent(state: TravelState):

    print("\n========== BUDGET AGENT INPUT ==========")
    print("Trip Constraints:")
    print(state.get("trip_constraints"))
    print("\nFlight Results:")
    print(state.get("flight_results"))
    print("\nHotel Results:")
    print(state.get("hotel_results"))
    print("\nWeather Results:")
    print(state.get("weather_results"))
    print("=========================================\n")

    prompt = f"""
Analyze whether this trip plan is realistic for the user's budget.

User request:
{state['user_query']}

Constraints:
{state.get('trip_constraints', {})}

Flight results:
{state.get('flight_results', '')}

Hotel results:
{state.get('hotel_results', '')}

Weather results:
{state.get('weather_results', '')}

Return a concise budget assessment with:
1. estimated cost categories
2. risk areas
3. money-saving suggestions
4. whether the plan seems feasible

COSTING RULES (follow strictly):
- Use the fare range stated in the flight results above. Do not substitute
  your own flight estimate, and do not widen or round the range.
- Use the nightly rates given in the hotel results. Multiply by the number
  of nights in the trip duration and show that multiplication.
- Only estimate a category when the results above contain no figure for it AND
  it is food, local transport, or activities. Label every such line "estimated".
- Never estimate flights or hotels. If the results give no fare or no nightly
  rate, that category is MISSING, not estimated. Write "not available from
  sources", exclude it from the total, and say the total is incomplete without
  it. Labelling an assumed hotel rate "estimated" does not make it permitted.
- Every figure you cite must trace to the results above or be labelled
  "estimated". Never mix the two silently.
- Your total must be the arithmetic sum of the categories you listed.
  Check the addition before writing it.
- If a needed figure is missing entirely, say which one is missing rather
  than inventing a placeholder number.
- Keep every figure in the currency it was given in. If a figure's currency is
  unclear or absent, treat it as missing rather than assuming the user's home
  currency.
- Do not convert any figure into a different currency. You have no exchange-rate
  source, so a converted number is an invented figure. This applies to the user's
  budget too: never restate their stated budget in another currency.
- If the cost categories end up in more than one currency, give a subtotal per
  currency (e.g. one USD subtotal, one GBP subtotal). Do not blend them into a
  single grand total.
- For feasibility, compare like currency with like currency only. If the budget
  and the costs are in different currencies, say the final comparison needs the
  user to convert at the current rate, and give the per-currency subtotals so they
  can. Do not perform that conversion yourself.
- Sanity-check each figure against the destination. If a nightly hotel rate or
  total is implausible for that city, say so rather than using it.
- Do not append an alternative calculation that fills a missing figure with an
  assumed value. If a figure is missing, the assessment ends with it missing.
  Never write "assuming X" and then produce a total.
"""

    result = _llm_text(
        "You are a practical travel budget analyst.",
        prompt,
    )

    print("\n========== BUDGET AGENT OUTPUT ==========")
    print(result)
    print("=========================================\n")

    return {
        "budget_results": result,
        "messages": [AIMessage(content="Budget agent completed.")],
        "llm_calls":  1,
    }


def itinerary_agent(state: TravelState):

    print("\n========== ITINERARY AGENT INPUT ==========")
    print("Trip Constraints:")
    print(state.get("trip_constraints"))

    print("\nFlight Results:")
    print(state.get("flight_results"))

    print("\nHotel Results:")
    print(state.get("hotel_results"))

    print("\nWeather Results:")
    print(state.get("weather_results"))

    print("\nBudget Results:")
    print(state.get("budget_results"))
    print("===========================================\n")

    prompt = f"""
Create a clear draft travel itinerary.

User request:
{state['user_query']}

Trip constraints:
{state.get('trip_constraints', {})}

Flight results:
{state.get('flight_results', '')}

Hotel results:
{state.get('hotel_results', '')}

Weather results:
{state.get('weather_results', '')}

Budget results:
{state.get('budget_results', '')}

Make the output structured, practical, and ready for human review.

REQUIRED SECTION — day-by-day plan:
- Include a "Day-by-Day Itinerary" section covering every single day of the
  trip duration ({state.get('trip_constraints', {}).get('duration', 'the trip')}).
- Each day gets its own heading (Day 1, Day 2, ...) with 2-3 concrete
  activities — specific places, areas, or landmarks, not generic filler.
- Ground the pacing in the weather results above (indoor options on rainy
  days, early starts in high heat).
- This section is mandatory. Do not omit or summarise it.

REQUIRED SECTION — weather:
- Under Weather Information, carry over all three parts of the weather
  results: current conditions, the forecast trend, and the packing tip.
- Do not reduce it to current conditions alone.

SOURCE URL RULES (follow strictly):
- Under Hotel Information, list the top 5 hotels. Put each hotel's Source URL
  on the line below its name, copied character-for-character from the hotel
  results above.
- Under Flight Information, list the source URL for every figure you carry
  over — fares, durations, booking windows. If you quote a number, its source
  must appear in the list. Include up to 6 links; if a figure's source would
  fall outside that limit, drop the figure rather than the link.
- Never invent, shorten, or tidy a URL. If a hotel has no URL above, write
  "URL not available".
- Do not list any hotel or airline that does not appear in the sections above.

SECTION RULES:
- Only include a section if the corresponding results were provided above.
  If flight, hotel, weather or budget results are empty, omit that section
  entirely — never write one from general knowledge.
"""

    result = _llm_text(
        "You are an expert itinerary planner.",
        prompt,
    )


    print("\n========== ITINERARY OUTPUT ==========")
    print(result)
    print("======================================\n")

    approval_request = f"""
Please review this draft travel plan.

{result}

Reply with approval or feedback.
"""

    return {
        "itinerary": result,
        "approval_request": approval_request,
        "messages": [AIMessage(content="Draft itinerary created for human review.")],
        "llm_calls":  1,
    }


def human_approval_agent(state: TravelState):
    feedback = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "expected_response": {
                "approved": True,
                "feedback": "Optional feedback for revision",
            },
        }
    )

    approved = feedback["approved"]
    human_feedback = feedback["feedback"]

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human approval step completed.")],
    }



def final_response_agent(state: TravelState):

    # If the input guardrail blocked the request, just return its message
    if not state.get("selected_agents") and state.get("final_response"):
        return {
            "final_response": state["final_response"],
            "messages": [AIMessage(content=state["final_response"])],
        }

    print("\n========== FINAL AGENT INPUT ==========")
    print("Approved:", state.get("approved"))
    print("Feedback:", state.get("human_feedback"))
    print("=======================================\n")

    sourced = f"""
Flight options (sourced):
{state.get('flight_results', '')}

Hotel options (sourced):
{state.get('hotel_results', '')}

Weather (sourced):
{state.get('weather_results', '')}

REQUIRED SECTION — day-by-day plan:
- Preserve the full day-by-day itinerary from the draft. Every day that
  appears in the draft must appear in the final plan, with its activities.
- Do not compress days, merge them, or replace them with a summary.

FIGURE RULES (follow strictly):
- Every number in the final plan must appear in the budget notes above.
  Do not introduce, refine, or fill in a figure the budget notes did not give.
- If the budget notes say a cost could not be estimated, repeat that. Do not
  supply your own estimate in its place.
- Your totals must match the budget notes' totals.
- Do not state a feasibility verdict unless every cost you cite and the user's
  stated budget are in the same currency. If they differ, repeat the
  per-currency subtotals and say the comparison requires the user to convert.
  Never write that something is "within budget" across currencies.

SECTION RULES:
- Only include a section if the corresponding results were provided above.
  If flight, hotel, weather or budget results are absent, omit that section
  entirely — never write one from general knowledge.

SOURCE URL RULES (follow strictly):
- Under Hotel Information, list the top 5 hotels. Put each hotel's Source URL
  on the line below its name, copied character-for-character from the sections
  above.
- Under Flight Information, list the source URL for every figure you carry
  over — fares, durations, booking windows. If you quote a number, its source
  must appear in the list. Include up to 6 links; if a figure's source would
  fall outside that limit, drop the figure rather than the link.
- Never invent, shorten, or tidy a URL. If you are listing a hotel and it has
  no URL in the sections above, write "URL not available" next to that hotel.
  If there are no hotels to list at all, omit the phrase entirely — do not
  write it on its own line.
- Do not list any hotel or airline that does not appear in the sections above.
  If the feedback asks for options not present there, say so instead of
  inventing them.
"""

    if state.get("approved"):
        prompt = f"""
The human approved this draft itinerary.

Produce the final polished travel plan.

Draft itinerary:
{state.get('itinerary', '')}

Budget notes:
{state.get('budget_results', '')}

The human feedback above supersedes the draft's choices and scope. It does not
supersede any figure: costs still come only from the budget notes.
{sourced}
"""
    else:
        prompt = f"""
The human did not approve the draft.

Original user request:
{state['user_query']}

Draft itinerary:
{state.get('itinerary', '')}

Human feedback:
{state.get('human_feedback', '')}

Budget notes:
{state.get('budget_results', '')}

The human feedback above supersedes the budget notes wherever they conflict.
{sourced}
"""

    result = _llm_text(
        "You produce final user-ready travel plans.",
        prompt,
    )

    print("\n========== FINAL RESPONSE ==========")
    print(result)
    print("====================================\n")

    return {
        "final_response": result,
        "messages": [AIMessage(content=result)],
        "llm_calls": 1,
    }