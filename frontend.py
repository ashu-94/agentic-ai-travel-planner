import base64
import html
import io
import re
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from graph import app


# ---------------------------------------------------------------- page config

st.set_page_config(
    page_title="Multi-Agent Travel Planner",
    page_icon="\u2708",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------- assets
# Drop images into an "assets" folder next to this file:
#   assets/flight.png    assets/hotel.png    assets/weather.png
# Any of .png .jpg .jpeg .webp works. Missing files fall back to drawn
# illustrations, so the app never breaks on a missing asset.

ASSETS = Path(__file__).parent / "assets"


def _find_asset(stem):
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        path = ASSETS / f"{stem}{ext}"
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def asset_uri(stem):
    """Inline an image as a data URI so it works without static file serving."""
    path = _find_asset(stem)
    if path is None:
        return ""
    mime = {
        ".png": "image/png", ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg", ".webp": "image/webp",
    }[path.suffix.lower()]
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


# ---------------------------------------------------------------------- theme

THEME = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --ink:      #0F1115;
    --surface:  #171A20;
    --raised:   #1E222A;
    --line:     #2B313B;
    --text:     #E7E9EE;
    --muted:    #939BA8;
    --sand:     #E0A458;
    --jade:     #5FB49C;
    --sky:      #7FB2E5;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.02em;
}

h1 { font-weight: 700 !important; font-size: 2.1rem !important; }
h2 { font-weight: 500 !important; font-size: 1.35rem !important; }
h3 { font-weight: 500 !important; font-size: 1.1rem !important; }

/* photo hero ------------------------------------------------------------- */
.hero {
    position: relative;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--line);
    background-size: cover;
    background-position: center 42%;
    min-height: 260px;
    display: flex;
    align-items: flex-end;
    margin-bottom: 1.6rem;
}
.hero .copy { padding: 1.6rem 1.9rem; max-width: 46rem; }
.hero .eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--sand);
    margin-bottom: 0.5rem;
}
.hero .title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.3rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.05;
    color: #FFFFFF;
    text-shadow: 0 2px 14px rgba(0,0,0,0.55);
}
.hero .sub {
    color: #C9D0DA;
    font-size: 0.94rem;
    margin-top: 0.55rem;
    line-height: 1.5;
    text-shadow: 0 1px 10px rgba(0,0,0,0.6);
}

/* plain masthead, used when no hero image is present --------------------- */
.masthead {
    border-bottom: 1px solid var(--line);
    padding-bottom: 1.1rem;
    margin-bottom: 1.6rem;
}
.masthead .eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--sand);
    margin-bottom: 0.45rem;
}
.masthead .title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.1rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.1;
    color: var(--text);
}
.masthead .sub { color: var(--muted); font-size: 0.92rem; margin-top: 0.4rem; }

/* route hero ------------------------------------------------------------- */
.route {
    display: flex;
    align-items: center;
    gap: 1.1rem;
    padding: 1rem 1.3rem;
    border: 1px solid var(--line);
    border-radius: 6px;
    background: var(--surface);
    margin: 0 0 1.3rem 0;
    flex-wrap: wrap;
}
.route .place {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem;
    font-weight: 500;
    color: var(--text);
    line-height: 1.2;
}
.route .placelabel {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
}
.route .sep { color: var(--sand); display: flex; align-items: center; }
.route .facts { margin-left: auto; display: flex; gap: 1.6rem; align-items: center; }
.route .fact {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    color: var(--muted);
    font-size: 0.86rem;
}
.route .fact svg { color: var(--sand); flex-shrink: 0; }

/* agent pipeline strip -------------------------------------------------- */
.pipeline { display: flex; align-items: center; gap: 0; flex-wrap: wrap; margin: 0.2rem 0 1.4rem 0; }
.stage {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.04em;
    padding: 0.36rem 0.7rem;
    border: 1px solid var(--line);
    border-radius: 3px;
    color: var(--muted);
    background: var(--surface);
    white-space: nowrap;
}
.stage.on { color: var(--ink); background: var(--sand); border-color: var(--sand); font-weight: 500; }
.stage.done { color: var(--jade); border-color: var(--jade); background: transparent; }
.rule { width: 18px; height: 1px; background: var(--line); flex-shrink: 0; }

/* meta row --------------------------------------------------------------- */
.metarow { display: flex; gap: 2.2rem; margin: 0 0 1.4rem 0; }
.metric .k {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
}
.metric .v {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.45rem;
    font-weight: 500;
    color: var(--text);
    line-height: 1.3;
}

/* panel header with icon ------------------------------------------------- */
.panelhead {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.15rem 0 0.9rem 0;
    border-bottom: 1px solid var(--line);
    margin-bottom: 1rem;
}
.panelhead svg { color: var(--sand); flex-shrink: 0; }
.panelhead .pt {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 500;
    color: var(--text);
}
.panelhead .ps { color: var(--muted); font-size: 0.82rem; margin-left: auto; text-align: right; }

/* illustrated / photo cards --------------------------------------------- */
.illos { display: flex; gap: 1rem; margin: 0.6rem 0 0 0; flex-wrap: wrap; }
.illo {
    flex: 1 1 220px;
    border: 1px solid var(--line);
    border-radius: 6px;
    background: var(--surface);
    overflow: hidden;
}
.illo .art { display: block; width: 100%; height: 130px; object-fit: cover; }
.illo svg.art { height: auto; background: var(--surface); }
.illo .cap { padding: 0.75rem 1rem 1rem 1rem; }
.illo .cap .t {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.98rem;
    font-weight: 500;
    color: var(--text);
    margin-bottom: 0.2rem;
}
.illo .cap .d { color: var(--muted); font-size: 0.82rem; line-height: 1.45; }

.empty { color: var(--muted); font-size: 0.9rem; font-style: italic; padding: 0.8rem 0; }

.sectionlabel {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--sand);
    margin: 2rem 0 0.5rem 0;
}

.stTabs [data-baseweb="tab-list"] { gap: 0.35rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    letter-spacing: 0.06em;
    padding: 0.55rem 0.9rem;
}

section[data-testid="stSidebar"] { border-right: 1px solid var(--line); }

div[data-testid="stDownloadButton"] button {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
    letter-spacing: 0.05em;
    border-radius: 3px;
    width: 100%;
}
</style>
"""

st.markdown(THEME, unsafe_allow_html=True)


# --------------------------------------------------------------------- icons
# Inline SVG rather than image files: no network fetch, no licensing, and they
# inherit the theme colour so they survive a palette change.

def _svg(paths, size=20):
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="1.6" stroke-linecap="round" '
        f'stroke-linejoin="round">{paths}</svg>'
    )


ICON_PLANE = _svg('<path d="M17.8 19.2 16 11l3.5-3.5a2.1 2.1 0 0 0-3-3L13 8 4.8 6.2a.5.5 0 0 0-.5.8l3.2 3.6-2 2-2.3-.4a.5.5 0 0 0-.5.8L5 15.5 6.5 18l2.4 2.3a.5.5 0 0 0 .8-.5l-.4-2.3 2-2 3.6 3.2a.5.5 0 0 0 .9-.5Z"/>')
ICON_BED = _svg('<path d="M2 20v-8a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v8"/><path d="M2 16h20"/><path d="M6 10V7a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v3"/><path d="M12 10V7a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v3"/>')
ICON_SUN = _svg('<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>')
ICON_CLOUD = _svg('<path d="M17.5 19a4.5 4.5 0 0 0 .5-8.98 6 6 0 0 0-11.7 1.2A3.5 3.5 0 0 0 6.5 19Z"/>')
ICON_RAIN = _svg('<path d="M17.5 15a4.5 4.5 0 0 0 .5-8.98 6 6 0 0 0-11.7 1.2A3.5 3.5 0 0 0 6.5 15Z"/><path d="M8 19v2M12 18v3M16 19v2"/>')
ICON_WALLET = _svg('<path d="M3 7a2 2 0 0 1 2-2h13a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/><path d="M3 10h18"/><circle cx="16.5" cy="14.5" r="1.2"/>')
ICON_CALENDAR = _svg('<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 10h18M8 3v4M16 3v4"/>')
ICON_CLOCK = _svg('<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>', 16)
ICON_PIN = _svg('<path d="M12 21s-7-5.6-7-11a7 7 0 0 1 14 0c0 5.4-7 11-7 11Z"/><circle cx="12" cy="10" r="2.6"/>', 16)
ICON_COIN = _svg('<circle cx="12" cy="12" r="9"/><path d="M9 8h6M9 12h6M10.5 8v8"/>', 16)
ICON_ARROW = _svg('<path d="M5 12h14M13 6l6 6-6 6"/>', 22)


def weather_icon(text):
    """Pick an icon from the forecast wording, so it reflects real data."""
    low = (text or "").lower()
    if any(w in low for w in ("rain", "shower", "drizzle", "storm", "thunder")):
        return ICON_RAIN
    if any(w in low for w in ("cloud", "overcast", "fog", "mist", "haze")):
        return ICON_CLOUD
    return ICON_SUN


def panel_head(icon, title, note=""):
    note_html = f'<div class="ps">{html.escape(note)}</div>' if note else ""
    return (f'<div class="panelhead">{icon}'
            f'<div class="pt">{html.escape(title)}</div>{note_html}</div>')


# --------------------------------------------------- drawn fallback artwork

_SAND, _JADE, _SKY = "#E0A458", "#5FB49C", "#7FB2E5"
_LINE, _MUTED, _DEEP = "#2B313B", "#3A4250", "#242B36"

ART_FLIGHT = f"""
<svg class="art" viewBox="0 0 280 120" xmlns="http://www.w3.org/2000/svg">
<g fill="none" stroke-linecap="round" stroke-linejoin="round">
  <path d="M0 104h280" stroke="{_LINE}" stroke-width="1"/>
  <path d="M14 104V88h11v16M31 104V76h14v28M51 104V94h9v10
           M226 104V82h13v22M245 104V92h10v12" stroke="{_MUTED}" stroke-width="1.4"/>
  <path d="M22 62q9-7 18 0q7-5 13 1H16q0-4 6-1Z" fill="{_DEEP}" stroke="none"/>
  <path d="M196 74q11-8 21 0q8-6 15 1h-45q0-5 9-1Z" fill="{_DEEP}" stroke="none"/>
  <path d="M28 96C70 96 120 70 168 40" stroke="{_SAND}" stroke-width="1.4"
        stroke-dasharray="4 5" opacity="0.75"/>
  <circle cx="28" cy="96" r="3" fill="{_SAND}" stroke="none"/>
  <g transform="translate(168 40) rotate(-32)">
    <path d="M0 0 26-8 18 0 26 8Z" fill="{_SAND}" stroke="none"/>
    <path d="M8-1 2-12 6-13 16-2Z" fill="{_SAND}" stroke="none" opacity="0.8"/>
    <path d="M8 1 2 12 6 13 16 2Z" fill="{_SAND}" stroke="none" opacity="0.8"/>
  </g>
</g></svg>
"""

ART_HOTEL = f"""
<svg class="art" viewBox="0 0 280 120" xmlns="http://www.w3.org/2000/svg">
<g fill="none" stroke-linecap="round" stroke-linejoin="round">
  <path d="M0 104h280" stroke="{_LINE}" stroke-width="1"/>
  <rect x="92" y="26" width="96" height="78" fill="{_DEEP}" stroke="{_MUTED}" stroke-width="1.4"/>
  <path d="M88 26h104l-8-10H96Z" fill="{_DEEP}" stroke="{_MUTED}" stroke-width="1.4"/>
  <g fill="{_SAND}" stroke="none">
    <rect x="102" y="36" width="13" height="11" rx="1.5"/>
    <rect x="145" y="36" width="13" height="11" rx="1.5" opacity="0.35"/>
    <rect x="166" y="36" width="13" height="11" rx="1.5"/>
    <rect x="102" y="56" width="13" height="11" rx="1.5" opacity="0.35"/>
    <rect x="123" y="56" width="13" height="11" rx="1.5"/>
    <rect x="166" y="56" width="13" height="11" rx="1.5" opacity="0.35"/>
    <rect x="123" y="76" width="13" height="11" rx="1.5"/>
    <rect x="145" y="76" width="13" height="11" rx="1.5" opacity="0.35"/>
  </g>
  <g fill="none" stroke="{_MUTED}">
    <rect x="123" y="36" width="13" height="11" rx="1.5"/>
    <rect x="145" y="56" width="13" height="11" rx="1.5"/>
    <rect x="102" y="76" width="13" height="11" rx="1.5"/>
    <rect x="166" y="76" width="13" height="11" rx="1.5"/>
  </g>
  <path d="M62 104V70h22v34M196 104V78h20v26" stroke="{_MUTED}" stroke-width="1.4"/>
  <g stroke="{_JADE}" opacity="0.6">
    <path d="M40 104c0-12 6-18 12-18s12 6 12 18" stroke-width="1.4"/>
    <path d="M52 104V88" stroke-width="1.2"/>
    <path d="M226 104c0-10 5-15 10-15s10 5 10 15" stroke-width="1.4"/>
    <path d="M236 104V92" stroke-width="1.2"/>
  </g>
</g></svg>
"""

ART_WEATHER = f"""
<svg class="art" viewBox="0 0 280 120" xmlns="http://www.w3.org/2000/svg">
<g fill="none" stroke-linecap="round" stroke-linejoin="round">
  <path d="M0 104h280" stroke="{_LINE}" stroke-width="1"/>
  <circle cx="176" cy="42" r="19" fill="{_SAND}" opacity="0.9" stroke="none"/>
  <g stroke="{_SAND}" stroke-width="1.6" opacity="0.7">
    <path d="M176 12v-6M198 42h6M192 26l4-4M192 58l4 4M160 26l-4-4"/>
  </g>
  <path d="M78 62q6-22 26-22 14 0 20 12 4-3 9-3 13 0 15 13 12 1 12 12 0 12-13 12H70
           q-14 0-14-13 0-12 13-13Z" fill="{_DEEP}" stroke="{_MUTED}" stroke-width="1.5"/>
  <path d="M186 74q8-14 20-6 9-5 14 5 10 0 10 9 0 8-10 8h-38q-9 0-9-8 0-8 9-8Z"
        fill="{_DEEP}" stroke="{_MUTED}" stroke-width="1.3" opacity="0.8"/>
  <g stroke="{_SKY}" stroke-width="1.7" opacity="0.75">
    <path d="M86 96v9M104 92v11M122 96v9M140 92v11"/>
  </g>
</g></svg>
"""


def _card_art(stem, fallback_svg, alt):
    """Use the photo if the asset exists, otherwise the drawn illustration."""
    uri = asset_uri(stem)
    if uri:
        return f'<img class="art" src="{uri}" alt="{html.escape(alt)}"/>'
    return fallback_svg


def empty_state():
    """Shown before the first plan, so the page isn't a blank dark slab."""
    cards = [
        ("flight", ART_FLIGHT, "Flights", "Aircraft above the clouds",
         "Live departure schedules from AviationStack, grounded against route "
         "pages found by web search."),
        ("hotel", ART_HOTEL, "Hotels", "Hotel exterior at dusk",
         "Real listings with the source URL each one came from. If search "
         "returns nothing, it says so."),
        ("weather", ART_WEATHER, "Weather", "Sky and cloud conditions",
         "Current conditions and a short forecast from a custom OpenWeather "
         "MCP server."),
    ]
    blocks = "".join(
        f'<div class="illo">{_card_art(stem, svg, alt)}'
        f'<div class="cap"><div class="t">{title}</div>'
        f'<div class="d">{desc}</div></div></div>'
        for stem, svg, title, alt, desc in cards
    )
    return f'<div class="illos">{blocks}</div>'


def header_block():
    """Photo hero if assets/flight.* exists, otherwise the plain masthead."""
    eyebrow = "LangGraph · MCP · Groq LLaMA 3.3"
    title = "Multi-agent travel planner"
    sub = ("Live flight schedules, sourced hotels and real forecasts, "
           "assembled by specialist agents and reviewed by you before "
           "anything is final.")

    uri = asset_uri("flight")
    if not uri:
        return ('<div class="masthead">'
                f'<div class="eyebrow">{eyebrow}</div>'
                f'<div class="title">{title}</div>'
                f'<div class="sub">{sub}</div></div>')

    # Gradient keeps the copy legible over any photograph.
    overlay = ("linear-gradient(90deg, rgba(15,17,21,0.94) 0%, "
               "rgba(15,17,21,0.80) 42%, rgba(15,17,21,0.30) 100%), "
               "linear-gradient(0deg, rgba(15,17,21,0.85) 0%, "
               "rgba(15,17,21,0.10) 60%)")
    return (f'<div class="hero" style="background-image:{overlay}, url({uri});">'
            f'<div class="copy">'
            f'<div class="eyebrow">{eyebrow}</div>'
            f'<div class="title">{title}</div>'
            f'<div class="sub">{sub}</div>'
            f'</div></div>')


# ------------------------------------------------------------------- helpers

AGENT_LABELS = {
    "flight_agent": "flight",
    "hotel_agent": "hotel",
    "weather_agent": "weather",
    "budget_agent": "budget",
    "itinerary_agent": "itinerary",
}


def md(text):
    """Streamlit reads $...$ as LaTeX, which eats currency amounts like
    '$105 - $150'. Escaping the dollar signs keeps fares readable."""
    return (text or "").replace("$", "\\$")


def show(text, empty_note="This agent was not part of the plan."):
    if text and text.strip():
        st.markdown(md(text))
    else:
        st.markdown(f'<div class="empty">{empty_note}</div>', unsafe_allow_html=True)


def route_hero(constraints, weather_text):
    """Origin, destination and trip facts, with a live weather glyph."""
    origin = (constraints.get("origin") or "").strip()
    destination = (constraints.get("destination") or "").strip()
    duration = (constraints.get("duration") or "").strip()
    budget = (constraints.get("budget") or "").strip()

    if not destination:
        return ""

    left = ""
    if origin:
        left = (f'<div><div class="placelabel">from</div>'
                f'<div class="place">{html.escape(origin)}</div></div>'
                f'<div class="sep">{ICON_ARROW}</div>')

    facts = []
    if duration:
        facts.append(f'<div class="fact">{ICON_CLOCK}{html.escape(duration)}</div>')
    if budget:
        facts.append(f'<div class="fact">{ICON_COIN}{html.escape(budget)}</div>')
    if weather_text and weather_text.strip():
        facts.append(f'<div class="fact">{weather_icon(weather_text)}forecast in</div>')
    facts.append(f'<div class="fact">{ICON_PIN}{html.escape(destination)}</div>')

    return ('<div class="route">'
            f'{left}'
            f'<div><div class="placelabel">to</div>'
            f'<div class="place">{html.escape(destination)}</div></div>'
            f'<div class="facts">{"".join(facts)}</div>'
            '</div>')


def pipeline_strip(selected, stage):
    """Which specialists the supervisor picked, in run order."""
    order = ["supervisor", "flight_agent", "hotel_agent", "weather_agent",
             "budget_agent", "itinerary_agent", "approval", "final"]
    names = {
        "supervisor": "supervisor",
        "approval": "review",
        "final": "final plan",
        **AGENT_LABELS,
    }
    reached = order.index(stage) if stage in order else -1

    chips = []
    for i, node in enumerate(order):
        if node in AGENT_LABELS and node not in selected:
            continue
        if i < reached:
            cls = "stage done"
        elif i == reached:
            cls = "stage on"
        else:
            cls = "stage"
        chips.append(f'<span class="{cls}">{names[node]}</span>')

    return '<div class="pipeline">' + '<span class="rule"></span>'.join(chips) + "</div>"


def metarow(result):
    agents = len(result.get("selected_agents") or [])
    calls = result.get("llm_calls", 0)
    return (
        '<div class="metarow">'
        f'<div class="metric"><div class="k">specialists</div><div class="v">{agents}</div></div>'
        f'<div class="metric"><div class="k">llm calls</div><div class="v">{calls}</div></div>'
        f'<div class="metric"><div class="k">mcp servers</div><div class="v">3</div></div>'
        "</div>"
    )


def build_markdown(result, query):
    """Assemble the full plan as a portable markdown document."""
    stamp = datetime.now().strftime("%d %B %Y, %H:%M")
    parts = [
        "# Travel plan",
        "",
        f"**Request:** {query}",
        f"**Generated:** {stamp}",
        "",
        "---",
        "",
        "## Final plan",
        "",
        result.get("final_response", "") or result.get("itinerary", ""),
        "",
        "---",
        "",
        "## Supporting research",
        "",
    ]
    for label, key in [
        ("Flights", "flight_results"),
        ("Hotels", "hotel_results"),
        ("Weather", "weather_results"),
        ("Budget", "budget_results"),
    ]:
        body = result.get(key, "")
        if body and body.strip():
            parts += [f"### {label}", "", body, ""]

    used = []
    if result.get("flight_results", "").strip():
        used.append("flight routes via AviationStack and Tavily web search")
    if result.get("hotel_results", "").strip():
        used.append("hotels via Tavily web search")
    if result.get("weather_results", "").strip():
        used.append("weather via OpenWeather")

    if used:
        footer = "_Sources: " + "; ".join(used) + ". "
    else:
        footer = "_"
    footer += ("Always confirm fares and availability with the operator "
               "before booking._")

    parts += ["---", "", footer]
    return "\n".join(parts)


HTML_SHELL = """<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
body { font-family: Georgia, 'Times New Roman', serif; max-width: 46rem;
       margin: 3rem auto; padding: 0 1.5rem; line-height: 1.65; color: #1a1a1a; }
h1, h2, h3 { font-family: Helvetica, Arial, sans-serif; letter-spacing: -0.01em; }
h1 { font-size: 1.9rem; border-bottom: 2px solid #E0A458; padding-bottom: 0.4rem; }
h2 { font-size: 1.25rem; margin-top: 2.2rem; }
h3 { font-size: 1.02rem; color: #555; }
a { color: #1a5f8a; word-break: break-all; }
pre { white-space: pre-wrap; font-family: inherit; font-size: 1rem; }
hr { border: none; border-top: 1px solid #ddd; margin: 2rem 0; }
@media print { body { margin: 0; max-width: none; } }
</style></head>
<body>__BODY__</body></html>
"""


def build_html(markdown_text, title="Travel plan"):
    """Printable HTML. The browser print dialog turns this into a PDF."""
    try:
        import markdown as _md

        body = _md.markdown(markdown_text, extensions=["extra", "sane_lists"])
    except Exception:
        body = "<pre>" + (
            markdown_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        ) + "</pre>"
    return HTML_SHELL.replace("__TITLE__", title).replace("__BODY__", body)


def _pdf_font():
    """Find a Unicode TTF. Helvetica is Latin-1 only, so the rupee sign and
    accented characters would render as black boxes without this."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    candidates = [
        ("Arial", "Arial-Bold",
         "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
        ("Segoe", "Segoe-Bold",
         "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"),
        ("DejaVuSans", "DejaVuSans-Bold",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    for regular, bold, reg_path, bold_path in candidates:
        try:
            pdfmetrics.registerFont(TTFont(regular, reg_path))
            pdfmetrics.registerFont(TTFont(bold, bold_path))
            pdfmetrics.registerFontFamily(regular, normal=regular, bold=bold)
            return regular, bold
        except Exception:
            continue
    return "Helvetica", "Helvetica-Bold"


_LINK_RE = re.compile(r"(https?://[^\s<>\)\]]+)")


def _pdf_inline(text):
    """Escape, then re-add bold, italic and clickable links."""
    out = html.escape(text, quote=False)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", out)
    return _LINK_RE.sub(r'<link href="\1" color="#1a5f8a">\1</link>', out)


@st.cache_data(show_spinner=False)
def build_pdf(markdown_text, title="Travel plan"):
    """Render the plan markdown to PDF bytes.

    Cached because Streamlit reruns the whole script on every widget click,
    and rebuilding the document each time is wasted work.
    """
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        HRFlowable, ListFlowable, ListItem, Paragraph,
        SimpleDocTemplate, Spacer,
    )

    regular, bold = _pdf_font()

    body = ParagraphStyle("body", fontName=regular, fontSize=9.5, leading=14,
                          spaceAfter=5, alignment=TA_LEFT, textColor="#1a1a1a")
    h1 = ParagraphStyle("h1", parent=body, fontName=bold, fontSize=19,
                        leading=23, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=body, fontName=bold, fontSize=13.5,
                        leading=17, spaceBefore=14, spaceAfter=5)
    h3 = ParagraphStyle("h3", parent=body, fontName=bold, fontSize=11,
                        leading=14, spaceBefore=10, spaceAfter=4)
    h4 = ParagraphStyle("h4", parent=body, fontName=bold, fontSize=10,
                        leading=13, spaceBefore=8, spaceAfter=3,
                        textColor="#444444")
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=10, spaceAfter=2)

    story, pending = [], []

    def flush():
        if not pending:
            return
        story.append(ListFlowable(
            [ListItem(Paragraph(b, bullet), leftIndent=12) for b in pending],
            bulletType="bullet", bulletFontName=regular, bulletFontSize=6,
            start="circle", leftIndent=12, spaceAfter=6,
        ))
        pending.clear()

    lines = markdown_text.splitlines()
    i = -1
    while i + 1 < len(lines):
        i += 1
        line = lines[i].strip()

        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if line and re.fullmatch(r"={3,}", nxt):
            flush()
            story.append(Paragraph(_pdf_inline(line), h2))
            i += 1
            continue

        if not line:
            flush()
            story.append(Spacer(1, 3))
            continue

        if line in ("---", "***", "___"):
            flush()
            story.append(Spacer(1, 5))
            story.append(HRFlowable(width="100%", thickness=0.6,
                                    color="#cccccc", spaceAfter=7))
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)", line)
        if heading:
            flush()
            style = {1: h1, 2: h2, 3: h3}.get(len(heading.group(1)), h4)
            story.append(Paragraph(_pdf_inline(heading.group(2)), style))
            continue

        item = re.match(r"^[-*+]\s+(.*)", line) or re.match(r"^\d+[\.\)]\s+(.*)", line)
        if item:
            pending.append(_pdf_inline(item.group(1)))
            continue

        flush()
        story.append(Paragraph(_pdf_inline(line), body))

    flush()

    buf = io.BytesIO()
    SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title=title, author="Multi-Agent Travel Planner",
    ).build(story)
    return buf.getvalue()


def slug(text, limit=40):
    keep = "".join(c if c.isalnum() else "-" for c in (text or "plan").lower())
    while "--" in keep:
        keep = keep.replace("--", "-")
    return keep.strip("-")[:limit] or "travel-plan"


def explain_failure(e):
    """Turn an exception into one plain sentence. A red traceback in a demo
    reads as broken; a sentence reads as handled."""
    detail = f"{type(e).__name__}: {e}".lower()

    if "tokens per day" in detail or "requests per day" in detail:
        return ("The language model's daily quota has been used up. Planning "
                "will work again once the quota resets, usually within a few "
                "hours.")
    if "rate limit" in detail or "429" in detail:
        return ("The language model is rate limited right now. Wait a minute "
                "and try again.")
    if "authentication" in detail or "api key" in detail or "401" in detail:
        return "An API key was rejected. Check the keys in your environment."
    if "connect" in detail or "timeout" in detail:
        return ("A data source could not be reached after several attempts. "
                "Try again in a moment.")
    return (f"Planning failed with {type(e).__name__}. The terminal has the "
            "full details.")


# ------------------------------------------------------------------- sidebar

with st.sidebar:
    st.markdown("### Session")
    user_id = st.text_input("User ID", value="demo_user")

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"{user_id}_{uuid.uuid4().hex[:8]}"

    if st.button("Start new thread", use_container_width=True):
        st.session_state.thread_id = f"{user_id}_{uuid.uuid4().hex[:8]}"
        st.session_state.pop("waiting_for_approval", None)
        st.session_state.pop("latest_result", None)
        st.session_state.pop("query", None)
        st.rerun()

    st.caption(f"Thread `{st.session_state.thread_id}`")
    st.caption("Each thread is checkpointed in Postgres, so a plan can be "
               "resumed after the review step.")

    st.divider()
    st.markdown("### Pipeline")

    st.markdown(
        f'<div class="route" style="flex-direction:column;align-items:flex-start;'
        f'gap:0.55rem;padding:0.9rem 1rem;">'
        f'<div class="fact">{ICON_PLANE}Flights · AviationStack + Tavily</div>'
        f'<div class="fact">{ICON_BED}Hotels · Tavily search</div>'
        f'<div class="fact">{ICON_SUN}Weather · custom OpenWeather server</div>'
        f'<div class="fact">{ICON_WALLET}Budget · grounded in the above</div>'
        f'<div class="fact">{ICON_CALENDAR}Itinerary · day by day</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "A supervisor screens the request, then flight, hotel and weather "
        "specialists run in parallel. Their results converge on the budget "
        "and itinerary agents before you review the draft."
    )


# -------------------------------------------------------------------- header

st.markdown(header_block(), unsafe_allow_html=True)


# --------------------------------------------------------------- input panel

query = st.text_area(
    "Where are you going?",
    value=st.session_state.get("query", ""),
    placeholder="Plan an 8-day trip from Hyderabad to Lucknow with budget "
                "hotels, a day-by-day itinerary and weather.",
    height=110,
)

col_run, col_hint = st.columns([1, 4])
with col_run:
    run = st.button("Create draft plan", type="primary", use_container_width=True)
with col_hint:
    st.caption("Non-travel requests are turned away by the input guardrail "
               "before any agent runs.")

config = {"configurable": {"thread_id": st.session_state.thread_id}}

if run:
    if not query.strip():
        st.warning("Add a travel request to get started.")
    else:
        st.session_state.query = query
        try:
            with st.status("Planning your trip", expanded=True) as status:
                st.write("Screening the request and choosing specialists...")
                st.write("Running flight, hotel and weather agents in parallel...")
                result = app.invoke(
                    {
                        "messages": [HumanMessage(content=query)],
                        "user_id": user_id,
                        "user_query": query,
                        "flight_results": "",
                        "hotel_results": "",
                        "weather_results": "",
                        "budget_results": "",
                        "itinerary": "",
                        "final_response": "",
                        "llm_calls": 0,
                    },
                    config=config,
                )
                status.update(label="Draft ready for review", state="complete",
                              expanded=False)

            st.session_state.latest_result = result
            st.session_state.waiting_for_approval = "__interrupt__" in result

        except Exception as e:
            print(f"[frontend] planning failed: {e!r}")
            st.error(explain_failure(e))
            st.stop()


result = st.session_state.get("latest_result")


# --------------------------------------------------------------- empty state

if not result:
    st.markdown('<div class="sectionlabel">What the agents do</div>',
                unsafe_allow_html=True)
    st.markdown(empty_state(), unsafe_allow_html=True)


# ------------------------------------------------------------ guardrail path

if result and not result.get("selected_agents") and "__interrupt__" not in result:
    st.markdown('<div class="sectionlabel">Request declined</div>',
                unsafe_allow_html=True)
    st.error(
        result.get("final_response")
        or result.get("supervisor_reasoning")
        or "This request was turned away by the input guardrail."
    )
    st.caption("This planner only handles travel: trips, flights, hotels, "
               "itineraries and destination weather.")
    st.session_state.waiting_for_approval = False
    st.stop()


# ------------------------------------------------------------- results panel

if result:
    selected = result.get("selected_agents") or []
    has_final = bool(result.get("final_response"))
    stage = "final" if has_final else "approval"

    constraints = result.get("trip_constraints") or {}
    weather_text = result.get("weather_results", "")

    hero = route_hero(constraints, weather_text)
    if hero:
        st.markdown(hero, unsafe_allow_html=True)

    st.markdown(pipeline_strip(selected, stage), unsafe_allow_html=True)
    st.markdown(metarow(result), unsafe_allow_html=True)

    with st.expander("Why these specialists were chosen"):
        st.write(result.get("supervisor_reasoning", ""))

    st.markdown('<div class="sectionlabel">Research</div>', unsafe_allow_html=True)

    tab_f, tab_h, tab_w, tab_b = st.tabs(
        ["Flights", "Hotels", "Weather", "Budget"]
    )
    with tab_f:
        st.markdown(panel_head(ICON_PLANE, "Flights",
                               "live schedule + web routes"),
                    unsafe_allow_html=True)
        show(result.get("flight_results", ""))
    with tab_h:
        st.markdown(panel_head(ICON_BED, "Hotels", "sourced listings only"),
                    unsafe_allow_html=True)
        show(result.get("hotel_results", ""))
    with tab_w:
        st.markdown(panel_head(weather_icon(weather_text), "Weather",
                               "current conditions + forecast"),
                    unsafe_allow_html=True)
        show(result.get("weather_results", ""))
    with tab_b:
        st.markdown(panel_head(ICON_WALLET, "Budget",
                               "costed from sourced figures"),
                    unsafe_allow_html=True)
        show(result.get("budget_results", ""))

    st.markdown('<div class="sectionlabel">Draft itinerary</div>',
                unsafe_allow_html=True)
    st.markdown(panel_head(ICON_CALENDAR, "Draft plan", "awaiting your review"),
                unsafe_allow_html=True)

    if "__interrupt__" in result:
        draft = result["__interrupt__"][0].value.get("draft_itinerary", "")
    else:
        draft = result.get("itinerary", "")
    show(draft, "No draft itinerary was produced.")


# ------------------------------------------------------------ human approval

if st.session_state.get("waiting_for_approval"):
    st.divider()
    st.markdown('<div class="sectionlabel">Your review</div>',
                unsafe_allow_html=True)
    st.caption("Nothing is finalised until you sign off. Ask for changes and "
               "the plan is rebuilt from the same sourced research.")

    decision = st.radio(
        "How does this draft look?",
        ["Approve it", "Revise it"],
        horizontal=True,
        label_visibility="collapsed",
    )
    feedback = st.text_area(
        "What should change?",
        placeholder="Raise the budget to 1.5 lakh and add a day in Varanasi.",
        disabled=decision == "Approve it",
        height=90,
    )

    if st.button("Send review", type="primary"):
        try:
            with st.spinner("Rebuilding the plan from your feedback..."):
                final_result = app.invoke(
                    Command(resume={
                        "approved": decision == "Approve it",
                        "feedback": feedback,
                    }),
                    config=config,
                )
            st.session_state.latest_result = final_result
            st.session_state.waiting_for_approval = False
            st.rerun()

        except Exception as e:
            print(f"[frontend] review failed: {e!r}")
            st.error(explain_failure(e))
            st.caption("Your draft is still saved. Send the review again once "
                       "the issue clears.")


# --------------------------------------------------------- final + downloads

final_result = st.session_state.get("latest_result")

if final_result and final_result.get("final_response"):
    st.divider()
    st.markdown('<div class="sectionlabel">Final plan</div>',
                unsafe_allow_html=True)
    st.markdown(panel_head(ICON_CALENDAR, "Your travel plan", "approved"),
                unsafe_allow_html=True)
    st.markdown(md(final_result["final_response"]))

    saved_query = st.session_state.get("query", "")
    doc_md = build_markdown(final_result, saved_query)
    doc_html = build_html(doc_md)
    name = f"travel-plan-{slug(saved_query)}"

    st.markdown('<div class="sectionlabel">Take it with you</div>',
                unsafe_allow_html=True)
    st.caption("The download includes the final plan plus the flight, hotel, "
               "weather and budget research behind it, with every source link "
               "intact.")

    try:
        doc_pdf = build_pdf(doc_md)
    except Exception as e:
        print(f"[frontend] pdf build failed: {e!r}")
        doc_pdf = None

    d1, d2, d3 = st.columns(3)
    with d1:
        if doc_pdf:
            st.download_button(
                "PDF (.pdf)",
                data=doc_pdf,
                file_name=f"{name}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
        else:
            st.button("PDF unavailable", disabled=True,
                      use_container_width=True,
                      help="reportlab is not installed. Run: pip install reportlab")
    with d2:
        st.download_button(
            "Markdown (.md)",
            data=doc_md,
            file_name=f"{name}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with d3:
        st.download_button(
            "Printable (.html)",
            data=doc_html,
            file_name=f"{name}.html",
            mime="text/html",
            use_container_width=True,
        )