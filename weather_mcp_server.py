# pip install mcp requests

from mcp.server.fastmcp import FastMCP
import requests
import os

from dotenv import load_dotenv
load_dotenv()

mcp = FastMCP("Weather Server")
 

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

@mcp.tool()
def get_current_weather(city: str):

    response = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={
            "q": city,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"
        }
    )

    data = response.json()

    if response.status_code != 200 or "main" not in data:
        return {"error": data.get("message", "city not found"), "city": city}

    return {
        "city": data["name"],
        "temperature_c": data["main"]["temp"],
        "feels_like_c": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "condition": data["weather"][0]["description"],
        "wind_speed": data["wind"]["speed"]
    }



@mcp.tool()
def get_forecast(city: str):

    url = (
        "https://api.openweathermap.org/data/2.5/forecast"
    )

    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }

    response = requests.get(
        url,
        params=params
    )

    data = response.json()

    if response.status_code != 200 or "list" not in data:
        return {"error": data.get("message", "forecast unavailable"), "city": city}


    # Return first 5 forecast entries

    forecast = [
        {
            "datetime": item["dt_txt"],
            "temperature": item["main"]["temp"],
            "weather": item["weather"][0]["description"],
        }
        for item in data["list"][:5]
    ]

    return {
        "city": city,
        "forecast": forecast
    }




if __name__ == "__main__":
    mcp.run()