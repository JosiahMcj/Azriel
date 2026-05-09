"""weather tool -- current conditions + 3-day forecast for a location.

Uses open-meteo (free, no key). Two endpoints:
- geocoding-api.open-meteo.com to resolve the city
- api.open-meteo.com for current weather + daily forecast

Usage:
  weather("Phoenix, AZ")
  weather("Jerusalem")
"""
import json
import urllib.parse
import urllib.request

GEO = "https://geocoding-api.open-meteo.com/v1/search"
WX = "https://api.open-meteo.com/v1/forecast"

CODE_DESC = {
    0: "clear", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "rime fog", 51: "light drizzle", 53: "drizzle",
    55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "rain showers", 81: "rain showers", 82: "violent rain showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm w/ hail", 99: "severe thunderstorm",
}


def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Azriel/0.7"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def weather(location: str) -> str:
    if not isinstance(location, str):
        return "ERROR: weather expects a string location."
    loc = location.strip()
    if not loc:
        return "ERROR: empty location."

    try:
        geo = _get(GEO + "?" + urllib.parse.urlencode({"name": loc, "count": 1}))
    except Exception as e:
        return f"ERROR: geocode failed ({type(e).__name__}: {e})"
    results = geo.get("results") or []
    if not results:
        return f"ERROR: location not found: {loc}"
    g = results[0]
    lat, lon = g["latitude"], g["longitude"]
    place = ", ".join([g.get(k) for k in ("name", "admin1", "country") if g.get(k)])

    try:
        wx = _get(WX + "?" + urllib.parse.urlencode({
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
            "precipitation_unit": "inch", "timezone": "auto", "forecast_days": 3,
        }))
    except Exception as e:
        return f"ERROR: forecast failed ({type(e).__name__}: {e})"

    cur = wx.get("current", {})
    code = cur.get("weather_code")
    cond = CODE_DESC.get(code, f"code {code}")
    lines = [
        f"{place}",
        f"now: {cur.get('temperature_2m')}°F (feels {cur.get('apparent_temperature')}°F), "
        f"{cond}, humidity {cur.get('relative_humidity_2m')}%, wind {cur.get('wind_speed_10m')} mph",
        "",
        "next 3 days:",
    ]
    daily = wx.get("daily", {})
    days = daily.get("time", [])
    for i, day in enumerate(days):
        c = daily["weather_code"][i]
        hi = daily["temperature_2m_max"][i]
        lo = daily["temperature_2m_min"][i]
        pp = daily["precipitation_probability_max"][i]
        lines.append(f" {day}: hi {hi}°F / lo {lo}°F, {CODE_DESC.get(c, str(c))}, precip {pp}%")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    print(weather(sys.argv[1] if len(sys.argv) > 1 else "Phoenix, AZ"))
