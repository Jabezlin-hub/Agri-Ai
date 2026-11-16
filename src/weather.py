"""Growing-season climate for a place, from the free Open-Meteo APIs (no key needed).

Uses the most recent completed season in Open-Meteo's historical weather archive:
mean temperature, mean relative humidity and total rainfall over the chosen months.
"""
from __future__ import annotations

import calendar
import json
import urllib.parse
import urllib.request
from datetime import date, timedelta

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HEADERS = {"User-Agent": "AgriAI-student-project"}


def _get(url: str, params: dict) -> dict:
    req = urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}", headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.load(resp)


def find_places(name: str, count: int = 5) -> list[dict]:
    """Places matching `name`, each with a readable label and coordinates."""
    data = _get(GEOCODE_URL, {"name": name, "count": count, "language": "en", "format": "json"})
    places = []
    for r in data.get("results", []):
        parts = [r.get("name"), r.get("admin1"), r.get("country")]
        places.append({"label": ", ".join(p for p in parts if p),
                       "lat": r["latitude"], "lon": r["longitude"]})
    return places


def last_season(start_month: int, months: int, today: date | None = None) -> tuple[date, date]:
    """Dates of the most recent completed season starting in `start_month`."""
    today = today or date.today()
    latest = today - timedelta(days=7)          # the archive lags a few days behind
    year = latest.year
    while True:
        start = date(year, start_month, 1)
        end_month = (start_month - 1 + months - 1) % 12 + 1
        end_year = year + (start_month - 1 + months - 1) // 12
        end = date(end_year, end_month, calendar.monthrange(end_year, end_month)[1])
        if end <= latest:
            return start, end
        year -= 1


def season_climate(lat: float, lon: float, start_month: int, months: int) -> dict:
    """Mean temperature (°C), mean humidity (%) and total rainfall (mm) for the season."""
    start, end = last_season(start_month, months)
    data = _get(ARCHIVE_URL, {
        "latitude": lat, "longitude": lon,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "daily": "temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean",
        "timezone": "auto",
    })
    d = data["daily"]
    temps = [v for v in d["temperature_2m_mean"] if v is not None]
    hums = [v for v in d["relative_humidity_2m_mean"] if v is not None]
    rain = [v for v in d["precipitation_sum"] if v is not None]
    if not temps or not hums:
        raise ValueError("No weather data returned for this place and period.")
    return {
        "temperature": round(sum(temps) / len(temps), 1),
        "humidity": round(sum(hums) / len(hums), 1),
        "rainfall": round(sum(rain), 1),
        "start": start, "end": end, "days": len(d["time"]),
    }
