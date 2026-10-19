"""Download F1 race results (Jolpica/Ergast API) and race-time weather (Open-Meteo).

Writes:
  data/results.csv  - one row per driver per Grand Prix, 2018 -> latest completed 2026 race
  data/races.csv    - one row per race with circuit, date and weather flags
  data/schedule_2026.csv
"""

import json
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
SEASONS = range(2018, 2027)
JOLPICA = "https://api.jolpi.ca/ergast/f1"

# Races widely remembered as wet/mixed for a meaningful part of the GP. Used to
# back up the reanalysis weather data, which can miss short local showers.
KNOWN_WET = {
    (2018, "hockenheimring"), (2019, "hockenheimring"), (2020, "istanbul"),
    (2021, "imola"), (2021, "spa"), (2021, "sochi"), (2021, "hungaroring"),
    (2022, "monaco"), (2022, "marina_bay"), (2022, "suzuka"),
    (2023, "monaco"), (2023, "zandvoort"), (2024, "villeneuve"),
    (2024, "interlagos"), (2024, "silverstone"), (2025, "silverstone"),
    (2025, "spa"), (2025, "zandvoort"),
}
# Races the reanalysis flags as rainy but which were run in dry conditions.
KNOWN_DRY = {
    (2018, "catalunya"), (2019, "rodriguez"), (2022, "hungaroring"),
    (2022, "zandvoort"), (2024, "catalunya"), (2025, "suzuka"),
}


def get_json(url, tries=5):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except Exception as e:  # rate limit or transient error
            wait = 2 ** attempt
            print(f"  retry {attempt + 1} for {url} ({e}); sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"failed: {url}")


def fetch_results():
    rows = []
    for season in SEASONS:
        offset = 0
        while True:
            d = get_json(f"{JOLPICA}/{season}/results.json?limit=100&offset={offset}")
            races = d["MRData"]["RaceTable"]["Races"]
            total = int(d["MRData"]["total"])
            for race in races:
                loc = race["Circuit"]["Location"]
                for res in race["Results"]:
                    rows.append({
                        "season": season,
                        "round": int(race["round"]),
                        "race": race["raceName"],
                        "circuit": race["Circuit"]["circuitId"],
                        "lat": float(loc["lat"]),
                        "lon": float(loc["long"]),
                        "date": race["date"],
                        "time": race.get("time", "14:00:00Z"),
                        "driver": res["Driver"]["driverId"],
                        "driver_name": f'{res["Driver"]["givenName"]} {res["Driver"]["familyName"]}',
                        "code": res["Driver"].get("code", ""),
                        "team": res["Constructor"]["constructorId"],
                        "team_name": res["Constructor"]["name"],
                        "grid": int(res["grid"]),
                        "position": int(res["position"]),
                        "points": float(res["points"]),
                        "status": res["status"],
                    })
            offset += 100
            time.sleep(0.3)
            if offset >= total:
                break
        print(f"{season}: {sum(r['season'] == season for r in rows)} result rows")
    return pd.DataFrame(rows)


def race_weather(lat, lon, date, time_utc):
    """Precipitation and temperature over the ~2h race window from ERA5 reanalysis."""
    start = datetime.fromisoformat(f"{date}T{time_utc.rstrip('Z')}")
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}&start_date={date}&end_date={date}"
        "&hourly=precipitation,temperature_2m&timezone=UTC"
    )
    h = get_json(url)["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])
    win = df[(df["time"] >= start - timedelta(hours=1)) & (df["time"] <= start + timedelta(hours=2))]
    return win["precipitation"].fillna(0).sum(), win["temperature_2m"].mean()


def main():
    results = fetch_results()
    results.to_csv(DATA / "results.csv", index=False)

    races = results.drop_duplicates(["season", "round"])[
        ["season", "round", "race", "circuit", "lat", "lon", "date", "time"]
    ].reset_index(drop=True)
    precip, temp = [], []
    for r in races.itertuples():
        p, t = race_weather(r.lat, r.lon, r.date, r.time)
        precip.append(p)
        temp.append(t)
        time.sleep(0.15)
    races["race_precip_mm"] = precip
    races["race_temp_c"] = temp
    known = races.apply(lambda r: (r.season, r.circuit) in KNOWN_WET, axis=1)
    dry = races.apply(lambda r: (r.season, r.circuit) in KNOWN_DRY, axis=1)
    races["wet"] = (((races["race_precip_mm"] >= 1.0) & ~dry) | known).astype(int)
    races.to_csv(DATA / "races.csv", index=False)
    print(f"{len(races)} races, {races['wet'].sum()} flagged wet")

    sched = get_json(f"{JOLPICA}/2026.json?limit=30")["MRData"]["RaceTable"]["Races"]
    pd.DataFrame([{
        "round": int(s["round"]), "race": s["raceName"], "circuit": s["Circuit"]["circuitId"],
        "date": s["date"], "time": s.get("time", ""),
    } for s in sched]).to_csv(DATA / "schedule_2026.csv", index=False)


if __name__ == "__main__":
    main()
