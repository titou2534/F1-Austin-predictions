"""Predict the 2026 United States GP (COTA, Austin) finishing order.

Features (all computed only from races *before* the one being predicted):
  - current-season form: driver & team average finish this season, last-5 form, points
  - Austin history: driver's average finish at COTA and how much better/worse than
    their form at the time they ran there ("track affinity")
  - wet-weather skill: driver's finish vs. teammate in wet races minus in dry races
  - reliability (DNF rate), experience, race-day weather (wet flag, temperature)

Usage:
  python3 model.py                 # pre-qualifying prediction
  python3 model.py --grid grid.csv # after qualifying: csv with columns code,grid
"""
import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

DATA = Path(__file__).parent / "data"
TARGET_CIRCUIT = "americas"
TARGET_SEASON, TARGET_ROUND = 2026, 18
COTA = (30.1328, -97.6411)
FINISHED = r"^(Finished|\+\d+ Laps?|Lapped)$"

BASE_FEATURES = [
    "drv_season_pos", "drv_last5_pos", "drv_last5_pts", "team_season_pos", "team_last3_pos",
    "drv_circuit_avg", "drv_circuit_n", "drv_circuit_affinity", "wet_skill", "wet",
    "wet_x_skill", "race_temp_c", "dnf_rate10", "experience",
]


def load():
    res = pd.read_csv(DATA / "results.csv")
    races = pd.read_csv(DATA / "races.csv")[["season", "round", "wet", "race_temp_c"]]
    df = res.merge(races, on=["season", "round"])
    df["dnf"] = (~df["status"].str.match(FINISHED)).astype(int)
    return df


def add_future_race(df, wet, temp):
    """Append Austin 2026 rows for every driver in the most recent race's entry list."""
    last = df[(df.season == TARGET_SEASON)]
    last = last[last["round"] == last["round"].max()]
    fut = last[["driver", "driver_name", "code", "team", "team_name"]].copy()
    fut["season"], fut["round"], fut["circuit"] = TARGET_SEASON, TARGET_ROUND, TARGET_CIRCUIT
    fut["race"] = "United States Grand Prix"
    fut["wet"], fut["race_temp_c"], fut["future"] = wet, temp, 1
    return pd.concat([df.assign(future=0), fut], ignore_index=True)


def shifted_expanding_mean(df, keys, col):
    g = df.groupby(keys)[col]
    return g.transform(lambda s: s.shift(1).expanding().mean())


def build_features(df):
    df = df.sort_values(["season", "round", "position"]).reset_index(drop=True)
    df["race_id"] = df["season"] * 100 + df["round"]

    # Driver form
    g = df.groupby("driver")
    df["experience"] = g.cumcount()
    df["drv_last5_pos"] = g["position"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df["drv_last5_pts"] = g["points"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df["dnf_rate10"] = g["dnf"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    df["drv_season_pos"] = shifted_expanding_mean(df, ["driver", "season"], "position")
    df["drv_season_pos"] = df["drv_season_pos"].fillna(df["drv_last5_pos"])

    # Team (car) form: average of the team's cars per race, then shifted over races
    team_race = df.groupby(["team", "season", "race_id"], as_index=False)["position"].mean()
    team_race = team_race.sort_values("race_id")
    tg = team_race.groupby("team")["position"]
    team_race["team_last3_pos"] = tg.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    team_race["team_season_pos"] = team_race.groupby(["team", "season"])["position"].transform(
        lambda s: s.shift(1).expanding().mean())
    team_race["team_season_pos"] = team_race["team_season_pos"].fillna(team_race["team_last3_pos"])
    df = df.merge(team_race[["team", "race_id", "team_last3_pos", "team_season_pos"]],
                  on=["team", "race_id"], how="left")

    # Circuit history and affinity (finish relative to form going into that race)
    df["resid_vs_form"] = df["position"] - df["drv_last5_pos"]
    df = df.sort_values(["race_id", "position"]).reset_index(drop=True)
    cg = df.groupby(["driver", "circuit"])
    df["drv_circuit_n"] = cg.cumcount()
    df["drv_circuit_avg"] = shifted_expanding_mean(df, ["driver", "circuit"], "position")
    df["drv_circuit_affinity"] = shifted_expanding_mean(df, ["driver", "circuit"], "resid_vs_form")

    # Wet skill: finish vs. teammate(s), wet minus dry, shrunk toward 0 for few wet races
    team_sum = df.groupby(["race_id", "team"])["position"].transform("sum")
    team_n = df.groupby(["race_id", "team"])["position"].transform("count")
    df["vs_mate"] = df["position"] - (team_sum - df["position"]) / (team_n - 1).replace(0, np.nan)
    df["vs_mate"] = df["vs_mate"].clip(-10, 10)
    for cond, name in [(1, "wet"), (0, "dry")]:
        v = df["vs_mate"].where(df["wet"] == cond)
        g = df.assign(_v=v.fillna(0), _n=v.notna().astype(int)).groupby("driver")
        df[f"{name}_sum"] = g["_v"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
        df[f"{name}_n"] = g["_n"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    dry_mean = df["dry_sum"] / df["dry_n"].clip(lower=1)
    wet_shrunk = (df["wet_sum"] + 3 * dry_mean) / (df["wet_n"] + 3)  # prior: wet = dry form
    df["wet_skill"] = wet_shrunk - dry_mean  # negative = gains on teammate in the wet
    df["wet_x_skill"] = df["wet"] * df["wet_skill"]

    df["experience"] = df["experience"].fillna(0)
    return df


def fit(train, features):
    w = np.where(train["season"] >= 2026, 2.0, 1.0)  # new 2026 regs: weight this season more
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_depth=4,
                                      min_samples_leaf=30, l2_regularization=1.0, random_state=0)
    m.fit(train[features], train["position"], sample_weight=w)
    return m


def evaluate(df, features):
    print("\n Back-testing (train on earlier data, predict held-out races) ")
    rows = []
    for label, test_mask in [
        ("2025 season", df.season == 2025),
        ("2026 rounds 9-17", (df.season == 2026) & (df["round"] >= 9)),
        ("Austin 2023-2025", (df.circuit == TARGET_CIRCUIT) & df.season.between(2023, 2025)),
    ]:
        test = df[test_mask & (df.future == 0)]
        first = test["race_id"].min()
        train = df[(df.race_id < first) & (df.season >= 2019) & (df.future == 0)]
        m = fit(train, features)
        test = test.assign(pred=m.predict(test[features]))
        test["pred_rank"] = test.groupby("race_id")["pred"].rank()
        base = test.groupby("race_id")["drv_season_pos"].rank()
        per_race = test.groupby("race_id").apply(lambda r: pd.Series({
            "rho": spearmanr(r["pred_rank"], r["position"])[0],
            "rho_base": spearmanr(base[r.index], r["position"])[0],
            "winner": int(r.loc[r["pred_rank"].idxmin(), "position"] == 1),
            "podium": len(set(r.nsmallest(3, "pred")["driver"]) & set(r[r.position <= 3]["driver"])),
        }), include_groups=False)
        rows.append({
            "test set": label, "races": len(per_race),
            "MAE (places)": round((test["pred_rank"] - test["position"]).abs().mean(), 2),
            "rank corr": round(per_race["rho"].mean(), 3),
            "baseline corr": round(per_race["rho_base"].mean(), 3),
            "winner hit %": round(100 * per_race["winner"].mean()),
            "podium overlap /3": round(per_race["podium"].mean(), 2),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print("(baseline = rank drivers by average finish so far this season)")


def austin_forecast():
    """Rain probability and temperature for the race window (Sun 25 Oct, 15:00 local)."""
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={COTA[0]}&longitude={COTA[1]}"
           "&hourly=precipitation_probability,temperature_2m&timezone=UTC&forecast_days=7")
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            h = pd.DataFrame(json.load(r)["hourly"])
        h["time"] = pd.to_datetime(h["time"])
        win = h[h["time"].between("2026-10-25 19:00", "2026-10-25 22:00")]
        if len(win):
            return win["precipitation_probability"].max() / 100, win["temperature_2m"].mean()
    except Exception as e:
        print(f"forecast unavailable ({e}); assuming dry, 27C")
    return 0.0, 27.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", help="CSV with columns code,grid once qualifying is done")
    ap.add_argument("--rain", type=float, help="override rain probability (0-1)")
    ap.add_argument("--sims", type=int, default=20000)
    args = ap.parse_args()

    p_rain, temp = austin_forecast()
    if args.rain is not None:
        p_rain = args.rain
    print(f"Austin race-day forecast: rain probability {p_rain:.0%}, {temp:.1f}C")

    raw = load()
    features = BASE_FEATURES + (["grid"] if args.grid else [])
    preds = {}
    for wet in (0, 1):
        df = build_features(add_future_race(raw, wet, temp))
        if args.grid:
            g = pd.read_csv(args.grid)
            df = df.merge(g.rename(columns={"grid": "grid_q"}), on="code", how="left")
            df.loc[df.future == 1, "grid"] = df["grid_q"]
            df["grid"] = df["grid"].replace(0, 22)  # pit-lane start
        if wet == 0:
            evaluate(df, features)
        train = df[(df.future == 0) & (df.season >= 2019)]
        model = fit(train, features)
        fut = df[df.future == 1].copy()
        fut["pred"] = model.predict(fut[features])
        preds[wet] = fut.set_index("driver")
        resid = train["position"] - model.predict(train[features])

    fut = preds[0].copy()
    fut["pred_dry"], fut["pred_wet"] = preds[0]["pred"], preds[1]["pred"]
    fut["pred"] = (1 - p_rain) * fut["pred_dry"] + p_rain * fut["pred_wet"]

    # Monte Carlo: add race-to-race noise (sized from model residuals) and re-rank
    rng = np.random.default_rng(42)
    sigma = resid.std() * 0.8
    wet_draw = rng.random(args.sims) < p_rain
    base = np.where(wet_draw[:, None], fut["pred_wet"].values, fut["pred_dry"].values)
    sims = base + rng.normal(0, sigma, (args.sims, len(fut)))
    ranks = sims.argsort(1).argsort(1) + 1
    fut["P(win)"] = (ranks == 1).mean(0)
    fut["P(podium)"] = (ranks <= 3).mean(0)
    fut["P(points)"] = (ranks <= 10).mean(0)
    fut["exp_finish"] = ranks.mean(0)

    out = fut.sort_values("exp_finish").reset_index()
    out.insert(0, "pred_pos", range(1, len(out) + 1))
    show = out[["pred_pos", "driver_name", "team_name", "exp_finish", "P(win)", "P(podium)", "P(points)",
                "drv_season_pos", "drv_circuit_avg", "drv_circuit_n", "wet_skill"]].copy()
    show.columns = ["Pos", "Driver", "Team", "Exp. finish", "P(win)", "P(podium)", "P(points)",
                    "2026 avg", "COTA avg", "COTA races", "Wet skill"]
    for c in ["P(win)", "P(podium)", "P(points)"]:
        show[c] = (show[c] * 100).round(1).astype(str) + "%"
    print("\n Predicted 2026 United States GP finishing order ")
    print(show.round(2).to_string(index=False))
    print("wet skill: places gained(-)/lost(+) vs. teammate in wet races relative to dry (shrunk).")
    show.to_csv(Path(__file__).parent / "austin_2026_prediction.csv", index=False)

    imp_model = model  # feature importance via permutation on the last fit
    from sklearn.inspection import permutation_importance
    recent = train[train.season >= 2025]
    pi = permutation_importance(imp_model, recent[features], recent["position"], n_repeats=5, random_state=0)
    print("\nFeature importance (permutation, 2025-26 races):")
    print(pd.Series(pi.importances_mean, index=features).sort_values(ascending=False).round(2).to_string())


if __name__ == "__main__":
    main()
