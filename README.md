# F1 Austin Predictor

Machine-learning model that predicts the finishing order of the 2026 United States Grand Prix (Circuit of the Americas, Austin).

## Features

Every feature uses only races *before* the one being predicted:

- **2026 season form**: driver and team average finish, plus last-5 form and points. 2026 races count double in training because of the new regulations.
- **Austin history**: driver's average finish at COTA, and whether they usually finish better or worse there than their form going in.
- **Wet-weather skill**: finish vs. teammate in wet races compared with dry races. Drivers with few wet races are pulled towards zero.
- Reliability (DNF rate), experience and race-day temperature.

The model is a gradient-boosted regressor (scikit-learn). It predicts a dry and a wet result for each driver, weights them by the forecast rain chance, and runs a Monte Carlo simulation for win, podium and points probabilities.

## Data

- Race results 2018–2026 from the [Jolpica / Ergast API](https://api.jolpi.ca/ergast/f1)
- Qualifying results (the starting grid) from [OpenF1](https://openf1.org)
- Historical race-time weather (ERA5) and the Austin forecast from [Open-Meteo](https://open-meteo.com), plus a hand-checked list of known wet and dry races

## Usage

```bash
pip install pandas numpy scikit-learn scipy
python3 fetch_data.py                 # refresh results + weather into data/
python3 model.py                      # pre-qualifying prediction
python3 fetch_data.py --grid          # after qualifying: latest grid into grid.csv
python3 model.py --grid grid.csv      # post-qualifying prediction
python3 model.py --rain 0.4           # override the rain probability
```

`grid.csv` is the qualifying order, so edit it by hand if a driver takes a grid penalty.

The output is printed to the terminal and saved to `austin_2026_prediction.csv`.
