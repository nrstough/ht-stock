"""Synthetic daily weather: seasonal temperature with AR(1) persistence,
seasonally flat rain, snow when precipitation meets a cold day."""
import numpy as np

from . import params


def simulate(dates, rng):
    w = params.WEATHER
    n = len(dates)
    doy = np.array([d.timetuple().tm_yday for d in dates], dtype=float)
    seasonal = w["tmax_mean"] + w["tmax_amp"] * np.cos(
        2 * np.pi * (doy - w["tmax_peak_doy"]) / 365.0
    )
    dev = np.zeros(n)
    for i in range(1, n):
        dev[i] = w["ar1"] * dev[i - 1] + rng.normal(0, w["shock_sigma"])
    tmax = np.round(seasonal + dev, 1)

    precip = rng.random(n) < w["rain_prob"]
    kind = np.empty(n, dtype=object)
    for i in range(n):
        if precip[i]:
            kind[i] = "snow" if tmax[i] < w["snow_below_f"] else "rain"
        else:
            kind[i] = "cloudy" if rng.random() < w["cloudy_prob"] else "sunny"

    # "forecast" of tomorrow's snow, known a day ahead (as it is in reality)
    snow_tomorrow = np.zeros(n, dtype=bool)
    snow_tomorrow[:-1] = kind[1:] == "snow"
    return tmax, kind, snow_tomorrow
