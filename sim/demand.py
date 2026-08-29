"""True (latent) demand model: multiplicative layers on a per-item base rate,
then noise. The generator knows true demand; the store only ever sees sales."""
import datetime as dt

import numpy as np

from . import params


def latent_mean(item_key, item, date, day_index, weather_kind, tmax,
                snow_tomorrow, holiday, payday, event_mult):
    """Expected demand before noise for one item on one day."""
    store = params.STORE
    doy = date.timetuple().tm_yday
    m = item["base"] * params.BASE_SCALE
    m *= item["dow"][date.weekday()]
    m *= 1 + item["seas_amp"] * np.cos(2 * np.pi * (doy - item["seas_peak"]) / 365.0)

    wm = item.get("window_mult")
    if wm:
        start = dt.date(date.year, *wm["start"])
        end = dt.date(date.year, *wm["end"])
        if start <= date <= end:
            m *= wm["mult"]

    # weather: shared traffic effect + per-item temperature preference
    m *= store["weather"][weather_kind]
    m *= 1 + item["temp_sens"] * (tmax - 65.0) / 10.0
    if snow_tomorrow:
        m *= store["weather"]["snow_prep"]
        m *= item.get("snow_prep_mult", params.SNOW_PREP_DEFAULT)

    if holiday:
        spec = params.HOLIDAYS[holiday]
        if spec != "closed":
            m *= spec.get(item_key, spec.get("__all__", 1.0))

    if payday:
        m *= store["payday_mult"]
    m *= store["trend_per_year"] ** (day_index / 365.0)
    m *= item.get("trend_per_year", 1.0) ** (day_index / 365.0)
    m *= event_mult
    return max(m, 0.0)


def realize(item, mean, traffic_noise, rng):
    """Draw actual demand: shared traffic noise + per-item lognormal noise.

    Counts use stochastic rounding rather than a Poisson draw: day-level
    variance in a store is dominated by common factors (weather, traffic,
    events) which are modeled explicitly, and a pure Poisson layer would
    unrealistically swamp low-volume items like cakes with counting noise.
    """
    latent = mean * traffic_noise
    latent *= np.exp(rng.normal(0, item["sigma"]) - item["sigma"] ** 2 / 2)
    latent = max(latent, 0.0)
    if item.get("continuous"):
        return round(latent, 1)
    floor = np.floor(latent)
    return float(floor + (rng.random() < latent - floor))
