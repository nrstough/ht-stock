"""The status-quo production policy: a reasonably smart manager, not a straw man.

Par = trailing same-weekday sales average x a safety pad, with recency-biased
reactions to yesterday, a crude but directionally-correct holiday instinct,
and basic snow awareness. What it lacks is precision: it learns trailing
averages of *sales* (so stockouts hide demand), lags trends and seasons by
weeks, and guesses holidays and weather coarsely.
"""
import math

import numpy as np

from . import params


class StatusQuoManager:
    def __init__(self, item_key, item):
        self.key = item_key
        self.item = item
        self.pol = params.POLICY
        # trailing (sales, sold_out_flag) per weekday
        self.same_dow_hist = {d: [] for d in range(7)}
        self.yesterday = None  # (produced, sold)

    def decide(self, date, holiday, snow_tomorrow, is_snow_day, is_rain_day=False):
        pol, item = self.pol, self.item
        hist = self.same_dow_hist[date.weekday()][-pol["lookback_weeks"]:]
        if hist:
            par = float(np.mean([h[0] for h in hist]))
            # sales averages are censored by sellouts; a real manager notices
            # "sold out three Fridays running" and raises the par
            par *= 1 + pol["chronic_sellout_bump"] * sum(h[1] for h in hist)
        else:
            par = item["base"] * params.BASE_SCALE * item["dow"][date.weekday()]
        par *= pol["pad"]

        if self.yesterday is not None:
            produced, sold = self.yesterday
            if produced > 0 and sold >= produced:            # stocked out
                par *= pol["stockout_bump"]
            elif produced > 0 and (produced - sold) / produced > pol["waste_cut_threshold"]:
                par *= pol["waste_cut"]

        if holiday and pol["holiday_instinct"]:
            spec = params.HOLIDAYS[holiday]
            if spec != "closed":
                true_mult = spec.get(self.key, spec.get("__all__", 1.0))
                par *= math.sqrt(true_mult)   # right direction, imprecise

        if snow_tomorrow and self.key == "bread":
            par *= pol["snow_prep_bread_mult"]
        if is_snow_day:
            par *= pol["snow_day_mult"]
        if is_rain_day:
            par *= pol["rain_day_mult"]

        batch = item["batch"]
        if item.get("continuous"):
            produced = max(batch, math.ceil(par / batch) * batch)
        else:
            produced = max(batch, int(math.ceil(par / batch)) * batch)
        return float(produced)

    def observe(self, date, produced, sold):
        self.same_dow_hist[date.weekday()].append((sold, int(sold >= produced > 0)))
        self.yesterday = (produced, sold)
