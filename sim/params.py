"""Every knob of the synthetic store in one place.

All demand effects are multiplicative on a per-item base rate. Tune anything
here and re-run generate.py to rebuild the dataset. Values are plausible
guesses for a mid-size mid-Atlantic grocery store's prepared-foods section --
they are meant to be corrected by someone who works in one.
"""

START = "2023-01-01"
END = "2025-12-31"          # 3 years
SEED = 42

# ---- store-level effects (apply to every item) ----
STORE = dict(
    trend_per_year=1.02,        # slow traffic growth
    traffic_noise_sigma=0.06,   # shared day-level noise: items co-move
    payday_mult=1.05,           # days 1-3 and 15-17 of the month
    event_prob=0.01,            # unpredictable local-event days
    event_mult=1.35,
    weather=dict(               # traffic multiplier by day's weather
        sunny=1.05, cloudy=1.00, rain=0.90, snow=0.55,
        snow_prep=1.25,         # day BEFORE a snow day (pantry loading)
    ),
)

# ---- weather generator (NC piedmont flavored) ----
WEATHER = dict(
    tmax_mean=69.5, tmax_amp=19.5, tmax_peak_doy=201,   # ~50F Jan, ~89F Jul
    ar1=0.70, shock_sigma=5.5,                          # day-to-day persistence
    rain_prob=0.30,
    snow_below_f=38.0,          # precip on a cold day falls as snow
    cloudy_prob=0.35,           # of the remaining dry days
)

# global scale on every item's base rate: tune total store volume in one place
# (0.72 lands weekly retail waste near the ~$2k/week order-of-magnitude estimate
# for a mid-size store's prepared foods)
BASE_SCALE = 0.72

# ---- items ----
# dow multipliers are Mon..Sun (python date.weekday() order).
# seas_amp/peak: annual cosine, 1 + amp*cos(2*pi*(doy-peak)/365).
# temp_sens: demand multiplier per +10F above 65F (negative = cold-weather item).
# sigma: per-item lognormal demand noise. batch: production rounding unit.
ITEMS = {
    "pizza-whole": dict(
        name="Whole Pizza", dept="Pizza", price=7.99, cost=2.60,
        base=25, dow=[0.80, 0.75, 0.85, 0.95, 1.60, 1.45, 1.00],
        seas_amp=0.06, seas_peak=30, temp_sens=0.0, sigma=0.16,
        trend_per_year=1.0, batch=1,
    ),
    "pizza-slice": dict(
        name="Pizza Slice", dept="Pizza", price=3.49, cost=0.90,
        base=40, dow=[1.05, 1.05, 1.10, 1.10, 1.15, 0.90, 0.70],
        seas_amp=0.04, seas_peak=30, temp_sens=0.0, sigma=0.15,
        trend_per_year=1.0, batch=8,
    ),
    "doughnut": dict(
        name="Doughnut", dept="Bakery", price=1.19, cost=0.25,
        base=110, dow=[0.85, 0.80, 0.85, 0.90, 1.05, 1.40, 1.35],
        seas_amp=0.05, seas_peak=350, temp_sens=-0.01, sigma=0.13,
        trend_per_year=1.0, batch=12,
    ),
    "cake": dict(
        name="Cake", dept="Bakery", price=12.99, cost=4.50,
        base=6, dow=[0.75, 0.70, 0.80, 0.90, 1.15, 1.55, 1.10],
        # sigma modest: real cakes carry multi-day shelf life that buffers
        # daily noise; the sim treats everything as day-fresh (see data README)
        seas_amp=0.06, seas_peak=160, temp_sens=0.0, sigma=0.25,
        trend_per_year=1.0, batch=1,
        window_mult=dict(start=(5, 10), end=(6, 15), mult=1.45),  # graduations
    ),
    "bread": dict(
        name="Bread Loaf", dept="Bakery", price=3.99, cost=1.10,
        base=35, dow=[0.95, 0.90, 0.95, 1.00, 1.10, 1.15, 0.95],
        seas_amp=0.08, seas_peak=340, temp_sens=0.0, sigma=0.14,
        trend_per_year=1.0, batch=1, snow_prep_mult=2.5,
    ),
    "rotisserie": dict(
        name="Rotisserie Chicken", dept="Hot Foods", price=7.99, cost=3.20,
        base=45, dow=[1.05, 0.95, 1.00, 1.05, 1.10, 1.00, 1.30],
        seas_amp=0.08, seas_peak=15, temp_sens=-0.025, sigma=0.13,
        trend_per_year=1.0, batch=4,
    ),
    "hotbar-lb": dict(
        name="Hot Bar (per lb)", dept="Hot Foods", price=9.99, cost=3.50,
        base=60, dow=[1.10, 1.10, 1.15, 1.10, 1.05, 0.80, 0.75],
        seas_amp=0.15, seas_peak=15, temp_sens=-0.045, sigma=0.12,
        trend_per_year=1.0, batch=5, continuous=True,
    ),
    "sushi": dict(
        name="Sushi Roll", dept="Fresh Foods", price=8.99, cost=3.80,
        base=30, dow=[0.95, 0.95, 1.00, 1.05, 1.35, 1.10, 0.85],
        seas_amp=0.10, seas_peak=196, temp_sens=0.03, sigma=0.16,
        trend_per_year=1.12, batch=1,   # sushi is a growth category
    ),
    "sub": dict(
        name="Sub / Sandwich", dept="Fresh Foods", price=5.99, cost=1.80,
        base=25, dow=[1.10, 1.05, 1.10, 1.10, 1.20, 0.95, 0.75],
        seas_amp=0.12, seas_peak=196, temp_sens=0.03, sigma=0.15,
        trend_per_year=1.0, batch=1,
    ),
}

# default day-before-snow multiplier for items without their own
SNOW_PREP_DEFAULT = 1.10

# ---- holiday demand multipliers, per item (1.0 if absent) ----
HOLIDAYS = {
    "new_years_day":   {"hotbar-lb": 1.20, "doughnut": 1.10, "sub": 0.90, "sushi": 0.85},
    "super_bowl":      {"pizza-whole": 3.00, "pizza-slice": 1.80, "sub": 1.90,
                        "hotbar-lb": 1.30, "doughnut": 0.90, "rotisserie": 1.10},
    "valentines":      {"cake": 1.80, "sushi": 1.60, "doughnut": 1.10},
    "easter":          {"cake": 1.50, "bread": 1.40, "rotisserie": 1.30, "doughnut": 1.20},
    "mothers_day":     {"cake": 2.60, "sushi": 1.30, "doughnut": 1.20},
    "memorial_day":    {"sub": 1.50, "rotisserie": 1.30, "hotbar-lb": 0.90, "pizza-whole": 1.10},
    "july4":           {"sub": 1.50, "rotisserie": 1.50, "bread": 1.20, "hotbar-lb": 0.85},
    "labor_day":       {"sub": 1.40, "rotisserie": 1.25, "hotbar-lb": 0.90},
    "halloween":       {"doughnut": 1.50, "pizza-whole": 1.60, "cake": 1.20},
    "thanksgiving_eve": {"bread": 2.20, "cake": 1.60, "doughnut": 1.30, "rotisserie": 0.90,
                         "hotbar-lb": 0.80, "pizza-whole": 1.30},
    "thanksgiving":    {"__all__": 0.45},   # reduced hours, most people home cooking
    "christmas_eve":   {"bread": 1.80, "cake": 1.70, "doughnut": 1.20, "hotbar-lb": 0.85},
    "christmas":       "closed",
    "new_years_eve":   {"sushi": 2.00, "cake": 1.30, "pizza-whole": 1.40},
}

# hardcoded floating holidays that aren't a simple nth-weekday rule
EASTER = {2023: (4, 9), 2024: (3, 31), 2025: (4, 20)}

# ---- status-quo production policy (the simulated store today) ----
POLICY = dict(
    pad=1.15,                 # produce over the trailing average; note the
                              # average is of censored SALES, so the effective
                              # pad over true demand is smaller than this
    lookback_weeks=4,         # same-weekday trailing window
    stockout_bump=1.10,       # reaction to yesterday's stockout
    chronic_sellout_bump=0.06,  # +6% par per sellout in the trailing window:
                                # corrects for sales averages censored by sellouts
    waste_cut=0.94,           # reaction to yesterday's heavy waste
    waste_cut_threshold=0.30, # "heavy" = >30% of production tossed
    # the manager is not naive: crude holiday instinct (sqrt of the true
    # multiplier -- right direction, imprecise) and basic snow awareness
    holiday_instinct=True,
    snow_prep_bread_mult=1.30,
    snow_day_mult=0.70,
    rain_day_mult=0.95,       # managers do trim a little on rainy mornings
)
