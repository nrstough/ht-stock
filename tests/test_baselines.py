"""The benchmark forecasters in model/baselines.py, on frames small enough to check by hand."""
import numpy as np
import pandas as pd

from model import baselines


def test_the_naive_forecast_never_hands_back_the_sales_it_is_forecasting():
    """An item's first day has no history, so its naive forecast is NaN rather than that
    day's own sales; later days use yesterday until a same-weekday history exists."""
    sold = [40, 11, 12, 13, 14, 15, 16, 17, 18]
    df = pd.DataFrame({"item": "bread", "sold": np.array(sold, dtype=float),
                       "date": pd.date_range("2026-01-05", periods=len(sold), freq="D")})
    b = {"item": df["item"].to_numpy(), "date": df["date"].to_numpy()}

    out = baselines.naive_forecast(df, b)

    assert np.isnan(out[0])
    assert out[1] == 40 and out[6] == 15        # no same weekday yet: the previous day
    assert out[7] == 40 and out[8] == 11        # one week back
