"""The decision layer: turn a demand distribution into a production quantity.

Newsvendor logic: produce at the quantile where the cost of one-too-many
(unit cost, minus salvage) balances the cost of one-too-few (lost margin).

    critical fractile  q* = underage / (underage + overage)
                          = (price - cost) / (price - salvage)

High-margin cheap items (doughnuts) justify padding high; low-margin
expensive items run lean. The forecaster's job is to hand this rule an
honest distribution.
"""
import math

import numpy as np


def critical_fractile(price, cost, salvage=0.0):
    underage = price - cost
    overage = cost - salvage
    return underage / (underage + overage)


def quantity(quantiles, taus, q_star, batch=1, continuous=False):
    """Interpolate the q* quantile from a discrete quantile grid (units)."""
    q = float(np.interp(q_star, taus, quantiles))
    q = max(q, 0.0)
    if continuous:
        return max(round(q / batch) * batch, batch)
    return float(max(int(round(q / batch)) * batch, batch))
