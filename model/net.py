"""Global quantile demand network.

One model for every item: a GRU encodes each item's trailing 28 days
(sales, sellouts, weather), a learned item embedding carries item identity,
and tomorrow's known covariates (calendar, holidays, weather forecast) join
at the head. The output is 9 demand quantiles, monotone by construction
(first quantile + cumulative softplus increments), trained with pinball loss.

Censoring: on sellout days the observed sales are only a lower bound of
demand, so the loss keeps only the under-prediction penalty there.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DemandNet(nn.Module):
    def __init__(self, n_items, ctx_dim, cov_dim, n_quantiles,
                 emb_dim=16, hidden=64):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.gru = nn.GRU(ctx_dim, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + emb_dim + cov_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_quantiles),
        )

    def forward(self, iidx, ctx, cov):
        _, h = self.gru(ctx)
        z = torch.cat([h[-1], self.item_emb(iidx), cov], dim=1)
        raw = self.head(z)
        q0 = raw[:, :1]
        inc = F.softplus(raw[:, 1:])
        return torch.cat([q0, q0 + torch.cumsum(inc, dim=1)], dim=1)


def censored_pinball(pred, y, taus, censored):
    """pred [B,Q], y [B], taus [Q], censored [B] in {0,1}."""
    u = y.unsqueeze(1) - pred                      # positive = under-prediction
    full = torch.maximum(taus * u, (taus - 1) * u)
    under_only = taus * torch.clamp(u, min=0)      # y is a lower bound of demand
    loss = torch.where(censored.unsqueeze(1) > 0, under_only, full)
    return loss.mean()
