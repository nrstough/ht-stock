"""Train the quantile demand network.

Run:  python -m model.train

Train = 2023-01-29 .. 2024-11-03, val = last ~8 weeks of 2024 (early
stopping), test = 2025 (never touched here). Saves the best checkpoint and
the normalization stats to model/artifacts/.
"""
import json
import os

import numpy as np
import torch

from . import features
from .net import DemandNet, censored_pinball

ARTIFACTS = os.path.join(os.path.dirname(__file__), "artifacts")

SEED = 7
BATCH = 256
MAX_EPOCHS = 120
PATIENCE = 12
LR = 1e-3


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    b = features.build()
    taus = torch.tensor(b["taus"], dtype=torch.float32)

    def tensors(mask):
        return (torch.tensor(b["iidx"][mask]),
                torch.tensor(b["ctx"][mask]),
                torch.tensor(b["cov"][mask]),
                torch.tensor(b["y"][mask]),
                torch.tensor(b["cens"][mask]))

    tr = tensors(b["split"] == "train")
    va = tensors(b["split"] == "val")
    n_train = len(tr[0])
    print(f"train {n_train} | val {len(va[0])} | test {(b['split']=='test').sum()}")

    model = DemandNet(
        n_items=len(b["items"]),
        ctx_dim=b["ctx"].shape[2],
        cov_dim=b["cov"].shape[1],
        n_quantiles=len(b["taus"]),
    )
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)

    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n_train)
        tot = 0.0
        for i in range(0, n_train, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            pred = model(tr[0][idx], tr[1][idx], tr[2][idx])
            loss = censored_pinball(pred, tr[3][idx], taus, tr[4][idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            val = censored_pinball(model(va[0], va[1], va[2]), va[3], taus, va[4]).item()
        sched.step(val)
        marker = ""
        if val < best_val - 1e-5:
            best_val, bad = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = " *"
        else:
            bad += 1
        print(f"epoch {epoch:3d}  train {tot/n_train:.4f}  val {val:.4f}{marker}")
        if bad >= PATIENCE:
            print("early stop")
            break

    os.makedirs(ARTIFACTS, exist_ok=True)
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(ARTIFACTS, "demandnet.pt"))
    with open(os.path.join(ARTIFACTS, "meta.json"), "w") as f:
        json.dump(dict(
            items=b["items"], stats=b["stats"], taus=list(map(float, b["taus"])),
            ctx_dim=int(b["ctx"].shape[2]), cov_dim=int(b["cov"].shape[1]),
            best_val=best_val, n_train=int(n_train), seed=SEED,
        ), f, indent=1)
    print(f"saved best model (val pinball {best_val:.4f}) to {ARTIFACTS}/")


if __name__ == "__main__":
    main()
