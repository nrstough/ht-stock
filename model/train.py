"""Train the quantile demand network.

Run:  python -m model.train --artifacts .rehearsal/artifacts

Same loop as always: pinball loss with a censoring branch, Adam, early stopping
on a held-out validation window. What is new is that the data source, the item
config, the artifacts directory and the split are all arguments, because the
point of this layer is that a store's export can be trained on without editing
code. Defaults reproduce the frozen configuration exactly.

Two things it refuses to do quietly. It will not write into model/artifacts/
without --force-frozen: those bytes back the dollar figures in the proposal and
a rehearsal must not overwrite them. And it aborts the moment the validation
loss is NaN, naming the row counts and the resolved boundaries, instead of
burning twelve epochs and then dying inside torch on a best_state that was
never set.
"""
import argparse
import datetime as dt
import json
import os

import numpy as np
import torch

from ht import config as ht_config
from ht import schema

from . import features
from .net import DemandNet, censored_pinball

ARTIFACTS = os.path.join(os.path.dirname(__file__), "artifacts")
CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "items.example.json")
SAFE_ARTIFACTS = ".rehearsal/artifacts"

SEED = 7
BATCH = 256
MAX_EPOCHS = 120
PATIENCE = 12
LR = 1e-3
THREADS = 4


class TrainError(RuntimeError):
    """Training cannot proceed, for a reason the data or the split already explains."""


def fit(b, *, epochs=MAX_EPOCHS, patience=PATIENCE, batch=BATCH, lr=LR, seed=SEED,
        threads=THREADS, verbose=True):
    """Train to the best validation pinball loss. Returns (model, history)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(threads)

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
    if verbose:
        print(f"train {n_train} | val {len(va[0])} | test {(b['split']=='test').sum()}")

    model = DemandNet(
        n_items=len(b["items"]),
        ctx_dim=b["ctx"].shape[2],
        cov_dim=b["cov"].shape[1],
        n_quantiles=len(b["taus"]),
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)

    # seeded from epoch 0, so a run whose val loss never improves still saves a
    # real checkpoint instead of handing load_state_dict a None
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_val, best_epoch, bad = float("inf"), -1, 0
    history = {"train": [], "val": [], "max_epochs": int(epochs)}

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        tot = 0.0
        for i in range(0, n_train, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            pred = model(tr[0][idx], tr[1][idx], tr[2][idx])
            loss = censored_pinball(pred, tr[3][idx], taus, tr[4][idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            val = censored_pinball(model(va[0], va[1], va[2]), va[3], taus, va[4]).item()
        if not np.isfinite(val):
            raise TrainError(
                f"validation loss is {val} at epoch {epoch}: the validation window holds "
                f"{len(va[0])} rows over "
                f"{len(np.unique(b['date'][b['split'] == 'val']))} dates "
                f"(train_end {b['train_end']}, val_start {b['val_start']}, "
                f"test_start {b['test_start']}). Early stopping cannot judge that, so "
                "there is nothing to select a checkpoint on.")
        sched.step(val)
        marker = ""
        if val < best_val - 1e-5:
            best_val, best_epoch, bad = val, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = " *"
        else:
            bad += 1
        history["train"].append(tot / n_train)
        history["val"].append(val)
        if verbose:
            print(f"epoch {epoch:3d}  train {tot/n_train:.4f}  val {val:.4f}{marker}")
        if bad >= patience:
            if verbose:
                print("early stop")
            break

    model.load_state_dict(best_state)
    history.update(best_val=best_val, best_epoch=best_epoch,
                   epochs_run=len(history["val"]), n_train=int(n_train))
    return model, history


def save(model, b, artifacts_dir, *, history, seed, panel_path, items_path,
         thin_history=False):
    """Write demandnet.pt and the widened meta.json.

    meta.json keeps every key the frozen artifact has and adds what a later scoring
    run needs to prove it is looking at the same feature layout: the spec and its
    hash, the vocabularies, the resolved boundaries, the item roster's exclusions,
    and the hashes of the panel and the item config the checkpoint was fitted to.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(artifacts_dir, "demandnet.pt"))

    panel_hash = panel_start = panel_end = None
    try:
        df = features.load(panel_path)
        panel_start = str(df.date.min().date())
        panel_end = str(df.date.max().date())
        panel_hash = schema.panel_hash(df)
    except Exception as exc:                 # a hash is provenance, not a precondition
        print(f"note: could not hash the panel for meta.json ({exc.__class__.__name__}: {exc})")

    meta = dict(
        items=b["items"], stats=b["stats"], taus=list(map(float, b["taus"])),
        ctx_dim=int(b["ctx"].shape[2]), cov_dim=int(b["cov"].shape[1]),
        best_val=history["best_val"], n_train=int(history["n_train"]), seed=seed,
        spec=b["spec"], spec_hash=b["spec_hash"],
        cov_layout=b["cov_layout"], ctx_channels=b["ctx_channels"],
        panel=panel_path, panel_hash=panel_hash,
        panel_start=panel_start, panel_end=panel_end,
        span_days=b["span_days"], train_end=b["train_end"],
        val_start=b["val_start"], test_start=b["test_start"],
        excluded_items=b["excluded_items"], dropped_rows=b["dropped_rows"],
        sellout_source=b["sellout_source"], censoring_known=bool(b["censoring_known"]),
        items_config_hash=ht_config.config_hash(items_path) if items_path else None,
        thin_history=bool(thin_history),
        max_epochs=history["max_epochs"], best_epoch=history["best_epoch"],
        epochs_run=history["epochs_run"],
        torch_version=torch.__version__,
        created_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    with open(os.path.join(artifacts_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)


def _describe(b, thin):
    """Everything a person needs to know before the first epoch burns a minute."""
    print(f"span {b['span_days']} days | train_end {b['train_end']} | "
          f"val_start {b['val_start']} | test_start {b['test_start']}")
    print(f"spec_hash {b['spec_hash']} | items {len(b['items'])} | "
          f"ctx {b['ctx'].shape} | cov {b['cov'].shape}")
    known = float(b["cens"].mean())
    print(f"sellout_source {b['sellout_source']} | censored share {known:.3f}")
    if not b["censoring_known"]:
        print("NO SELLOUT DATA: the model is being fitted to the distribution of CENSORED "
              "SALES, not demand, so the quantities it recommends will run roughly 1-8% "
              "low on the busiest days. That is the safe direction and a supported mode. "
              "Do not compensate by inflating the critical fractile.")
    for e in b["excluded_items"]:
        print(f"excluded {e['item']}: {e['reason']}")
    for item, n in sorted(b["dropped_rows"].items()):
        print(f"dropped {n} non-contiguous context windows for {item}")
    if thin:
        print("SHORT HISTORY: no held-out test set, early stopping is unreliable and the "
              "seasonal covariates are unidentified -- treat these forecasts as provisional.")


def _parse_args(argv):
    ap = argparse.ArgumentParser(description="train the quantile demand network")
    ap.add_argument("--panel", default=None, help="canonical panel CSV (default: the simulator's)")
    ap.add_argument("--items", default=CONFIG)
    ap.add_argument("--artifacts", default=ARTIFACTS)
    ap.add_argument("--spec", choices=("legacy", "auto"), default="legacy")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--threads", type=int, default=THREADS)
    ap.add_argument("--val-days", type=int, default=None)
    ap.add_argument("--test-days", type=int, default=None)
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--allow-short", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="build features, print the split and exclusion report, train nothing")
    ap.add_argument("--force-frozen", action="store_true",
                    help="permit writing into model/artifacts, whose bytes back the proposal")
    return ap.parse_args(argv)


def _guard_frozen(artifacts_dir, force):
    if force or os.path.realpath(artifacts_dir) != os.path.realpath(ARTIFACTS):
        return
    raise SystemExit(
        f"refusing to write into {ARTIFACTS}: model/artifacts/demandnet.pt and meta.json are "
        "frozen provenance -- results/results.json and the proposal's dollar figures are "
        f"settled against them. Train into {SAFE_ARTIFACTS} instead, or pass --force-frozen "
        "if you really do mean to replace the published model.")


def _validate_panel(df, items_path, split_opts):
    """--spec auto: a bad panel must fail at the data layer with a data message.

    split_opts carries the split the caller actually asked for, so the history floor the
    validator checks is the one this run will use -- 126 days for train/val/test, 98 with
    --no-test, 70 with --allow-short -- rather than always the strictest of the three.
    """
    from ht import validate as ht_validate

    items = ht_config.load_items(items_path)
    report = ht_validate.validate(schema.conform(df), items, split_opts=split_opts)
    print(ht_validate.format_report(report))
    return report["ok"]


def main(argv=None):
    args = _parse_args(argv)
    if not args.dry_run:
        _guard_frozen(args.artifacts, args.force_frozen)

    df = features.load(args.panel) if args.panel else features.load()

    if args.spec == "auto":
        split_opts = dict(val_days=args.val_days, test_days=args.test_days,
                          no_test=args.no_test, allow_short=args.allow_short)
        if not _validate_panel(df, args.items, split_opts):
            print("\nvalidation found errors: fix the panel before training on it. "
                  "Nothing was written.")
            return 1
        spec = features.spec_for_panel(df, **split_opts)
    else:
        spec = features.legacy_spec()

    try:
        b = features.build(df, spec=spec)
    except (features.InsufficientHistory, features.EmptySplit) as exc:
        print(f"\n{exc}\n")
        print("No model was trained. This is a data problem, not a code problem: the panel "
              "does not carry enough history to hold out a validation window worth stopping "
              "on. Ask the store for more item movement, or re-run with --no-test or "
              "--allow-short and read the caveats those modes print.")
        return 2

    thin = bool(args.allow_short)
    _describe(b, thin)
    if args.dry_run:
        print("--dry-run: nothing trained, nothing written.")
        return 0

    try:
        model, history = fit(b, epochs=args.max_epochs, patience=args.patience,
                             batch=args.batch, lr=args.lr, seed=args.seed,
                             threads=args.threads)
    except TrainError as exc:
        print(f"\n{exc}\n")
        print("No model was saved.")
        return 2

    save(model, b, args.artifacts, history=history, seed=args.seed,
         panel_path=args.panel or features.DATA, items_path=args.items,
         thin_history=thin)
    print(f"saved best model (val pinball {history['best_val']:.4f}) to {args.artifacts}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
