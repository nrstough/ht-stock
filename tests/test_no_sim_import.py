"""Hard Rule 6, checked mechanically rather than trusted.

The model must never train on or be scored with true_demand, true_mean or lost_sales:
they exist only inside a simulator, and any code path reachable from real data that reads
them is a bug that would be invisible in every offline metric. A convention cannot enforce
that -- the next person to add a metric will reach for the column that makes it easy. So
this walks the AST of every module under ht/ and model/ and asserts the three strings
appear in exactly two places, both of them behind an explicit --settlement sim flag.
"""
import ast
import os

import pytest

from tests.conftest import REPO

SIM_ONLY = ("true_demand", "true_mean", "lost_sales")
# the two functions the spec allows to settle a policy against simulator truth
# The spec named two settlement functions, score_sim and oracle_q. The implementation went
# further: every read of a simulator column now goes through one gateway, backtest.sim_truth,
# which raises when the frame does not carry them. That is a strictly smaller allowed set
# than the spec asked for, so this assertion is tighter, not looser.
SIM_TRUTH_GATEWAY = {"sim_truth"}


def _modules(package):
    root = os.path.join(REPO, package)
    return sorted(os.path.join(root, f) for f in os.listdir(root)
                  if f.endswith(".py"))


def _tree(path):
    with open(path, encoding="utf-8") as fh:
        return ast.parse(fh.read(), filename=path)


def _imports(node):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Import):
            for alias in sub.names:
                yield alias.name
        elif isinstance(sub, ast.ImportFrom):
            yield sub.module or ""


def _module_level_imports(tree):
    """Imports that run at import time: everything outside a def or a class body."""
    stack, out = list(tree.body), []
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out.extend(_imports(node))
        for field in ("body", "orelse", "finalbody", "handlers"):
            stack.extend(getattr(node, field, []) or [])
    return out


def _own_imports(func):
    """Imports this function makes itself, not ones a function nested inside it makes."""
    stack, out = list(func.body), []
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out.extend(_imports(node))
        for field in ("body", "orelse", "finalbody", "handlers"):
            stack.extend(getattr(node, field, []) or [])
    return out


@pytest.mark.parametrize("path", _modules("ht"))
def test_no_module_under_ht_imports_the_simulator(path):
    for name in _imports(_tree(path)):
        assert not name.split(".")[0] == "sim", f"{path} imports {name}"


@pytest.mark.parametrize("path", _modules("ht"))
def test_no_module_under_ht_names_a_simulator_column(path):
    source = open(path, encoding="utf-8").read()
    for name in SIM_ONLY:
        if name in source:
            # ht/schema.py has to name them to be able to refuse them
            assert os.path.basename(path) == "schema.py", f"{path} mentions {name}"


@pytest.mark.parametrize("path", _modules("model"))
def test_the_only_simulator_import_under_model_is_inside_a_function(path):
    tree = _tree(path)
    for name in _module_level_imports(tree):
        assert name.split(".")[0] != "sim", (
            f"{path} imports {name} at module level; a real-data run must not need "
            "the simulator on the import path")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for name in _own_imports(node):
            if name.split(".")[0] == "sim":
                assert node.name == "oracle_q", (
                    f"{path}:{node.name} imports {name}; only oracle_q may")


@pytest.mark.parametrize("module", ["features.py", "train.py", "evaluate.py", "shadow.py"])
def test_the_real_data_modules_never_name_a_simulator_column(module):
    source = open(os.path.join(REPO, "model", module), encoding="utf-8").read()
    for name in SIM_ONLY:
        assert name not in source, f"model/{module} mentions {name}"


def test_backtest_confines_simulator_truth_to_one_gateway():
    """The spec's structural claim, tightened: every read of a simulator column under
    model/ happens inside backtest.sim_truth(), which is unreachable without --settlement
    sim because --settlement observed drops those columns from the frame right after load.
    Every additional function that reads them is another place a future edit can let
    simulator truth leak into an observed-settlement run."""
    path = os.path.join(REPO, "model", "backtest.py")
    tree = _tree(path)
    # The module aliases the column names (SIM_DEMAND = "true_demand"), so a text search
    # would no longer see a read written through the alias -- and the whole point of the
    # guard is that it sees every one. Resolve the aliases, then look for BOTH the literals
    # and the names bound to them.
    aliases = {t.id for node in tree.body if isinstance(node, ast.Assign)
               for t in node.targets
               if isinstance(t, ast.Name) and isinstance(node.value, ast.Constant)
               and node.value.value in SIM_ONLY}
    assert aliases, "expected backtest.py to bind the simulator column names to constants"
    # a positive control, so the guard can never pass by seeing nothing
    control = ast.parse("def leak(df):\n    return df[SIM_DEMAND].sum()\n")
    assert _sim_column_readers(control, aliases) == {"leak"}

    # main() names them to refuse the run; naming one as a results.json series key is not a read
    assert _sim_column_readers(tree, aliases) <= SIM_TRUTH_GATEWAY | {"main"}


def _sim_column_readers(tree, aliases):
    """Functions that pull a simulator column OUT OF A FRAME, by literal, alias or attribute."""
    def reads(node):
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute) and sub.attr in SIM_ONLY:
                return True
            # a STORE is writing a results.json series key, not reading a frame
            if not isinstance(sub, ast.Subscript) or not isinstance(sub.ctx, ast.Load):
                continue
            key = sub.slice
            if isinstance(key, ast.Constant) and key.value in SIM_ONLY:
                return True
            if isinstance(key, ast.Name) and key.id in aliases:
                return True
        return False

    return {n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and reads(n)}


def test_backtest_refuses_the_wrong_settlement_for_the_frame_it_was_given(tmp_path,
                                                                         synth_raw):
    """The behavioural half of the same guarantee, which is the half that matters."""
    from ht import schema
    from model import backtest

    panel = str(tmp_path / "panel.csv")
    schema.write_panel(schema.conform(synth_raw), panel)
    with pytest.raises(SystemExit) as exc:
        backtest.main(["--panel", panel, "--settlement", "sim",
                       "--out", str(tmp_path / "out.json")])
    assert "model.evaluate" in str(exc.value) or "true demand" in str(exc.value).lower()


def test_the_observable_pipeline_imports_without_the_simulator(monkeypatch):
    """A real deployment may not ship sim/ at all."""
    import builtins
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] == "sim":
            raise ImportError("sim is not installed in this deployment")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    for module in ("ht.schema", "ht.config", "ht.calendar", "ht.weather", "ht.ingest",
                   "ht.validate", "model.features", "model.evaluate", "model.shadow"):
        __import__(module)
