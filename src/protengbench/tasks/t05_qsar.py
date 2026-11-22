"""t05: predict activity on chemical series the model has never seen.

The agent gets a training table of measured pIC50 values for compounds from
four fused-heteroaromatic cores (two substituents each, a kinase-hinge-binder
style series), and a list of unlabelled compounds from the two remaining
cores. It must predict their activity, nominate the 20 compounds to make
first, report how many distinct compounds the training table really contains,
and say how well it expects to rank compounds from a new core.

Hidden truth (models/qsar.py): activity is a transferable function of
substituent properties plus a series-specific term that does not transfer
(see that module). The data have the faults of real assay tables:

* replicate measurements of one compound, written under different ids;
* salt forms and non-canonical SMILES, so one compound appears as several
  strings (the naive count of distinct strings is too high);
* measurement noise.

Two mistakes this task is built to catch, both common in published QSAR:

* Validating with a random split. Compounds of the same core sit on both sides
  of the split, so series-specific effects are learnt and the score is far too
  optimistic for a new core. The honesty check compares the performance the
  agent claims for a new core with what it actually achieves on the held-out
  cores; a random-split estimate misses by a wide margin, a core-grouped
  estimate does not.
* Describing molecules only by bit patterns of the training substituents.
  Half of each test core's compounds carry a substituent (four per position)
  that never occurs in training; a fingerprint model has nothing to say
  about them, while a model of substituent properties can extrapolate.

How each output is judged:

* Coverage: every test compound has one finite prediction.
* Standardisation: `n_unique_training_compounds` must equal the true number
  of distinct compounds (exact).
* Ranking: Spearman correlation of predictions with the true activity on the
  held-out cores must reach RANK_MIN.
* Selection: the 20 nominated compounds must average above the median
  of the true activity of the test set.
* Honesty: the claimed Spearman for a new core must not exceed the realised
  value by more than HONESTY_OVER (overclaiming is the harmful error); a
  claim far below it (HONESTY_UNDER) is also rejected, because a number
  chosen to be safe is not an estimate.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit import Chem
from scipy.stats import spearmanr

from protengbench.checks.base import CheckContext, CheckGroup, Outcome, check
from protengbench.checks.io import SchemaError
from protengbench.core.registry import register
from protengbench.core.seeds import streams
from protengbench.core.task import TaskPlugin
from protengbench.models import qsar

N_TRAIN_CORES = 4
TRAIN_PER_CORE = 90
TEST_PER_CORE = 80
REPLICATE_FRACTION = 0.30
SALT_FRACTION = 0.25
MEASUREMENT_NOISE = 0.20
SERIES_SD = 0.50
N_NOVEL = 4  # substituents per position that never occur in training
NOVEL_TEST_FRACTION = 0.5
OFFSET_SD = 0.15
TOP_N = 20

RANK_MIN = 0.25
SELECTION_QUANTILE = 0.50
HONESTY_OVER = 0.22  # claimed minus realised
HONESTY_UNDER = 0.60  # realised minus claimed


@dataclass(frozen=True)
class QsarCase:
    train_cores: tuple[str, ...]
    test_cores: tuple[str, ...]
    train_rows: tuple[tuple[str, str, float], ...]  # (id, smiles as written, measured pIC50)
    n_unique_train: int
    test_rows: tuple[tuple[str, str], ...]  # (id, smiles as written)
    test_truth: tuple[float, ...]  # noise-free pIC50, same order as test_rows


def _written(mol: Chem.Mol, rng: np.random.Generator, salt: bool) -> str:
    smi = Chem.MolToSmiles(mol, doRandom=bool(rng.random() < 0.5))
    if salt:
        smi += qsar.SALTS[int(rng.integers(0, len(qsar.SALTS)))]
    return smi


@lru_cache(maxsize=64)
def make_case(seed: int) -> QsarCase:
    r_sar, r_pick, r_noise, r_fmt = streams(seed, 4)
    sar = qsar.draw_sar(r_sar, SERIES_SD, OFFSET_SD)
    names = list(qsar.CORES)
    order = [names[i] for i in r_pick.permutation(len(names))]
    train_cores, test_cores = tuple(order[:N_TRAIN_CORES]), tuple(order[N_TRAIN_CORES:])

    novel1 = set(r_pick.choice(qsar.R1_GROUPS, size=N_NOVEL, replace=False).tolist())
    novel2 = set(r_pick.choice(qsar.R2_GROUPS, size=N_NOVEL, replace=False).tolist())

    def draw(core: str, n: int, novel: bool) -> list[tuple[str, str]]:
        pairs = [(a, b) for a in qsar.R1_GROUPS for b in qsar.R2_GROUPS]
        has_novel = [a in novel1 or b in novel2 for a, b in pairs]
        pool = [p for p, h in zip(pairs, has_novel, strict=True) if h == novel]
        idx = r_pick.choice(len(pool), size=n, replace=False)
        return [pool[i] for i in idx]

    train: list[tuple[str, str, float]] = []
    n_unique = 0
    for core in train_cores:
        for r1, r2 in draw(core, TRAIN_PER_CORE, False):
            mol = qsar.assemble(core, r1, r2)
            truth = sar.pic50(core, r1, r2)
            n_unique += 1
            reps = 1 + (int(r_noise.integers(1, 4)) if r_noise.random() < REPLICATE_FRACTION else 0)
            for _ in range(reps):
                value = truth + float(r_noise.normal(0.0, MEASUREMENT_NOISE))
                train.append(
                    ("", _written(mol, r_fmt, bool(r_fmt.random() < SALT_FRACTION)), value)
                )
    perm = r_fmt.permutation(len(train))
    train_rows = tuple(
        (f"M{k + 1:04d}", train[i][1], round(train[i][2], 3)) for k, i in enumerate(perm)
    )

    test: list[tuple[str, float]] = []
    for core in test_cores:
        n_new = int(TEST_PER_CORE * NOVEL_TEST_FRACTION)
        for r1, r2 in draw(core, TEST_PER_CORE - n_new, False) + draw(core, n_new, True):
            mol = qsar.assemble(core, r1, r2)
            test.append(
                (
                    _written(mol, r_fmt, bool(r_fmt.random() < SALT_FRACTION)),
                    sar.pic50(core, r1, r2),
                )
            )
    perm = r_fmt.permutation(len(test))
    test_rows = tuple((f"C{k + 1:04d}", test[i][0]) for k, i in enumerate(perm))
    test_truth = tuple(test[i][1] for i in perm)
    return QsarCase(train_cores, test_cores, train_rows, n_unique, test_rows, test_truth)


def write_inputs(case: QsarCase, input_dir: Path) -> None:
    with (input_dir / "train.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "smiles", "pIC50"])
        w.writerows(case.train_rows)
    with (input_dir / "test.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "smiles"])
        w.writerows(case.test_rows)
    (input_dir / "assay.json").write_text(
        json.dumps(
            {
                "target": "kinase, biochemical IC50 assay",
                "activity": "pIC50 = -log10(IC50 in mol/L), higher is more potent",
                "train_note": "several rows can be repeat measurements of one compound",
                "test_note": "test compounds come from chemical cores absent from the training table",
                "top_n": TOP_N,
            },
            indent=2,
        )
        + "\n"
    )


# --- checks ------------------------------------------------------------------------


def _case(ctx: CheckContext) -> QsarCase:
    c: QsarCase = ctx.truth
    return c


@check("FILES", gate=True)
def files_present(ctx: CheckContext) -> Outcome:
    missing = [n for n in ("predictions.csv", "report.json") if not (ctx.output_dir / n).is_file()]
    return Outcome(not missing, f"missing {', '.join(missing)}")


def _number(v: object, name: str) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
        raise SchemaError(f"{name} must be a finite number, got {v!r}")
    return float(v)


@check("SCHEMA", gate=True)
def schema(ctx: CheckContext) -> Outcome:
    case = _case(ctx)
    ids = {r[0] for r in case.test_rows}
    with (ctx.output_dir / "predictions.csv").open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or [f.strip() for f in reader.fieldnames] != [
            "id",
            "pic50_pred",
        ]:
            raise SchemaError("predictions.csv header must be: id,pic50_pred")
        preds: dict[str, float] = {}
        for row in reader:
            key = row["id"].strip()
            if key not in ids:
                raise SchemaError(f"predictions.csv: unknown id {key!r}")
            if key in preds:
                raise SchemaError(f"predictions.csv: duplicate id {key!r}")
            try:
                value = float(row["pic50_pred"])
            except ValueError as exc:
                raise SchemaError(f"predictions.csv: bad number for {key}") from exc
            if not math.isfinite(value):
                raise SchemaError(f"predictions.csv: non-finite value for {key}")
            preds[key] = value
    raw = json.loads((ctx.output_dir / "report.json").read_text())
    if not isinstance(raw, dict):
        raise SchemaError("report.json: top level must be an object")
    n_unique = raw.get("n_unique_training_compounds")
    if isinstance(n_unique, bool) or not isinstance(n_unique, int):
        raise SchemaError("n_unique_training_compounds must be an integer")
    claimed = _number(raw.get("expected_spearman_new_core"), "expected_spearman_new_core")
    if not -1.0 <= claimed <= 1.0:
        raise SchemaError("expected_spearman_new_core must lie in [-1, 1]")
    top = raw.get("top20")
    if not isinstance(top, list) or len(top) != TOP_N or len({str(t) for t in top}) != TOP_N:
        raise SchemaError(f"top20 must list {TOP_N} distinct compound ids")
    if any(str(t) not in ids for t in top):
        raise SchemaError("top20 contains an id that is not a test compound")
    ctx.outputs.update(preds=preds, n_unique=n_unique, claimed=claimed, top=[str(t) for t in top])
    return Outcome(True)


@check("COVERAGE")
def coverage(ctx: CheckContext) -> Outcome:
    n_missing = len(_case(ctx).test_rows) - len(ctx.outputs["preds"])
    return Outcome(n_missing == 0, f"{n_missing} test compounds have no prediction")


def _realised(ctx: CheckContext) -> float:
    case = _case(ctx)
    preds = ctx.outputs["preds"]
    xs = [preds[r[0]] for r in case.test_rows if r[0] in preds]
    ys = [t for r, t in zip(case.test_rows, case.test_truth, strict=True) if r[0] in preds]
    if len(set(xs)) < 2:
        return 0.0
    rho = float(spearmanr(xs, ys).statistic)
    return 0.0 if math.isnan(rho) else rho


@check("DEDUP")
def dedup(ctx: CheckContext) -> Outcome:
    got, want = ctx.outputs["n_unique"], _case(ctx).n_unique_train
    return Outcome(
        got == want,
        f"reported {got} distinct training compounds, there are {want}",
        abs(got - want),
        0.0,
    )


@check("RANK")
def rank(ctx: CheckContext) -> Outcome:
    rho = _realised(ctx)
    return Outcome(
        rho >= RANK_MIN,
        f"Spearman on the held-out cores is {rho:.2f}, {RANK_MIN} required",
        RANK_MIN - rho,
        0.0,
    )


@check("SELECTION")
def selection(ctx: CheckContext) -> Outcome:
    case = _case(ctx)
    truth = dict(zip((r[0] for r in case.test_rows), case.test_truth, strict=True))
    cutoff = float(np.quantile(list(truth.values()), SELECTION_QUANTILE))
    mean = float(np.mean([truth[t] for t in ctx.outputs["top"]]))
    return Outcome(
        mean >= cutoff,
        f"the {TOP_N} nominated compounds average pIC50 {mean:.2f}; "
        f"the {SELECTION_QUANTILE:.0%} quantile of the test set is {cutoff:.2f}",
        cutoff - mean,
        0.0,
    )


@check("HONESTY")
def honesty(ctx: CheckContext) -> Outcome:
    realised, claimed = _realised(ctx), ctx.outputs["claimed"]
    over = claimed - realised
    ok = -HONESTY_UNDER <= over <= HONESTY_OVER
    return Outcome(
        ok,
        f"claimed Spearman {claimed:.2f} for a new core, realised {realised:.2f}; "
        f"overclaiming by more than {HONESTY_OVER} or underclaiming by more than "
        f"{HONESTY_UNDER} is rejected",
        over,
        HONESTY_OVER,
    )


@register
class QsarTask(TaskPlugin):
    task_id = "t05-qsar-scaffold-hop"
    version = "1.0.0"
    timeout_s = 600.0

    def make_case(self, seed: int) -> QsarCase:
        return make_case(seed)

    def write_inputs(self, case: QsarCase, input_dir: Path) -> None:
        write_inputs(case, input_dir)

    def argv(self, lang: str = "python") -> tuple[str, ...]:
        return ("{submission}", "--indir", "{input_dir}", "--outdir", "{output_dir}")

    def checks(self) -> CheckGroup:
        return CheckGroup(
            "t05",
            [
                CheckGroup("load", [files_present, schema], gate=True),
                CheckGroup("data", [coverage, dedup]),
                CheckGroup("model", [rank, selection, honesty]),
            ],
        )
