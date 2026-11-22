"""FR-4: realistic wrong solutions fail, each at the check named for the mistake.

Each control edits the output of the reference solution into what a plausible
wrong method would have produced. The expected code must be among the
failures, and the controls that stand for a *single* mistake must fail nothing
that is unrelated to it (no collateral failures), which keeps each check
honest about what it measures.
"""

from __future__ import annotations

import pytest
from conftest import edit_json, failed, find_seed, reference, variant

pytestmark = pytest.mark.slow


def _run(task_id: str, seed: int, edit):
    case, out = reference(task_id, seed)
    return failed(task_id, case, variant(out, edit))


# ---- t01 -------------------------------------------------------------------------


T1 = "t01-model-triage"


def _t1(kind: str) -> int:
    return find_seed(T1, lambda c: c.kind == kind)


def test_t01_index_alignment_wrong_rmsd_and_pairs() -> None:
    codes = _run(
        T1,
        _t1("domain_rotation"),
        edit_json(
            "triage.json",
            lambda r: r.update(
                n_matched=r["n_matched"] + 40, rmsd_global_A=r["rmsd_global_A"] * 4, tm_score=0.2
            ),
        ),
    )
    assert {"MATCHED", "RMSD_GLOBAL", "TM_SCORE"} <= set(codes)


def test_t01_rotation_called_local_error() -> None:
    def edit(r: dict) -> None:
        r["verdict"] = "local_error"
        r["rotation_angle_deg"] = None

    assert "VERDICT" in _run(T1, _t1("domain_rotation"), edit_json("triage.json", edit))


def test_t01_global_rmsd_only_calls_everything_accurate() -> None:
    def edit(r: dict) -> None:
        r.update(
            verdict="accurate",
            moved_segment=None,
            hinge_residues=None,
            rotation_angle_deg=None,
            rmsd_parts_A=None,
        )

    codes = _run(T1, _t1("domain_rotation"), edit_json("triage.json", edit))
    assert codes[0] == "VERDICT" and "SEGMENT" in codes


def test_t01_constant_verdict_fails_a_different_kind() -> None:
    def edit(r: dict) -> None:
        r.update(
            verdict="domain_rotation",
            moved_segment=[10, 40],
            hinge_residues=[9, 41],
            rotation_angle_deg=20.0,
            rmsd_parts_A={"moved": 0.5, "rest": 0.5},
        )

    assert "VERDICT" in _run(T1, _t1("accurate"), edit_json("triage.json", edit))


def test_t01_wrong_segment_and_angle() -> None:
    def edit(r: dict) -> None:
        r["moved_segment"] = [r["moved_segment"][0] + 15, r["moved_segment"][1] + 15]
        r["rotation_angle_deg"] = r["rotation_angle_deg"] * 2.0

    codes = _run(T1, _t1("domain_rotation"), edit_json("triage.json", edit))
    assert {"SEGMENT", "ANGLE"} <= set(codes)


def test_t01_hinge_at_the_wrong_residues() -> None:
    def edit(r: dict) -> None:
        r["hinge_residues"] = [r["hinge_residues"][0] + 9, r["hinge_residues"][1] + 9]

    assert "HINGE" in _run(T1, _t1("domain_rotation"), edit_json("triage.json", edit))


def test_t01_per_part_rmsd_of_a_single_global_fit() -> None:
    """Per-domain RMSD copied from the global fit instead of fitting each part separately."""

    def edit(r: dict) -> None:
        r["rmsd_parts_A"] = {"moved": r["rmsd_global_A"], "rest": r["rmsd_global_A"]}

    assert "PART_RMSD" in _run(T1, _t1("domain_rotation"), edit_json("triage.json", edit))


# ---- t02 -------------------------------------------------------------------------

T2 = "t02-stability-design"


def _t2_seed() -> int:
    return find_seed(T2, lambda c: True)


def test_t02_forbidden_position_is_rejected() -> None:
    case, out = reference(T2, _t2_seed())
    pos = sorted(case.forbidden)[0]
    wt = case.land.sequence[pos - 1]
    mut = "A" if wt != "A" else "G"

    def edit(r: dict) -> None:
        r["substitutions"] = [*r["substitutions"][:4], f"{wt}{pos}{mut}"]

    assert "CONSTRAINTS" in failed(T2, case, variant(out, edit_json("design.json", edit)))


def test_t02_too_many_substitutions_and_repeated_positions() -> None:
    def many(r: dict) -> None:
        r["substitutions"] = r["substitutions"] * 2

    assert "CONSTRAINTS" in _run(T2, _t2_seed(), edit_json("design.json", many))


def test_t02_additive_top_five_with_overconfident_claim() -> None:
    """Top five singles by measured effect, summed: predicts too much and claims too little uncertainty."""
    seed = _t2_seed()
    case, out = reference(T2, seed)
    rows = sorted((r for r in case.single_rows if r[0] not in case.forbidden), key=lambda r: r[3])
    chosen, used, total = [], set(), 0.0
    for pos, wt, mut, v in rows:
        if pos in used:
            continue
        chosen.append(f"{wt}{pos}{mut}")
        used.add(pos)
        total += v
        if len(chosen) == 5:
            break

    def edit(r: dict) -> None:
        r.update(substitutions=chosen, predicted_ddG_kcal_per_mol=total, sigma_kcal_per_mol=0.2)

    codes = failed(T2, case, variant(out, edit_json("design.json", edit)))
    assert "CALIBRATION" in codes  # the additive sum overshoots the true value by far more than 0.5


def test_t02_prediction_far_from_truth_fails_calibration() -> None:
    def edit(r: dict) -> None:
        r["predicted_ddG_kcal_per_mol"] = r["predicted_ddG_kcal_per_mol"] - 4.0

    assert _run(T2, _t2_seed(), edit_json("design.json", edit)) == ["CALIBRATION"]


def test_t02_a_huge_sigma_cannot_buy_a_pass() -> None:
    def edit(r: dict) -> None:
        r["predicted_ddG_kcal_per_mol"] = 0.0
        r["sigma_kcal_per_mol"] = 50.0

    assert "CALIBRATION" in _run(T2, _t2_seed(), edit_json("design.json", edit))


def test_t02_a_single_good_substitution_is_not_enough() -> None:
    def edit(r: dict) -> None:
        r["substitutions"] = r["substitutions"][:1]

    assert "STABILISED" in _run(T2, _t2_seed(), edit_json("design.json", edit))


# ---- t03 -------------------------------------------------------------------------

T3 = "t03-md-switch-kinetics"


def _t3_seed() -> int:
    return find_seed(T3, lambda c: True)


def test_t03_all_atom_superposition_signature_fails_rmsf() -> None:
    """Fitting on every atom lets the mobile segment drag the fit: the core looks more mobile."""

    def edit(r: dict) -> None:
        r["rmsf_A"] = [v * 1.35 for v in r["rmsf_A"]]

    assert "RMSF" in _run(T3, _t3_seed(), edit_json("analysis.json", edit))


def test_t03_switching_segment_shifted() -> None:
    def edit(r: dict) -> None:
        r["switching_residues"] = [r["switching_residues"][0] + 6, r["switching_residues"][1] + 6]

    assert "SWITCH_RESIDUES" in _run(T3, _t3_seed(), edit_json("analysis.json", edit))


def test_t03_every_frame_in_one_state() -> None:
    def edit(d) -> None:
        lines = (d / "states.csv").read_text().splitlines()
        (d / "states.csv").write_text(
            "\n".join([lines[0], *[f"{i},0" for i in range(len(lines) - 1)]]) + "\n"
        )

    assert "STATES" in _run(T3, _t3_seed(), edit)


def test_t03_rates_off_by_a_factor_two_and_population_wrong() -> None:
    def edit(r: dict) -> None:
        r["rate_major_to_minor_per_ns"] *= 2.0
        r["rate_minor_to_major_per_ns"] *= 0.5
        r["population_minor"] = 0.5

    codes = _run(T3, _t3_seed(), edit_json("analysis.json", edit))
    assert {"RATES", "POPULATION"} <= set(codes)


# ---- t04 -------------------------------------------------------------------------

T4 = "t04-spr-kinetics"


def _t4(model: str) -> int:
    return find_seed(T4, lambda c: c.model == model)


def test_t04_wrong_model_label() -> None:
    def edit(r: dict) -> None:
        r["model"] = "mass_transport"
        r["parameters"]["kt_RU_per_M_per_s"] = 1e9

    assert "MODEL" in _run(T4, _t4("one_to_one"), edit_json("fit.json", edit))


def test_t04_kinetics_scaled_but_kd_consistent() -> None:
    def edit(r: dict) -> None:
        p = r["parameters"]
        p["kon_per_M_per_s"] *= 3
        p["koff_per_s"] *= 3  # KD unchanged: a fit stuck on the wrong ridge of the likelihood

    codes = _run(T4, _t4("one_to_one"), edit_json("fit.json", edit))
    assert "KINETICS" in codes and "KD_CONSISTENT" not in codes


def test_t04_inconsistent_kd() -> None:
    def edit(r: dict) -> None:
        r["parameters"]["KD_M"] *= 2

    assert "KD_CONSISTENT" in _run(T4, _t4("one_to_one"), edit_json("fit.json", edit))


def test_t04_kd_wrong_but_self_consistent() -> None:
    def edit(r: dict) -> None:
        p = r["parameters"]
        p["kon_per_M_per_s"] /= 2.5
        p["KD_M"] = p["koff_per_s"] / p["kon_per_M_per_s"]

    codes = _run(T4, _t4("one_to_one"), edit_json("fit.json", edit))
    assert "KD" in codes and "KD_CONSISTENT" not in codes


def test_t04_two_site_surface_fitted_with_swapped_rates() -> None:
    def edit(r: dict) -> None:
        p = r["parameters"]
        p["koff1_per_s"] *= 6
        p["koff2_per_s"] /= 6

    assert "SITES" in _run(T4, _t4("two_site"), edit_json("fit.json", edit))


@pytest.mark.parametrize("se", [1e-6, 3.0])
def test_t04_overconfident_and_absurdly_wide_errors(se: float) -> None:
    def edit(r: dict) -> None:
        r["se_log"] = {"kon": se, "koff": se}

    assert "UNCERTAINTY" in _run(T4, _t4("one_to_one"), edit_json("fit.json", edit))


# ---- t05 -------------------------------------------------------------------------

T5 = "t05-qsar-scaffold-hop"


def _t5_seed() -> int:
    return find_seed(T5, lambda c: True)


def _t5_realised(seed: int) -> float:
    import csv

    from scipy.stats import spearmanr

    case, out = reference(T5, seed)
    pred = {
        r["id"]: float(r["pic50_pred"]) for r in csv.DictReader((out / "predictions.csv").open())
    }
    return float(spearmanr([pred[r[0]] for r in case.test_rows], case.test_truth).statistic)


def test_t05_random_split_claim_overstates_performance() -> None:
    """A random cross-validation reports about 0.9; here the claim exceeds the realised value by 0.3."""
    seed = find_seed(T5, lambda c: True)
    claim = min(1.0, _t5_realised(seed) + 0.30)
    if claim - _t5_realised(seed) < 0.25:
        pytest.skip("realised Spearman too close to 1 on this seed to overclaim by 0.25")

    def edit(r: dict) -> None:
        r["expected_spearman_new_core"] = claim

    assert _run(T5, seed, edit_json("report.json", edit)) == ["HONESTY"]


def test_t05_a_safe_low_claim_is_also_rejected() -> None:
    seed = find_seed(T5, lambda c: True)

    def edit(r: dict) -> None:
        r["expected_spearman_new_core"] = _t5_realised(seed) - 0.7

    assert _run(T5, seed, edit_json("report.json", edit)) == ["HONESTY"]


def test_t05_unstandardised_training_table_miscounted() -> None:
    def edit(r: dict) -> None:
        r["n_unique_training_compounds"] += 120  # distinct strings, not distinct compounds

    assert _run(T5, _t5_seed(), edit_json("report.json", edit)) == ["DEDUP"]


def test_t05_shuffled_predictions_have_no_rank_skill() -> None:
    import csv
    import random

    def edit(d) -> None:
        rows = list(csv.reader((d / "predictions.csv").open()))
        body = rows[1:]
        vals = [r[1] for r in body]
        random.Random(1).shuffle(vals)
        with (d / "predictions.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(rows[0])
            w.writerows([[r[0], v] for r, v in zip(body, vals, strict=True)])

    assert "RANK" in _run(T5, _t5_seed(), edit)


def test_t05_nominating_the_least_active_compounds() -> None:
    seed = _t5_seed()
    case, out = reference(T5, seed)
    worst = [
        r[0]
        for _, r in sorted(zip(case.test_truth, case.test_rows, strict=True), key=lambda x: x[0])[
            :20
        ]
    ]
    codes = failed(
        T5, case, variant(out, edit_json("report.json", lambda r: r.update(top20=worst)))
    )
    assert codes == ["SELECTION"]


def test_t05_missing_predictions_fail_coverage() -> None:
    def edit(d) -> None:
        lines = (d / "predictions.csv").read_text().splitlines()
        (d / "predictions.csv").write_text("\n".join(lines[:-10]) + "\n")

    assert "COVERAGE" in _run(T5, _t5_seed(), edit)


# ---- t06 -------------------------------------------------------------------------

T6 = "t06-inhibitor-mechanism"


def test_t06_wrong_mechanism() -> None:
    seed = find_seed(T6, lambda c: True)
    case, _ = reference(T6, seed)
    other = "uncompetitive" if case.mechanism == "competitive" else "competitive"

    def edit(r: dict) -> None:
        r["mechanism"] = other
        r.pop("alpha", None)

    assert "MECHANISM" in _run(T6, seed, edit_json("inhibition.json", edit))


def test_t06_cheng_prusoff_applied_to_the_wrong_mechanism() -> None:
    """Ki from IC50 with the competitive equation although the mechanism is not competitive."""
    seed = find_seed(T6, lambda c: True)

    def edit(r: dict) -> None:
        r["Ki_from_IC50_uM"] = r["Ki_from_IC50_uM"] * 1.4

    assert "KI_FROM_IC50" in _run(T6, seed, edit_json("inhibition.json", edit))


def test_t06_biased_constants_and_ic50() -> None:
    seed = find_seed(T6, lambda c: True)

    def edit(r: dict) -> None:
        r["Km_uM"] *= 1.6  # substrate depletion read as a lower rate
        r["IC50_uM_at_10uM_S"] *= 1.4

    codes = _run(T6, seed, edit_json("inhibition.json", edit))
    assert {"KM_KCAT", "IC50"} <= set(codes)


def test_t06_ki_biased_threefold() -> None:
    seed = find_seed(T6, lambda c: True)

    def edit(r: dict) -> None:
        r["Ki_uM"] *= 3.0

    assert "KI" in _run(T6, seed, edit_json("inhibition.json", edit))


def test_t06_mixed_inhibition_with_alpha_pinned_to_one() -> None:
    seed = find_seed(T6, lambda c: c.mechanism == "mixed")

    def edit(r: dict) -> None:
        r["alpha"] = 1.0

    assert "ALPHA" in _run(T6, seed, edit_json("inhibition.json", edit))
