# Design: six tasks along one engineering pipeline

The tasks follow the order of a real protein-engineering and early drug
discovery programme. Each has one realistic scientific trap, and the grader
sees the trap through a fingerprint in the output, not by watching the method.

| id | stage | the decision the maths drives | the trap |
|---|---|---|---|
| t01-model-triage | folding: judge a predicted structure | is this model wrong, or are its domains right and only their arrangement wrong? | residue numbering offset and unresolved residues; global RMSD misleads; the confidence column is uninformative |
| t02-stability-design | design: pick stabilising mutations | which combination of at most five substitutions is most stabilising? | neighbouring stabilising mutations are not additive; synergy hides in the double-mutant data |
| t03-md-switch-kinetics | dynamics: read a trajectory | which segment switches, how populated is each state, how fast does it switch? | frames are rotated and translated; superposing on all atoms smears the mobile region |
| t04-spr-kinetics | kinetics: fit binding sensorgrams | kon, koff and KD with honest uncertainty, and which model | mass-transport limitation, drift, bulk shift, heterogeneous ligand |
| t05-qsar-scaffold-hop | drug discovery: model activity | which unseen compounds from new chemical series are most active, and how good is the model really? | duplicate and salt-form records; random cross-validation leaks across scaffolds |
| t06-inhibitor-mechanism | pharmacology: characterise an inhibitor | mechanism of inhibition and Ki | substrate depletion invalidates initial rates; tight binding; IC50 to Ki conversion by the wrong equation |

## Common task anatomy

```
tasks/<id>/
  instruction.md       what the agent reads (goal, files, output schema, units, constraints)
  task.toml            metadata, time limit, resources, network mode
  environment/         container recipe and the example case the agent can see
  solution/            expert reference (hidden from the agent)
  tests/               verifier entry point and seeds (hidden)
src/protengbench/tasks/<id>.py       generator, writer of agent inputs, check suite
src/protengbench/models/             physics and statistics shared by generator and grader
```

The agent sees `instruction.md` and one fully public example case. Hidden
cases come from seeds it never sees.

## Grader layers (every task)

1. **Gate.** The output file exists, parses, has exactly the named keys or
   columns, finite numbers, sensible ranges. Failure is reported as a format
   failure and later checks are skipped.
2. **Accuracy.** Quantities are compared with planted truth or with an
   independent computation, with tolerances placed in the gap between valid
   methods and wrong methods (measured, section "Calibration").
3. **Method validity.** Fingerprints that only sound reasoning produces
   (for example a hinge located within two residues, a predicted
   performance that matches the true held-out performance).

## Task specifications

### t01 model triage

Reference: the closed conformation of Escherichia coli adenylate kinase
(PDB 1AKE, 214 residues, a three-domain enzyme whose LID and NMP domains swing
on hinges). Model: a full-atom copy of the reference with one of three planted
faults: a rigid rotation of one domain (12 to 45 degrees about the axis through
its two hinge C-alpha atoms), a displaced 9 to 12 residue segment (4 to 7
Angstrom), or nothing but coordinate noise (0.3 to 0.7 Angstrom). The model
carries a three-residue N-terminal tag and numbering from 1; the reference is
renumbered from an arbitrary start and loses a short unresolved loop, so no
constant offset maps one file onto the other. The confidence column is high
everywhere except on the tag.

Output `triage.json`: matched residue count, global C-alpha RMSD after optimal
superposition, TM-score, the verdict (`domain_rotation`, `local_error`,
`accurate`), the moved segment, hinge residues, rotation angle and per-part
RMSD.

Truth: the planted pairing and fault. Global RMSD and TM-score are recomputed
by the grader. A naive index alignment fails the matched count; a global-only
analysis fails the verdict and segment checks.

### t02 stability design

Host: human ubiquitin (PDB 1UBI). A hidden energy model gives each single
substitution a change in folding free energy (ddG, kcal/mol, negative is
stabilising) from burial, volume and hydrophobicity terms, plus pairwise
coupling between substitutions in contact (antagonistic for stabilising pairs
within 8 Angstrom) and a few synergistic pairs. The agent receives a noisy
single-mutant scan, the contact list, a subset of measured double mutants,
forbidden positions (functional surface) and a limit of five substitutions.

Output `design.json`: the chosen substitutions, the predicted combined ddG and
a one-sigma uncertainty.

Truth: the hidden model evaluated on the chosen set. Accept if constraints
hold, the true ddG is within a tolerance of the best set found by exhaustive
search, and the prediction is calibrated.

### t03 switch kinetics from a trajectory

Host: ubiquitin C-alpha elastic network. The generator runs overdamped
Langevin dynamics of the network (fluctuations from the pseudo-inverse of the
Hessian), toggles one segment between two states with a two-state Markov
chain (known rates), and applies a random rotation and translation to every
frame. The output of the agent: aligned per-residue RMSF, the switching
residues, the state populations and the two rate constants.

Truth: the hidden state sequence and the exact internal coordinates.
Alignment on the rigid core is what the grader accepts as correct.

### t04 SPR kinetics

Sensorgrams are simulated from the stated binding equations (one-to-one
Langmuir with optional mass transport, or a two-site heterogeneous surface)
with a five-concentration series, a blank for double referencing, bulk shift,
drift and noise. Output: model name, kon, koff, KD, Rmax and 95 percent
intervals.

Truth: simulation parameters; model identity by construction.

### t05 QSAR across scaffolds

Compounds are assembled with RDKit from six fused-heteroaromatic cores and a
library of substituents at two positions. Activity (pIC50) is a transferable
function of substituent properties plus a series-specific term that does not
transfer. Four cores form the training table, which carries the faults of real
assay data: replicate measurements under different ids, salt forms and
non-canonical SMILES. The two remaining cores form the test set, and half of
their compounds carry substituents never seen in training. The agent writes
predictions, a top-20 list, the true number of distinct training compounds and
a statement of expected Spearman correlation on a new core.

Truth: hidden activities. Checks: coverage, exact standardised count, rank
correlation, mean activity of the top 20 against the test median, and an
honesty check that the claimed correlation is within a tolerance of the
realised one (overclaiming by more than 0.22 fails; underclaiming by more than
0.60 fails).

### t06 inhibitor mechanism

Progress curves of an enzyme reaction at several substrate and inhibitor
concentrations are simulated from the mass-action ODEs for competitive,
uncompetitive, mixed and non-competitive inhibition, with substrate depletion
and, in some seeds, tight binding. Output: mechanism, Ki (and alpha for
mixed), IC50 at a stated substrate concentration, and the Ki converted from
that IC50 by the correct equation.

Truth: simulation parameters.

## Calibration

Tolerances are set from data. `validation/campaign.py` runs each reference
and each wrong baseline on fresh 128-bit seeds, records every metric, and the
recorded results are in `validation/results/` (one JSON per task, with seeds,
timings and package versions). A threshold sits inside the measured gap
between the reference and the wrong methods. For t02 the reference reaches
88 to 100 percent of the best achievable stability (threshold 90 percent)
while the additive baseline reaches 28 to 98 percent with a median of about 67; an
additive pick that lands near the optimum by luck is a legitimate pass, and
the calibration check is the second guard (the baseline's claims are
overconfident). For t05 the reference's claim minus realised
correlation lies in -0.35 to +0.12 (cutoff +0.22) where a random-split
estimate gives +0.20 to +0.76, and the duplicate count catches it on every run.
`docs/VALIDATION_REPORT.md` summarises the campaign.
