# Data and provenance

Two public-domain experimental structures from the Protein Data Bank are
bundled in `data/structures/`. Everything else is generated from stated
equations. There is no patient data, no personal data and nothing taken from
a proprietary or licensed dataset. The code is the authority for every range
below.

## Experimental structures

| file | PDB id | molecule | used by | note |
|---|---|---|---|---|
| `1AKE.pdb` | 1AKE | *E. coli* adenylate kinase, closed conformation, 214 residues (protein atoms only) | t01 | domains NMP (30-59) and LID (122-159) are rotated by the generator |
| `1UBI.pdb` | 1UBI | human ubiquitin, 1.80 Angstrom, 76 residues | t02, t03 | host of the stability landscape and of the elastic-network trajectory |

PDB depositions are released into the public domain (wwPDB policy). The 1UBI header (id, title, resolution) is kept as downloaded. The structure is a
starting geometry only: every quantity the graders check comes from a
planted, hidden model, never from the deposit.

## How a case is generated

* A case is a pure function of one 128-bit seed. `streams(seed, n)` spawns
  independent generators through `numpy.random.SeedSequence`, one per source
  of variation, so adding a source never changes the others.
* Rejection sampling keeps only cases where the task is solvable with margin
  (for example t03 requires at least six transitions in each direction and a
  minor-state population between 8 and 50 percent).
* Seeds: `tests/dev_seeds.txt` per task (public, development only) and
  `private_seeds.txt` (made with `proteng seeds new`, never committed, in
  `.gitignore`) for held-out evaluation. Validation-campaign seeds are drawn
  fresh and recorded in `validation/results/`.

## Distributions by task

### t01 model triage
Fault kind drawn with probabilities 1/2 domain rotation, 1/4 local error, 1/4
accurate. Rotation 12-45 degrees about the hinge axis; local error 9-12
residues displaced 4-7 Angstrom; coordinate noise 0.3-0.7 Angstrom rms.
Reference renumbered from a random start and missing a short loop; model has
a three-residue tag.

### t02 stability design
Hidden energy landscape over ubiquitin: single-substitution ddG from burial,
hydrophobicity, packing and Pro/Gly terms; antagonism between stabilising
neighbours; twelve synergistic pairs. Scan noise 0.12 kcal/mol (singles) and
0.15 (doubles); forbidden positions: twelve fixed functional residues plus
four drawn per seed; at most five substitutions.

### t03 switch kinetics
Elastic network of the ubiquitin C-alpha trace, 4000 frames of 0.1 ns,
slowest relaxation 1-3 ns, target RMSF 0.6-1.0 Angstrom. One segment of 9-13
residues moves 4-6 Angstrom between two states; rates 0.01-0.04 per ns
(major to minor) and 0.04-0.12 per ns (minor to major). Every frame is
randomly rotated and translated.

### t04 SPR kinetics
Blank plus five analyte concentrations at 0.25-4 x KD, 120 s injection and
300 s dissociation, white noise 0.25 RU, bulk shift proportional to
concentration, slow common drift and a per-cycle baseline offset. One surface
of: one-to-one, mass-transport-limited, or two independent sites whose
dissociation rates differ by at least 20-fold.

### t05 QSAR
Six fused-heteroaromatic cores (quinazoline, pyrazolopyrimidine,
thienopyrimidine, purine, pyrrolopyrimidine, furopyrimidine) joined to
substituents at two positions with RDKit `molzip`. Four cores train (90
records each), two test (80 each); 30 percent of training records are
replicates, 25 percent are written as salts or non-canonical SMILES;
measurement noise 0.20 log units; half of each test core's compounds carry a
substituent never seen in training. The fragment library is invented
chemistry in generic drug-like space, not a vendor catalogue.

### t06 inhibitor mechanism
Mass-action progress curves, 600 s, substrate at 0.25-6 x Km and inhibitor at
0-10 x the apparent Ki, background 500 RFU, noise 40 RFU. Mechanism drawn from
competitive, uncompetitive, mixed (alpha at most 0.25 or at least 4) and
non-competitive; some seeds are tight-binding (Ki near the enzyme
concentration).

## Limitations

* Synthetic data is simpler than a real assay or trajectory: no outliers from
  failed experiments, no instrument artefacts beyond those listed, and
  elastic-network dynamics instead of an all-atom force field (t03).
* The hidden structure-activity and stability models are smooth functions
  with a few planted interactions; real landscapes are rougher. A task that
  a method passes here shows it avoids the named traps, not that it will
  rank real compounds well.
* Two protein hosts only. Results do not transfer to membrane proteins or
  intrinsically disordered regions.
