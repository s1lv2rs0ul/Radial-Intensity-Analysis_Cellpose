# Engineering curriculum — hardening this pipeline the way software gets shipped

A learning document as much as a plan. Each step applies one named school of
thought from software engineering to this repo, has a measurable exit
criterion, and lands as its own commit — so the history itself teaches the
method. Statuses are updated as steps land.

## The two laws the whole plan obeys

**1. Measure first (Knuth: "premature optimization is the root of all evil";
Amdahl's law caps every speedup at the share of time you actually touch).**
Our own instrument (`summary/runtime_profile.csv`, see `SCALING.md`) says:
segmentation 48 % (already C++ inside TensorFlow), figure rendering 35 %
(matplotlib), everything else — all the Python analysis math — **17 %**.
So vectorizing the analysis code harder has a ceiling of a few percent:
not worth touching. The real hardware levers, in order: process-parallelism
across images (~3–4× realistic on 10 cores), possibly GPU inference
(`tensorflow-metal`, must be measured not assumed), and a figure-rendering
diet. Anything else is performance theater.

**2. No behavior change without a safety net (Feathers, *Working Effectively
with Legacy Code*: "legacy code is simply code without tests").**
Working, validated, untested code is the normal state of scientific software —
and the reason refactors and "small" changes silently corrupt results. Hence
the step order below: tests *before* restructuring, restructuring *before*
optimization.

## The steps

| # | Step | School of thought | Exit criterion | Status |
|---|---|---|---|---|
| 0 | Observability: `runtime_profile.*`, QC flags, `run_parameters.json` provenance | SRE/DevOps: you can't improve what you can't see | every run self-measures | **done** |
| 1 | **Golden (characterization) tests** — `tests/golden_test.py` freezes today's outputs on a pinned 4-image reference set into `tests/golden_reference.json` | Feathers: characterize the existing behavior before changing anything | two consecutive runs pass (determinism proven); any science-shifting change fails loudly | **done** |
| 2 | **Module extraction** — notebook cells → `radial_pipeline.py`; the notebook becomes a thin interactive client; also the GUI prerequisite | Fowler, *Refactoring*: behavior-preserving transformation, mechanically verified | golden test passes unchanged against the module — that *is* the proof of preservation | planned |
| 3 | **Synthetic canary + metamorphic/property tests** — generator draws a colony with known truth (N nuclei, known SMAD2 ramp); tests assert recovery. Metamorphic relations: µm-answers invariant to pixel size (the caged founding incident), invariant to image rotation, equivariant to intensity scaling. Property-based fuzzing (Hypothesis) on the pure math | Metamorphic testing (Chen, 1998): when you don't know the true output, you still know how it must transform under input transformations — the single most useful QC idea for scientific code | `pytest` green: count ±2 %, peak radius ±5 µm, pixel-size invariance exact in µm-space; notch → imputation refuses | planned |
| 4 | **Contracts + mass-balance ledger** — physics assertions halt an image on unit errors (median nucleus area outside physical bounds, colony radius > field of view); `nucleus_ledger.csv` with `detected + split_added = kept + rejected` asserted; batch tripwire (>50 % image failures aborts, naming the common exception); `qc_metrics.json`; `batch_history.csv` one row per run | Design by Contract (Meyer/Hoare): pre/postconditions as executable spec. Fail-fast for *config/unit* errors only — *quality* problems keep flag/rescue/downweight (fault-tolerance school), as decided in Aug 2026 | the 2026 pixel-size incident, replayed, halts with a message naming the unit error | planned |
| 5 | **Profile-guided performance** — dtype audit (float32 end-to-end), figure diet (reuse figure objects, DPI tiers), `ProcessPoolExecutor` across images with TF thread pinning (workers × intra-op threads ≤ cores, else oversubscription makes it *slower*), optional `tensorflow-metal` experiment | Knuth/Amdahl: optimize only what the profile indicts; keep only what the before/after numbers justify | measured with our own `runtime_profile.csv`; target ≥2.5× batch wall-clock on 10 cores, zero golden-test drift | planned |
| 6 | **CI** — GitHub Actions: tier-1 tests (pure math, no TensorFlow, seconds) on every push; canary tier on demand | Continuous integration: the red X arrives before the data does | a deliberately broken formula turns the badge red | planned |
| 7 | Ship discipline — version tags, pinned env, changelog, version-stamped outputs | Release engineering | (largely already in place) | mostly done |

## Failure philosophy (decided Aug 2026, unchanged)

- **Quality problems** (focus, density, coarse sampling, gaps): never halt —
  flag, rescue, downweight. Discarding a rescuable image throws away paid
  biology; the 76→822-nuclei rescue exists because of this rule.
- **Config/unit problems** (wrong µm/px, wrong channel order): halt that image
  immediately and loudly. Silent wrongness is strictly worse than a crash —
  the pixel-size incident produced a *plausible* wrong result that cost a
  diagnosis session.

## Testing vocabulary used above

- **Golden / characterization test**: pin current real-data outputs; detects
  *any* change, intended or not. Local-only (needs the reference images).
- **Canary test**: synthetic data with known truth; portable, CI-friendly.
- **Metamorphic test**: assert relations between runs (`f(transform(x))` vs
  `f(x)`) when truth is unknown.
- **Property-based test**: a framework (Hypothesis) generates thousands of
  inputs trying to falsify an invariant you state.
- **SPC / control charts** (Shewhart): once `batch_history.csv` holds ~10
  batches, fit control limits and flag drifting batches — not before; limits
  from n=2 are numerology.
