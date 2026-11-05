# Q-DAGFlow

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23986453.svg)](https://doi.org/10.5281/zenodo.23986453)

Simulator, schedulers and experiment data for the paper *"Q-DAGFlow: Precision-Guaranteed, Queue-Hedged
Co-Scheduling of Hybrid Quantum–HPC AI Workflows with LP-Certified Execution-Mode Assignment"* (T. H. Sardar).

## What this repository contains — and what it does not

| Content | Nature |
|---|---|
| `data/esp_table.json` | Fidelity (ESP) and circuit durations **computed from public IBM calibration snapshots** shipped with `qiskit-ibm-runtime` (`fake_sherbrooke`, `fake_kyiv`, `fake_torino`, `fake_fez`) |
| `data/esp_validation.json` | **Density-matrix simulations** (Qiskit Aer) with full device noise models, used to validate the bias model |
| `data/emulation_timing.json` | **Measured** statevector emulation times (Qiskit Aer, 8–26 qubits) on a 2-vCPU host |
| `data/exp_*.json` | **Raw outputs of the discrete-event simulator**, one record per run |
| `data/*.csv`, `figs/` | Tables and figures derived from the raw outputs by the analysis scripts |

All scheduling results are **simulation results**. No physical quantum processor, computer cluster or GPU was used: every
simulation, analysis and emulation-time measurement ran on a single Linux virtual machine with two virtual CPUs. The
simulator's queue, outage and drift processes are parameterised models, and the simulated classical tier (128 nodes by
default, 16-500 in the sensitivity study; 16 GB per node; 940 Mb/s, 0.18 ms round-trip latency) uses modelled values
typical of commodity desktop nodes on switched Gigabit Ethernet (paper, Table A1).

## Layout

| Path | Content |
|---|---|
| `calib/` | circuit templates; ESP tables; noise-model validation; emulation timing |
| `qdag/model.py` | platform, stochastic environment, CAMS mode synthesis, hedged modes, precision-violation model |
| `qdag/workflows.py` | workflow generators (QK-HPO, VQA-Loop, SQD-Hybrid, WfCommons Montage/Epigenomics/1000Genome hybrids) |
| `qdag/lp.py` | LP-MA: LP relaxation (HiGHS via SciPy), Markov/arg-max rounding, budget repair, CPM, typed bound |
| `qdag/policies.py` | HEFT-Q, PEFT-Q, SHEFT-Q, QG-Dyn, FG-Dyn, CF-Dyn, QGH-Dyn, FT-Dyn, Q-DAGFlow, strict lower bound |
| `qdag/sim.py` | discrete-event simulator (hedged replicas, task-keyed common random numbers) |
| `experiments.py` | experiment definitions: `main`, `strong` (strengthened baselines QGH-Dyn and FT-Dyn), `ablation`, `ablation_L`, `practice`, `budget`, `sens`, `scal` |
| `analyze_main.py`, `analyze_rest.py`, `analyze_robust.py`, `fig_*.py`, `make_values.py` | statistics, figures, and every number quoted in the paper (`results/values.json`) |
| `verify.py` | re-runs 20 stored runs and checks that they reproduce exactly |
| `verify_theory.py` | numerical checks of every lemma, proposition and theorem of the paper (Lemma 1(a)–(c), eq. (4), Lemma 2, Proposition 2, Lemma 3, Theorem 1, eq. (13)) |

## Install and verify

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python verify.py            # expects "mismatches: 0"
python verify_theory.py     # expects "all checks passed"
```

## Reproduce everything

```bash
python experiments.py main        # 4,320 runs: 540 instances x (7 methods + lower bound)
python experiments.py ablation; python experiments.py ablation_L; python experiments.py practice
python experiments.py budget;   python experiments.py sens;       python experiments.py scal
python experiments.py strong
python analyze_main.py && python fig_main.py
python analyze_rest.py practice ablation ablation_L budget sens scal
python analyze_robust.py
python make_values.py
```

Seeds 1–30 define each instance (workflow + platform realisation); all methods share the same realisation.
Results are deterministic for the pinned package versions and independent of `PYTHONHASHSEED` (LP rows are built in sorted order); other versions of SciPy/HiGHS or WfCommons may change
individual values slightly. Optional: `python calib/calib_esp.py`, `calib/validate_esp.py` and `calib/emul_timing.py`
regenerate the calibration inputs (emulation timings will differ on other hardware).

## Citation

Please cite the archived software as: T. H. Sardar, *Q-DAGFlow: simulator, schedulers and experiment data* (v1.0.0), Zenodo, 2026. https://doi.org/10.5281/zenodo.23986453 (ORCID [0000-0001-7880-0225](https://orcid.org/0000-0001-7880-0225)). See also `CITATION.cff`. Please also cite Qiskit and WfCommons, which this work uses as dependencies.

## License

MIT (see `LICENSE`).
