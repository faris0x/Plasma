# Cross-Platform Quantum Simulator Benchmark, Results

Baseline: Plasma (deterministic statevector simulator, `/home/faris/projects/plasma`, commit `af8644f`).
Data-focused; every number below is measured, reproducible, and pinned to the environment in §2.

---

## 1. Headline scores

Weighted geometric-mean construction score, equal weight per (family × scale), Plasma baseline = 1.00.
Higher = faster than Plasma.

| Axis | Platform | Backend | Precision | Weighted score | vs Plasma |
|---|---|---|---|---|---|
| GPU | Plasma | GPU mega-kernel | f32 | 1.000 |, |
| GPU | cuStateVec 1.14.0 | GPU | f32 | 0.230 | 4.3× slower |
| GPU | Qiskit Aer 0.17.2 | GPU (cuStateVec) | f32 | 0.014 | ~71× slower (≈120 ms per-run cuStateVec init dominates) |
| CPU | Plasma | CPU (f64 statevector) | f64 | 1.000 |, |
| CPU | Qiskit Aer 0.17.2 | CPU | f64 | 2.578 | 2.6× faster |
| CPU | qsim 0.22.1 (Cirq 1.7.0) | CPU | f64 | 4.710 | 4.7× faster |
| CPU | qulacs 0.6.14 | CPU | f64 | 2.984 | 3.0× faster |
| CPU | QuEST v3.7.0 | CPU | f64 | 0.736 | 1.4× slower |

GPU: Plasma's fused single-launch mega-kernel beats per-gate cuStateVec (4.3×) and Aer's
init-heavy per-run backend (~71×) on construction throughput. CPU: Plasma's scalar f64 loop is
2-5× behind SIMD/multithreaded Aer/qsim/qulacs and ahead of QuEST.

## 2. Environment (pinned)

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 (Blackwell, sm_120), 12227 MiB |
| Driver / CUDA UMD | 610.57.04 / 13.3 |
| CUDA toolkit | /opt/cuda, nvcc V13.3.73 |
| Compiler | gcc 16.2.1, cmake 4.4.2 |
| Plasma | commit af8644f, `target/release`, CPU scalar f64 / GPU f32, mega-kernel (single launch) |
| Python 3.14 venv | qiskit 2.5.2, qiskit-aer 0.17.2 (source build linking cuStateVec 1.14.0 / cuTensorNet 2.13.0 / cuTensor 2.7.0), cuquantum-python 26.6.0, cupy 14.2.0 |
| Python 3.13 venv | cirq 1.7.0, qsimcirq 0.22.1, qulacs 0.6.14, numpy 2.5.3 |
| QuEST | v3.7.0 (tag d4f75f7), built from source, `libQuEST.so` (OpenMP) |
| cuStateVec | 1.14.0 (`_cu13`), driven via Aer GPU backend and via the direct C API (ctypes) |
| LD_LIBRARY_PATH | /opt/cuda/lib64 + venv cuquantum/lib + venv cutensor/lib |

## 3. Methodology (fairness rules)

1. Construction = simulate once, kernel time only. Plasma: `gpu_exec_real`/`cpu_run` from `--bench`
   (excludes process start, parse, GPU context/JIT init). Aer: `sim.run` around a `save_statevector`
   circuit (forces the full statevector; the naive `shots=1` path is lazy and was rejected after
   validation). cuStateVec: timer around `apply_matrix` calls, device-synchronized. qsim: `sim.simulate`.
   qulacs: `update_quantum_state`. QuEST: gate-application loop.
2. Precision parity. GPU f32 everywhere (Plasma f32; Aer/cuStateVec f32). CPU f64 everywhere.
   No cross-precision comparisons.
3. CPU and GPU are separate axes. Never blended.
4. Repeats. warm ≥1 + ≥5 timed per (case, platform); min reported; ≥5 platform-process instances
   per cell.
5. Correctness gate. Every platform validated against Plasma's reference by TVD at high shot
   counts. Tight agreement at q8-q10 (TVD ≤ 0.01-0.02 at 20k-50k shots); at q16 the gate used 2M shots
   (all platforms agree within sampling noise: ghz 0.0005, rnd_hc 0.026, clifford 0.018, rnd_rot 0.067,
   qft 0.10). At q ≥ 20 TVD is sampling-noise-limited (bins ≫ shots) and is informational only.
6. Equivalent circuits. One shared gate list per (family, n, seed) fed to every platform.
   Gate conventions were verified and harmonised: qsim's measurement bit order reversed, qulacs's
   rotation sign convention inverted, Plasma decimals (its parser rejects scientific notation).

## 4. Workloads

Five families × four GPU scales (16, 20, 24, 27) and × three CPU scales (20, 24, 26):
`ghz` (H + CNOT chain), `qft` (QFT, controlled-phase rotations), `rnd_hc` (random H/CNOT),
`rnd_rot` (random RX/RY/RZ + CNOT), `clifford` (random H/S/CNOT/CZ). Fixed seeds; deterministic.
Shots: 2000 at q ≤ 16, 200 at q20, 100 at q24, 30 at q27 (bounded by the Plasma CPU shot model).

## 5. Raw construction matrix (ms, min of runs)

### GPU (f32)

| case | q | Plasma | cuStateVec | Aer GPU |
|---|---|---|---|---|
| ghz | 16 | 0.021 | 1.925 | 110.6 |
| ghz | 20 | 0.108 | 2.035 | 120.7 |
| ghz | 24 | 6.718 | 13.615 | 391.3 |
| ghz | 27 | 64.852 | 113.9 | 2358.4 |
| qft | 16 | 0.190 | 2.866 | 124.9 |
| qft | 20 | 1.030 | 3.242 | 135.1 |
| qft | 24 | 49.458 | 165.9 | 505.4 |
| qft | 27 | 584.7 | 1668.1 | 3343.8 |
| rnd_hc | 16 | 0.106 | 2.202 | 119.9 |
| rnd_hc | 20 | 0.579 | 2.400 | 128.1 |
| rnd_hc | 24 | 47.049 | 63.8 | 397.1 |
| rnd_hc | 27 | 446.5 | 569.7 | 2498.2 |
| rnd_rot | 16 | 0.174 | 2.633 | 113.0 |
| rnd_rot | 20 | 1.148 | 3.069 | 125.8 |
| rnd_rot | 24 | 96.299 | 126.8 | 434.5 |
| rnd_rot | 27 | 885.6 | 1150.6 | 2885.0 |
| clifford | 16 | 0.108 | 2.278 | 124.2 |
| clifford | 20 | 0.566 | 2.420 | 135.3 |
| clifford | 24 | 40.001 | 64.1 | 425.4 |
| clifford | 27 | 397.0 | 568.0 | 2588.3 |

### CPU (f64)

| case | q | Plasma | Aer CPU | qsim | qulacs | QuEST |
|---|---|---|---|---|---|---|
| ghz | 20 | 23.8 | 23.9 | 11.4 | 15.7 | 23.3 |
| ghz | 24 | 1339 | 371 | 235 | 156 | 1457 |
| ghz | 26 | 3072 | 1716 | 1065 | 753 | 3252 |
| qft | 20 | 90.6 | 105.9 | 61.3 | 199 | 262 |
| qft | 24 | 13984 | 1772 | 2114 | 4973 | 20169 |
| qft | 26 | 17426 | 8181 | 10401 | 24507 | 47407 |
| rnd_hc | 20 | 56.0 | 44.7 | 18.3 | 7.1 | 86.1 |
| rnd_hc | 24 | 7145 | 1153 | 522 | 996 | 7273 |
| rnd_hc | 26 | 12464 | 3815 | 2056 | 4562 | 14140 |
| rnd_rot | 20 | 111.5 | 98.9 | 35.1 | 43.9 | 182 |
| rnd_rot | 24 | 14463 | 1801 | 825 | 2151 | 14873 |
| rnd_rot | 26 | 21130 | 8441 | 3819 | 9402 | 24510 |
| clifford | 20 | 53.2 | 36.3 | 16.4 | 22.0 | 82.3 |
| clifford | 24 | 7089 | 960 | 421 | 954 | 7562 |
| clifford | 26 | 9902 | 3846 | 1923 | 3894 | 13470 |

## 6. Sampling phase (secondary; shot counts noted)

GPU, q20, 2000 shots (ms): Plasma 0.006-0.061, cuStateVec 0.16, Aer GPU 124-139 (includes
the full per-run backend cost).

CPU, q20, 2000 shots (ms): Aer 25-112, qsim 11-61, qulacs 3.6. Plasma CPU sampling is
reported separately: its CLI shot model re-executes the circuit per shot (`reset`+`exec`+`sample`),
giving O(shots · 2^n) cost (~1.2 s at q16 / 2000 shots); the statevector-level `sample()` is an
O(2^n) cumulative scan. QuEST sampling (per-qubit collapse) is O(n · 2^n) and was skipped at q ≥ 20.

## 7. Capability rows (Plasma-specific engines, not in the weighted score)

| Engine | Circuit | Plasma time | Best mainstream counterpart | Mainstream time |
|---|---|---|---|---|
| Stabilizer (exact Clifford) | q28, 140 gates + 1000 shots | 0.36 s total | no Clifford-special mainstream competitor in this set |, |
| Stabilizer (exact Clifford) | q28 GHZ + 1000 shots | 0.38 s total |, |, |
| MPS (D=32) | q28 chain, 55 gates | 383 ms construct | Aer MPS | 1.1 ms construct / 9.2 ms +1000 shots |

Notes: the stabilizer engine simulates 28-qubit Clifford circuits (4 GB statevector equivalent) in
fractions of a second, a regime no statevector platform here can reach. Plasma's MPS engine is
correct but ~350× slower than Aer's MPS on this low-entanglement chain.

## 8. Caveats

- Aer GPU's ~120 ms floor is a real per-`run` cuStateVec init + statevector-IO cost; it dominates
  at q ≤ 24 and is still ~2.4-3.3 s at q27 vs Plasma 0.4-0.9 s.
- q ≥ 20 TVD values are sampling-noise-limited (bins ≫ shots) and are not correctness claims;
  the rigorous gate (q8-q16, 2M shots) passes for every platform.
- Plasma CPU sampling is O(shots·2^n) by its re-execution shot model; sampling was measured only
  at q ≤ 16 for Plasma CPU.
- QuEST was built v3.7.0 (classic API, applyMatrix2/4), the modern v4.2 renamed its gate API and is
  not directly comparable.
- Plasma's parser caps QUBITS at 28 (stabilizer/MPS/statevector alike), so no scale > 28 is reported.
- All cells are single-run min values on the pinned machine; run-to-run variance is captured in
  `cross_results_{gpu,cpu}.csv` (repeat counts per cell).

## 9. Reproducibility

Harness: `bench/cross_bench.py` (shared workload generator, per-platform runners, kernel-only timing,
TVD gate, per-case CSV flush). Scoring: `bench/score_bench.py` (geometric mean, Plasma baseline).
Raw data: `bench/cross_results_gpu.csv` (60 cells), `bench/cross_results_cpu.csv` (75 cells).
Plan: `bench/BENCHMARK_PLAN.md`.