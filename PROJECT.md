# Plasma Project Specification

Copyright (c) 2026 Faris Alfarhan

Licensed GNU GPL version 3.

Version 0.2.

This document specifies the project: architecture, engineering decisions, and
determinism machinery. The language, including syntax, semantics, IR, and
limits, is specified separately in [SPEC.md](SPEC.md).

## 1. Overview

Plasma is a high-level quantum computing language and deterministic quantum
circuit simulator, developed independently and bundled with planckOS, a
from-scratch x86_64 operating system whose application is a quantum-computing
workstation. The latest Plasma is shipped with each planckOS release. The name
is a portmanteau of Planck and ASM (its syntax is line-based and
assembly-like). Source files use the `.qs0` extension (FAT32 8.3 short
filenames).

The design goals:

- Given a program and a seed, output is reproducible. All randomness comes from
  a host-side seeded RNG; the GPU is a pure linear-algebra coprocessor.
- The frontend (lexer, parser, IR) is heap-free so it can move into the
  planckOS kernel later. The CPU backend runs on `no_std` plus
  `extern crate alloc`.
- Hand-written PTX kernels are JIT-compiled by the CUDA driver at runtime, with
  no CUDA C, no nvcc, no NVRTC, and no CUDA toolkit on the target.

## 2. Architecture

```
plasma.qs0
   │  read
   ▼
┌─────────────────────────┐   ┌────────────────────┐
│ frontend (no_std core)  │   │                    │
│  lexer → parser → IR v2 │──▶│  CPU backend       │  sim.rs (reference)
│  (heap-free, fixed caps)│   │  + fused executor  │  sim.rs + fusion.rs
└─────────────────────────┘   │                    │
         │                    └────────────────────┘
         ▼
   ┌─────────────┐            ┌────────────────────┐
   │ fusion.rs   │──(later)──▶│  GPU backend       │  gpu/ (hand PTX +
   │ FusedOp stream           │  driver API FFI    │   CUDA driver FFI)
   └─────────────┘            └────────────────────┘
```

- **Frontend** (`lexer.rs`, `parser.rs`, `ir.rs`): line-based parser lowering
  source to a flat, versioned `IrOp` array (`IR_VERSION = 2`) plus an `f64`
  constant pool for parametric angles. Control flow (`IF`, `SHOT` bodies) is
  encoded as `(offset, len)` ranges into the flat array; the parser uses a
  reserve-and-patch pattern, so the IR is heap-free and kernel-embeddable.
- **CPU backend** (`sim.rs`): the reference interpreter. `f64` SoA state
  (separate `re`/`im` vectors), in-place single-writer gate butterflies.
- **Fused executor** (`fusion.rs` + `sim.rs`): a backend-agnostic gate-fusion
  pass producing a `FusedOp` stream consumed by `--fuse` (see §5).
- **GPU backend** (`gpu/mod.rs`, `gpu/kernels.ptx`): raw CUDA driver FFI
  (zero crates), PTX kernels, host-side sampling.
- **CLI** (`main.rs`): `plasma run <file.qs0> [--gpu] [--seed N] [--shots N]
  [--bench] [--dump-statevector] [--fuse]`, with per-phase microsecond timing.

## 3. Determinism engineering

The determinism guarantees are normative in SPEC.md §7 and §9 (Invariants A
and B). This section describes how the machinery upholds them.

- **Host-side RNG.** Measurement outcomes and shot samples are drawn from the
  seeded `Lcg64` (MMIX variant, period 2^64) on the host.
- **Deterministic GPU reduction.** GPU probabilities are computed by a
  two-stage fixed-order reduction: `prob_partial` writes one partial per block
  to `scratch[block_id]` (thread 0 sequentially sums per-thread partials in
  thread order), then `prob_finalize` sums the partials in block order. The GPU
  is bit-reproducible run-to-run for a given program.
- **Exact results bit-stable.** Golden statevectors and deterministic-basis
  histograms are recorded in `tests/golden/` and verified by
  `tests/golden.py`, which compares values (not text) so format changes with
  identical math pass. A change that alters what an old program produces is a
  build failure.
- **Cross-backend agreement.** CPU and GPU produce bit-identical histograms
  wherever amplitudes are exactly representable (structured circuits) and
  statistically identical (TVD within the sampling-noise floor) on deep
  random circuits, where f32-vs-f64 rounding moves individual samples between
  adjacent bins.
- **Fixed bugs with determinism relevance:**
  - *Uninitialized device memory*: the GPU single-execution path previously
    ran on whatever `cuMemAlloc` left in the buffer. Fixed by explicit reset
    and a `cuMemsetD8` zero at init.
  - *Pure-batch SHOT-body handling*: the single-transfer fast path now applies
    only to programs with at most one SHOT (`is_batchable`), sampling the
    correct unitary region for bare and body shots.
  - *Hand-rolled `f64::sqrt`*: kept, because `core::f64` gates out all
    transcendental methods on this `no_std` toolchain (see §4).

## 4. Deterministic float math

`core::f64` on this toolchain provides no transcendental methods: `sqrt`,
`sin`, `cos`, `round`, `floor`, `trunc`, `rem_euclid` are all gated out
(only `abs`, casts, and bit operations are available). The math in
`src/math.rs` is therefore deterministic and heap-free:
faithful f64 ports of the algorithms in my
[NumCore](https://github.com/NumCore/NumCore/tree/main/numcore/src/math) firmware. They are self-contained copies
(algorithm, not a dependency), keeping Plasma and planckOS independent:

- **`sin_cos`**: 22-iteration CORDIC rotation mode plus a first-order Taylor
  correction on the residual angle, with quadrant folding. Worst-case error
  ~2.3e-13 (measured 1.5e-13). Pure shift/add, bit-deterministic on every
  platform. Powers all parametric-gate rotation math.
- **`sqrt`**: CLZ + 32-entry LUT initial guess for `1/√x`, 3 Newton-Raphson
  iterations, then one final Newton step refining `√x` directly. Relative
  error ~4e-16 in f64 (machine-ε; the f64 Newton converges tighter than
  the fixed-point bound). Exponent scaling is done via a bit-built `2^k` so it
  handles the full f64 range (`sqrt(1e300)` and `sqrt(1e-300)` correct).

Both are validated by unit tests and by the sin^2+cos^2 identity sweep.

## 5. Gate fusion

`fusion.rs` lowers the flat IR into a `FusedOp` stream:

- **U1 / U2 blocks**: straight-line runs of single-qubit gates on one qubit
  collapse to a 2×2 unitary; runs on the same two-qubit set collapse to a
  4×4 unitary (built in the `s = bit(q0) + 2·bit(q1)` basis).
- **Control flow preserved**: `IF` / `SHOT` bodies are fused recursively and
  their `(offset, len)` ranges are re-patched against the fused list.
- **Pass-through**: measurements, multi-qubit gates (Toffoli, CSWAP, MCX),
  reset, and classical computation flush the block and pass through unchanged.
- **Guarantees**: the fused stream never grows the op count; fused execution
  agrees with the unfused reference within float tolerance (evaluation order
  changes, so not bit-for-bit). Verified by agreement tests (state <1e-9,
  histogram within sampling tolerance).

Measured: on a gate-dense same-pair circuit, 601 gates fuse to 2 fused ops and
a ~325× CPU speedup (854 ms to 2.6 ms for 100k shots). On random 10-qubit
circuits the op count barely drops (752 to 686) because there are no long
same-pair runs; there the primary GPU benefit is kernel-launch elimination
(686 launches to one persistent kernel), not fewer operations.

## 6. GPU backend

- **Interface**: raw CUDA driver API (`cuInit`,
  `cuModuleLoadData`, `cuLaunchKernel`, ...); the PTX module (`kernels.ptx`,
  479 lines, PTX ISA 8.6 / sm_80) is embedded with `include_bytes!` and
  JIT-compiled by the driver. No CUDA C, no nvcc, no NVRTC, no toolkit.
- **Kernels**: single-writer gate butterflies (each amplitude pair is written
  exactly once), grid-stride loops with `mad.lo.u32` thread-id math, f32
  interleaved AoS state.
- **Pure-batch fast path**: for pure-unitary programs with at most one SHOT,
  gates are applied once, the final state is transferred once, and all samples
  are drawn host-side via binary search over a cumulative probability array:
  identical RNG stream to the general path, one transfer instead of N.
- **Shared-memory mega-kernel (M4-P2)**: for states up to 2^13 amplitudes
  (64 KB, within the sm_120 opt-in dynamic-shared limit), the entire
  pure-unitary gate sequence runs in ONE block in shared memory with only
  intra-block `bar.sync` between gates: one launch, no global round-trips, no
  grid barrier. Bit-identical to the per-gate path (same f32 arithmetic).
  Measured: 10q/752 gates 1006 µs to 359 µs; 12q/800 gates 662 µs; 13q/600
  gates 750 µs, all one launch. Larger states fall back to per-gate kernels.
  `PLASMA_NO_MEGA=1` disables it for A/B testing.
- **Observables (M5)**: `EXPECT <pauli>` computes `<ψ|P|ψ>` on the GPU via
  `expect_partial`/`expect_finalize` kernels, the same fixed-order
  two-stage reduction as probabilities, so expectation values are
  bit-reproducible run-to-run. `SAVE_STATEVECTOR/AMPLITUDES/PROBABILITIES`
  capture the state via a device-to-host transfer. Results are collected in a
  `Results` struct threaded through both backends.
- **Transfer pipeline (M6)**: a pinned host staging buffer (`cuMemHostAlloc`,
  one allocation at init, reused across shots) plus async
  `cuMemcpyDtoHAsync` + `cuStreamSynchronize` replaces the per-call sync
  copy into a pageable `Vec`. Measured on a 512 MB state: 191 ms to 18.5 ms
  (~10×). Falls back to sync copies if pinned allocation fails.
  `PLASMA_NO_PINNED=1` disables it for A/B testing.
- **Adaptive precision (M8)**: `estimate_f32_amp_error` models the f32 GPU
  amplitude drift as `~4*ε_f32*sqrt(gates)` (random-walk accumulation);
  `--precision auto` uses the GPU f32 fast path only when the estimate is
  within `--tolerance` (default 1e-5), else falls back to the f64 CPU
  reference with a diagnostic (`f32`/`f64` force either). Validated by the
  golden harness: observed CPU/GPU drift respects the model bound.
  The GPU probability and Pauli-expectation reductions now use Kahan
  (compensated) summation at every level (per-thread, per-block, finalize),
  removing the O(sqrt(n)) f32 summation drift: on a 2^18-term expectation
  the naive f32 sum drifted 2.3e-5 while the compensated sum matched the
  exact value; the kernel output exactly equals the compensated sum of its
  f32 state, leaving only the f32 amplitude rounding (~1e-7) vs the f64
  reference.
- **Tensor-core `mma` experiment (M7-A, verified, measured slower, not wired
  in)**: `mma_u2_lowpair` applies a fused 4×4 unitary to a state using
  `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`, with the complex
  product done as two real GEMMs (`A1=[U_r|-U_i]`, `A2=[U_i|U_r]`,
  `B=[in_r;in_i]`). The fragment layouts come directly from the PTX ISA
  tables. Verified against the CPU reference to TF32 precision (2.5e-4
  relative) on CNOT and random unitaries, with zero compute-sanitizer errors.
  However, on a 4M-amplitude state it measured 106.6 Gamp/s vs 145.4
  Gamp/s for the FMA 4×4 kernel: the 4×4 gate underuses the 16×8 mma tile
  (25% M-utilization) and the state is memory-bound, so the tensor-core path
  is ~1.36× slower and loses precision to TF32's 10-bit mantissa. The FMA
  path remains the default; the mma kernel is kept as a documented,
  verified-correct experimental kernel.
- **Grid-barrier experiment (measured, not adopted)**: a cooperative grid
  barrier (monotonic phase counter) costs ~0.87 µs at 4 blocks and ~1.86 µs
  at 144 blocks, worse than per-gate launches except at the smallest grids,
  so a whole-circuit kernel with a grid barrier was rejected in favour of the
  shared-memory mega-kernel, which needs no grid barrier at all.
- **Current limitation**: one kernel launch per gate beyond 2^13 amplitudes
  (per-gate path); states that fit shared memory use the mega-kernel. A
  persistent multi-block kernel with grid barriers was measured and rejected
  (barrier cost exceeds the win at these sizes).

## 7. Benchmarking

`bench/bench.py` (gitignored corpus + venv) compares Plasma GPU against
Qiskit Aer GPU (`AerSimulator(method="statevector", device="GPU")`), built
from source against CUDA 13. The current release wheel ships against CUDA 12,
so giving Aer the newer toolchain removes any handicap. The harness is honest:

- **Noise-gated TVD**: CPU/GPU agreement is judged by total-variation distance
  against Aer's own run-to-run sampling noise (`ok = tvd < 3·noise + 0.02`),
  not naive equality.
- **Accuracy**: `bench/accuracy.py` compares exact statevector amplitudes
  (f32 Plasma vs f64 Aer) on deep circuits via `--dump-statevector`.
- **Recorded results** (`bench/results.csv`, 35 cases): Plasma wins by 3-10×
  on 10-16q and on deep 20-28q circuits; it loses on large shallow circuits
  where the state-transfer wall dominates, a documented, understood trade-off
  that the pure-batch optimization targets.

## 8. Feature backends

- **Stabilizer engine**: `src/stabilizer.rs` performs exact CH/tableau
  simulation of Clifford circuits (H/S/X/CNOT/CZ/SWAP plus measure/expect and
  control flow). The `--stabilizer` flag adaptively runs any Clifford-only
  program on it (with a diagnostic when it falls back to the statevector); for
  the same seed, EXPECT values are exact and shot histograms statistically
  identical to the reference. The executor is backend-generic (`SimBackend`
  trait). Validated on 60 randomized Clifford circuits plus focused
  measurement tests.
- **Noise models**: sampled-Kraus depolarizing, amplitude damping (correct
  pure-state form), phase damping, and readout error (`--noise-*`);
  deterministic (seeded RNG, fixed draw order), forces the CPU reference.
  Readout mitigation inverts the per-qubit confusion matrix
  (`--mitigate-readout`).
- **Estimator**: `ESTIMATE c0 P0 c1 P1 ...` computes the weighted Pauli sum
  directly from amplitudes (CPU/GPU exact agreement).
- **MPS engine**: `src/mps.rs` performs matrix-product-state simulation
  (`--mps D`) with truncated-SVD re-splitting (exact 2x2 Hermitian block
  diagonalization), SWAP-based qubit routing with permutation
  tracking, and exact marginal sampling via bra-ket sandwich contraction.
  Exact for low-entanglement circuits (validated: EXPECT bit-identical,
  histograms statistically identical on Bell/chain/measurement circuits);
  SVD truncation error is tracked and reported. Known limitation: the deep
  conditional sampling of highly-entangled routed circuits can be inaccurate
  (the state and first conditionals are exact); the statevector is the
  reference for exact results.
- **Multi-GPU statevector (design)**: split the statevector across N GPUs
  (each owns a contiguous slab of amplitudes); the single-qubit butterfly
  becomes an all-to-all exchange via peer-to-peer copies, and two-qubit gates
  between slab boundaries require the boundary pair exchanged each step. Not
  implemented: the machine has a single GPU (RTX 5070), so an untested
  distributed path would violate the correctness contract. The GPU backend's
  fixed-order reductions and host-side sampling carry over unchanged.
