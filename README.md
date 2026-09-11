# Plasma

Copyright (c) 2026 Faris Alfarhan

Licensed GNU GPL version 3.

Version 0.2.1.

## What is Plasma?

A high-level quantum computing language and deterministic quantum circuit
simulator, developed independently and bundled with planckOS, a from-scratch
x86_64 operating system whose application is a quantum-computing workstation.
The latest Plasma is shipped with each planckOS release. The name is a
portmanteau of Planck and ASM (its syntax is line-based and assembly-like).
Source files use the `.qs0` extension (FAT32 8.3 short filenames).

Two documents describe Plasma:

- [SPEC.md](SPEC.md), the language specification (syntax, semantics, IR,
  limits, backward-compatibility rule).
- [PROJECT.md](PROJECT.md), the project specification (architecture,
  determinism engineering, math, GPU backend, benchmarks).

## Quick start

```
cargo build --release
plasma run <file.qs0> [--gpu] [--seed N] [--shots N] [--bench]
                          [--fuse] [--stabilizer] [--mps D]
                          [--noise-depolarizing P] [--noise-readout P]
                          [--noise-amp-damping P] [--noise-phase-damping P]
                          [--mitigate-readout] [--precision auto|f32|f64] [--tolerance T]
                          [--dump-statevector] [--format text|json]
```

## Language

Line-based, uppercase keywords, `//` comments:

```
QUBITS 2
H 0
CNOT 0 1
MEASURE 0 -> c0
IF c0 == 1 THEN
  X 0
ENDIF
SHOT 100
PRINT
```

Gates: `H X CNOT TOFF`, rotations `RZ RX RY PHASE` (with angles), exact
`S T SX`, entanglement `SWAP ISWAP CZ CPHASE CSWAP MCX`. State handling:
`RESET`, `MEASURE_X`/`MEASURE_Y`, multi-qubit `MEASURE`. Control flow:
`IF`/`ENDIF` (with `==`/`!=`), `SHOT`/`ENDSHOT`, `PRINT`, `END`/`HALT`.
Classical computation: `SET NOT AND OR XOR ADD SUB`. See
[SPEC.md](SPEC.md) for the full semantics.

## Examples

Worked programs live in `examples/`, each with a step-by-step explanation:

- `GHZ.QS0` - a three-qubit Greenberger-Horne-Zeilinger state (entanglement
  and measurement correlation; also runs on the stabilizer engine).
- `QFT4.QS0` - the four-qubit quantum Fourier transform (rotations, CPHASE,
  SWAP, phase encoding).
- `BELLEST.QS0` - a Bell state verified through the `ESTIMATE` estimator.
- `MEASXY.QS0` - X/Y-basis measurement and mid-circuit `RESET`.
- `SUBRT.QS0` - reusable subcircuits via `GATE`/`ENDGATE` and `CALL`.
- `TOFFCTL.QS0` - Toffoli, multi-controlled-X, Fredkin, and classical control.
- `CLIFF.QS0` - a Clifford-only circuit for the stabilizer engine.
- `CHAIN.QS0` - a low-entanglement chain for the MPS engine.
- `NOISEQ.QS0` - depolarizing/readout noise and readout mitigation.

## Backends

- CPU: deterministic reference interpreter (`src/sim.rs`), plus a fused
  execution path (`--fuse`) over the gate-fusion pass (`src/fusion.rs`).
- Stabilizer engine (`--stabilizer`): exact CH/tableau simulation of
  Clifford-only circuits (H/S/X/CNOT/CZ/SWAP/ISWAP) (`src/stabilizer.rs`).
- MPS engine (`--mps D`): matrix-product-state simulation with truncated-SVD
  re-splitting and O(n×D^3) marginal sampling for low-entanglement circuits
  (`src/mps.rs`).
- Noise models (`--noise-*`): sampled-Kraus depolarizing / amplitude damping /
  phase damping / readout error, deterministic per seed; readout error
  mitigation via `--mitigate-readout`.
- Estimator: `ESTIMATE c0 P0 c1 P1 ...` evaluates a weighted Pauli sum.
- GPU: PTX kernels loaded through the CUDA driver API
  (`src/gpu/`). No CUDA C, no nvcc, no NVRTC. Supports the full v0.2 op set.

## Design principles

- Given a program and a seed, output is reproducible. Measurement outcomes and
  histogram samples come from a host-side seeded RNG, never the GPU.
- The frontend is `no_std`-ready so it can move into the planckOS kernel later.
- The GPU path is PTX-only, so no CUDA compiler ships on the target.
## Requirements

### Operating system

Linux (x86_64). The binary links the standard glibc libraries and the NVIDIA
CUDA driver.

### Runtime

- **NVIDIA CUDA driver** (`libcuda.so.1`): required to load the binary, even
  for CPU-only runs, because the GPU backend is always linked. CUDA is only
  initialized when `--gpu` is used.
- **No CUDA toolkit at runtime**: the GPU path loads hand-written PTX and the
  driver's embedded JIT compiles it for the actual GPU. No nvcc, NVRTC, or
  headers are needed on the target.

### GPU (optional, for `--gpu`)

Any NVIDIA GPU supported by the installed driver's PTX JIT. The PTX targets
sm_80 and is forward-compatible; verified on Blackwell (sm_120).

### Memory

- CPU statevector (f64): `2^n * 16` bytes; max `n = 28` (~4 GB).
- GPU statevector (f32): `2^n * 8` bytes VRAM; `n = 30` is 8 GB.

Plasma has zero external dependencies (no crates); the core is `no_std` and all
float math is hand-rolled.

### Build

Rust toolchain (cargo/rustc, edition 2021), `libcuda` present to link against,
and `git` for the branch/commit shown by `--version` (falls back to "unknown"
if unavailable).
