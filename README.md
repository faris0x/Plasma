# Plasma

Copyright (c) 2026 Faris Alfarhan

Licensed GNU GPL version 3.

## What is Plasma?

A quantum assembly language and deterministic statevector simulator, built for
planckOS and developed independently until the planckOS GPU shim is complete.
The name is a portmanteau of Planck and ASM. Source files use the `.qs0`
extension (FAT32 8.3 short filenames).

## Quick start

```
cargo build --release
plasma run <file.qs0> [--gpu] [--seed N] [--shots N] [--bench]
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

Gates: `H`, `X`, `CNOT`, `TOFF`. Control flow: `MEASURE`, `IF`/`ENDIF`,
`SHOT`/`ENDSHOT`, `PRINT`, `END`/`HALT`. See [SPEC.md](SPEC.md) for the full
semantics.

## Backends

- CPU: deterministic reference interpreter (`src/sim.rs`).
- GPU: hand-written PTX kernels loaded through the CUDA driver API
  (`src/gpu/`). No CUDA C, no nvcc, no NVRTC.

## Design principles

- Determinism is king: measurement outcomes and histogram samples come from a
  host-side seeded RNG, never the GPU.
- `no_std`-ready core so the frontend can move into the planckOS kernel later.
- PTX-only GPU path so no CUDA compiler ships on the target.
