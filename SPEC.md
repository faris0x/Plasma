# Plasma Language Specification

Copyright (c) 2026 Faris Alfarhan

Licensed GNU GPL version 3.

Version 0.1 (draft). This is a draft specification; **the language is subject
to change.** Version numbers track planckOS (Plasma v0.1).

## 1. Model

A Plasma program operates on `n` qubits and up to `MAX_CLASSICAL` classical
bits.

- The quantum state is a vector of `2^n` complex amplitudes, initialised to
  `|0...0>`.
- Unitary gates evolve the state deterministically.
- `MEASURE` collapses the state probabilistically and writes a classical bit.
- Classical bits drive `IF` and the shot-based sampling loop.
- All randomness comes from a host-side seeded RNG; the language is
  deterministic given a seed.

## 2. Lexical structure

- Line-based: statements end at a newline.
- Keywords are uppercase and case-sensitive.
- `//` begins a comment that runs to the end of the line.
- Integers are unsigned decimals.
- `c<n>` references classical bit `n`.
- `->` and `==` are the only multi-character symbols.

Tokens: `QUBITS H X CNOT TOFF MEASURE IF THEN ENDIF SHOT ENDSHOT PRINT END
HALT`, integers, classical references, `->`, `==`.

## 3. Grammar (EBNF)

```
program     := qubits newline body end-marker
qubits      := "QUBITS" integer
body        := { newline | line }
line        := gate | measure | if-stmt | shot-stmt | "PRINT"
gate        := "H" q | "X" q | "CNOT" q q | "TOFF" q q q
measure     := "MEASURE" q [ "->" c ]
if-stmt     := "IF" c "==" ( "0" | "1" ) "THEN" body "ENDIF"
shot-stmt   := "SHOT" integer [ body "ENDSHOT" ]
end-marker  := "END" | "HALT" | <eof>
q           := integer          -- qubit index, 0 <= q < num_qubits
c           := "c" integer      -- classical bit index
```

## 4. Semantics

### 4.1 QUBITS

Declares `n` qubits and creates `n` classical bits, all zero.
`1 <= n <= MAX_QUBITS`.

### 4.2 Gates

Let `bit(k) = 1 << k`. All gates are applied in program order.

- **H q** — Hadamard on qubit `q`. For every pair of amplitudes
  `(a_i, a_j)` with `j = i ^ bit(q)` and `i & bit(q) == 0`:
  `a_i' = (a_i + a_j)/sqrt(2)`, `a_j' = (a_i - a_j)/sqrt(2)`.
- **X q** — Pauli-X on qubit `q`. Swaps each amplitude pair
  `(a_i, a_j)` with `j = i ^ bit(q)`.
- **CNOT c t** — controlled-X. Swaps each pair `(i, i ^ bit(t))` for which
  bit `c` of `i` is set.
- **TOFF c1 c2 t** — Toffoli. Swaps each pair `(i, i ^ bit(t))` for which
  bits `c1` and `c2` of `i` are both set.

### 4.3 MEASURE

`MEASURE q [ -> c ]`:

1. Born rule: `P(1) = sum(|a_i|^2)` over amplitudes with bit `q` set.
2. The outcome is drawn from the seeded host RNG: `outcome = 1` if
   `r < P(1)`, else `0`.
3. Collapse: amplitudes whose bit `q` does not match the outcome are set to
   zero; the surviving branch is renormalised by `1 / sqrt(P)`, where `P` is
   the probability of the surviving branch (`P(1)` for outcome 1, `1 - P(1)`
   for outcome 0).
4. The outcome is stored in classical bit `c`, or in bit `q` if no target
   is given.

### 4.4 IF

`IF c == v THEN body ENDIF` executes `body` iff classical bit `c` equals `v`.
Bodies may nest up to the nesting limit.

### 4.5 SHOT

- **Bare** `SHOT n` (no body): the body is the sequence of operations that
  precede the `SHOT`. It is executed `n` times, resetting the state to
  `|0...0>` each iteration.
- **Body** `SHOT n ... ENDSHOT`: the explicit body is executed `n` times,
  resetting each iteration.
- Each iteration samples one basis state from the final state per the Born
  rule (inverse cumulative distribution) and increments that histogram bin.
- Classical bits persist across iterations.

### 4.6 PRINT

Emits the accumulated histogram.

### 4.7 END / HALT

Terminate the program.

## 5. IR bytecode

The parser lowers source to a flat, versioned bytecode (`IrOp`). The IR is
the language's canonical form and the wire contract between the front-end and
the backends (and, later, the planckOS shim).

- `IR_VERSION = 1`.
- Ops: `H`, `X`, `CNOT`, `TOFF`, `MEASURE`, `IFEQ`, `SHOT`, `PRINT`.
- Control flow (`IFEQ`, `SHOT` bodies) is encoded as `(offset, length)`
  ranges into the flat op array.
- The IR is fixed-capacity and heap-free.

## 6. Validation and limits

- Qubit operands must satisfy `0 <= q < num_qubits`.
- Classical operands must satisfy `c < MAX_CLASSICAL`.
- Limits: `MAX_QUBITS = 28`, `MAX_OPS = 16384`, `MAX_TOKENS = 16384`,
  nesting depth `<= 32`.
- Violations produce diagnostics with line and column, never silent failure.

## 7. Backends

- **CPU**: reference interpreter (`src/sim.rs`).
- **GPU**: hand-written PTX kernels loaded through the CUDA driver API
  (`src/gpu/`).
- Both backends must produce statistically identical histograms for a given
  seed.
