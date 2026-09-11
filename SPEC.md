# Plasma Language Specification

Copyright (c) 2026 Faris Alfarhan

Licensed GNU GPL version 3.

Version 0.2.1. Draft; the language is subject to change. This specification
describes the current language. Plasma is a superset of its v0.1 form, and
the backward-compatibility rule (§9) is normative: everything specified here
is guaranteed to keep its meaning in every later version.

For the project's engineering and architecture, see
[PROJECT.md](PROJECT.md).

## 1. Model

A Plasma program operates on `n` qubits and up to `MAX_CLASSICAL` classical
bits.

- The quantum state is a vector of `2^n` complex amplitudes, initialised to
  `|0...0>`.
- Unitary gates evolve the state deterministically.
- `MEASURE` collapses the state probabilistically and writes a classical bit.
- Classical bits drive `IF`, classical computation, and the shot loop.
- All randomness comes from a host-side seeded RNG; the language is
  deterministic given a seed.

## 2. Lexical structure

- Line-based: statements end at a newline.
- Keywords are uppercase and case-sensitive. Identifiers may contain `_`
  (used by `MEASURE_X`, `MEASURE_Y`).
- `//` begins a comment that runs to the end of the line.
- Integers are unsigned decimals. Angles are unsigned decimals (a decimal
  point is optional).
- `c<n>` references classical bit `n`.
- Multi-character symbols: `->`, `==`, `!=`.

Keywords: `QUBITS H X CNOT TOFF RZ RX RY PHASE S T SX SWAP ISWAP CZ CPHASE CSWAP
MCX RESET MEASURE MEASURE_X MEASURE_Y IF THEN ENDIF SHOT ENDSHOT GATE ENDGATE
PRINT SET NOT AND OR XOR ADD SUB END HALT`.

## 3. Grammar (EBNF)

```
program     := qubits newline body end-marker
qubits      := "QUBITS" integer
body        := { newline | line }
line        := gate | measure | reset | classical | expect | estimate | save
             | if-stmt | shot-stmt | gate-def | gate-call | "PRINT"
gate        := "H" q | "X" q | "CNOT" q q | "TOFF" q q q
             | ("RZ" | "RX" | "RY" | "PHASE") q angle
             | ("S" | "T" | "SX") q
             | ("SWAP" | "ISWAP" | "CZ") q q
             | "CPHASE" q q angle
             | "CSWAP" q q q
             | "MCX" mask q
measure     := ("MEASURE" | "MEASURE_X" | "MEASURE_Y") q { q } [ "->" c { c } ]
reset       := "RESET" q
classical   := "SET" c int
             | "NOT" c
             | ("AND" | "OR" | "XOR" | "ADD" | "SUB") c c
expect      := "EXPECT" pauli
estimate    := "ESTIMATE" coefficient pauli { coefficient pauli }
save        := "SAVE_STATEVECTOR" | "SAVE_AMPLITUDES" | "SAVE_PROBABILITIES"
if-stmt     := "IF" c ("==" | "!=") byte "THEN" body "ENDIF"
shot-stmt   := "SHOT" integer [ body "ENDSHOT" ]
gate-def    := "GATE" name integer body "ENDGATE"
gate-call   := name q { q }
end-marker  := "END" | "HALT" | <eof>
byte        := integer          -- classical value, 0 <= v <= 255
coefficient := [ "-" ] float    -- signed decimal (ESTIMATE)
name        := identifier       -- any non-keyword word, case-sensitive
q           := integer          -- qubit index, 0 <= q < num_qubits
c           := "c" integer      -- classical bit index
angle       := float            -- radians
mask        := integer          -- u32 control mask (MCX)
pauli       := ( "I" | "X" | "Y" | "Z" ) integer { ( "I" | "X" | "Y" | "Z" ) integer }
```

## 4. Semantics

### 4.1 QUBITS

Declares `n` qubits and creates `n` classical bits, all zero.
`1 <= n <= MAX_QUBITS`.

### 4.2 Gates

Let `bit(k) = 1 << k`. Gates are applied in program order. A "single-writer
pair" `(i, j)` has `j = i ^ bit(q)` and `i & bit(q) == 0`.

- **H q**, Hadamard. `a_i' = (a_i + a_j)/√2`, `a_j' = (a_i - a_j)/√2`.
- **X q**, Pauli-X. Swaps each amplitude pair.
- **RZ q θ**, `diag(e^{-iθ/2}, e^{iθ/2})`.
- **RX q θ**, `[[cos(θ/2), -i·sin(θ/2)], [-i·sin(θ/2), cos(θ/2)]]`.
- **RY q θ**, `[[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]`.
- **PHASE q θ**, `diag(1, e^{iθ})`.
- **S q**, `diag(1, i)`. **T q**, `diag(1, e^{iπ/4})`.
- **SX q**, `sqrt(X) = (1/2)[(1+i)·I + (1-i)·X]`.
- **CNOT c t**, controlled-X: swaps pair `(i, i ^ bit(t))` when bit `c` set.
- **TOFF c1 c2 t**, Toffoli: swaps when bits `c1` and `c2` are both set.
- **SWAP a b**, swaps the amplitudes of qubits `a` and `b`.
- **ISWAP a b**, swaps with an `i` phase on the exchanged term.
- **CZ a b**, phase `-1` on `|11>`.
- **CPHASE a b θ**, phase `e^{iθ}` on `|11>`.
- **CSWAP a b c**, Fredkin: `|a>` controls the swap of `b` and `c`.
- **MCX mask t**, multi-controlled-X: flips `t` when every bit of `mask` is
  set. `mask != 0`, `mask < 2^num_qubits`, and the target bit must not be in
  the mask.

### 4.3 MEASURE

`MEASURE q [-> c]`, `MEASURE_X q [-> c]`, `MEASURE_Y q [-> c]`, and the
multi-qubit forms `MEASURE q1 q2 ... [-> c1 c2 ...]` (same for X/Y):

1. Born rule: `P(1) = sum(|a_i|^2)` over amplitudes with bit `q` set.
2. The outcome is drawn from the seeded host RNG: `outcome = 1` if
   `r < P(1)`, else `0`.
3. Collapse: amplitudes whose bit `q` does not match the outcome are zeroed;
   the surviving branch is renormalised by `1/sqrt(P)` where `P` is the
   probability of the surviving branch.
4. The outcome is stored in classical bit `c`, or in bit `q` if no target
   is given. In a multi-qubit measure, each qubit-target pair is emitted as a
   separate `MEASURE`; the arrow target list must match the qubit list length.
   Multi-qubit `MEASURE` is syntactic sugar for sequential single-qubit
   measurements in left-to-right order, so each earlier collapse affects the
   probabilities of the later ones.

`MEASURE_X` measures in the X basis (rotate with `H`, measure Z, rotate
back). `MEASURE_Y` measures in the Y basis (rotate with `S†`, `H`, measure Z,
rotate back with `H`, `S`).

### 4.4 RESET

`RESET q` collapses `|q>` to `|0>` (measure, then apply `X` if the outcome
was `1`). No classical bit is written.

### 4.5 IF

`IF c == v THEN body ENDIF` executes `body` iff the byte `c` equals `v`
(`0 <= v <= 255`). `IF c != v` executes `body` iff `c != v`. Bodies nest up to
the nesting limit.

### 4.6 Classical computation

Classical bits are unsigned bytes (`0..=255`). They are integers, not
booleans: every operation treats them as 8-bit values, and `IF` compares the
full byte. The boolean idiom is comparing to 0 or 1. Computation executes
host-side on every backend and is deterministic by construction.

- `SET c v`, `c = v` (`0 <= v <= 255`).
- `NOT c`, `c ^= 1` (flips the low bit; on a value in {0,1} this is logical
  negation).
- `AND c1 c2`, `c1 &= c2`. `OR c1 c2`, `c1 |= c2`. `XOR c1 c2`, `c1 ^= c2`.
- `ADD c1 c2`, `c1 = (c1 + c2) mod 256`. `SUB c1 c2`, `c1 = (c1 - c2) mod 256`.

Classical bits are the only interface between measurement outcomes and
subsequent gates.

### 4.7 EXPECT

`EXPECT <pauli>` computes `<ψ|P|ψ>` directly from the amplitudes (no
sampling) for a Pauli product `P = P_{q1} P_{q2} ...` with each `P_q` in
`{I, X, Y, Z}`. Example: `EXPECT Z0X1` is `Z` on qubit 0 tensored with `X` on
qubit 1. The result is a real number appended to the observable results, in
program order. Every qubit in the product must be declared. The result agrees
across backends to float precision (bit-identical when the value is exactly
representable).

### 4.8 ESTIMATE (Estimator)

`ESTIMATE c0 P0 c1 P1 ...` evaluates the weighted Pauli sum (a Hamiltonian
expectation) `<H> = c0 <P0> + c1 <P1> + ...` directly from the amplitudes,
where each `P_i` is a Pauli product and each `c_i` a (possibly negative)
decimal coefficient, e.g. `ESTIMATE 0.5 Z0Z1 0.3 X0X1 -0.2 Y0Y1`. One real
value is appended to the observable results per `ESTIMATE`, in program order.
Terms are stored in a heap-free pool (`MAX_ESTIMATE_TERMS`); a term may not
reference undeclared qubits.

### 4.9 SAVE_*

- `SAVE_STATEVECTOR`, save the current statevector (re, im interleaved).
- `SAVE_AMPLITUDES`, save the current amplitudes as `(index, re, im)`.
- `SAVE_PROBABILITIES`, save the current basis probabilities `|amp|^2`.

Each captures the state at the point it executes, appended to the observable
results.

### 4.10 SHOT

- **Bare** `SHOT n` (no body): the body is the sequence of operations that
  precede the `SHOT`. It is executed `n` times, resetting the state to
  `|0...0>` each iteration.
- **Body** `SHOT n ... ENDSHOT`: the explicit body is executed `n` times,
  resetting each iteration.
- Each iteration samples one basis state from the final state per the Born
  rule (inverse cumulative distribution) and increments that histogram bin.
- Classical bits persist across iterations.

### 4.11 PRINT

Emits the accumulated histogram.

### 4.12 Subroutines (jump-based)

`GATE name <nparams>` ... `ENDGATE` defines a reusable subcircuit whose body
is a sequence of gate ops (including nested calls and `IF` blocks) that treat
qubits `0..nparams-1` as formal parameters. A call is any other line
beginning with the gate's name:

```text
GATE bell 2
H 0
CNOT 0 1
ENDGATE
bell 1 2        # call with arguments (qubits 1, 2)
```

- `nparams` must be in `1..=3`; a call supplies exactly `nparams` qubit
  arguments, bound positionally to the body's qubit operands (param `k` gets
  argument `k`).
- Bodies live in a sub-program region of the flat op array, separate from
  the main executable region, and are reached only by the `CALL` op, the
  executor jumps into the body, remapping qubit operands through the call's
  arguments, and returns to the caller. No code is duplicated.
- Calls are legal inside sub-program bodies and compose (nested calls bind
  through the enclosing call's mapping).
- `GATE` definitions may appear anywhere in the program body before the end
  marker; the main executable region excludes all sub-program bodies.
- A `GATE` name must not be a keyword (names are case-sensitive and keywords
  are uppercase), so a user-defined gate cannot shadow a built-in gate. A call
  to an undefined name is a diagnostic error.
- Programs containing `CALL` are not fused (the fused stream cannot host sub
  bodies); `--fuse` transparently falls back to the regular executor.

### 4.13 END / HALT

`END` and `HALT` terminate the program. They are also the end marker: the
parser reads the program body up to the first `END`, `HALT`, or end-of-file,
so any content after them is ignored (never parsed or executed), and `GATE`
definitions must appear before them.

## 5. IR bytecode

The parser lowers source to a flat, versioned bytecode (`IrOp`). The IR is
the language's canonical form and the wire contract between the front-end and
the backends (and, later, the planckOS shim).

- `IR_VERSION = 2`.
- Ops (v0.1, unchanged): `H X CNOT TOFF MEASURE IFEQ SHOT PRINT`.
- Ops (v0.2, additive): `RZ RX RY PHASE` (qubit + constant-pool index),
  `S T SX SWAP ISWAP CZ CPHASE CSWAP MCX RESET MEASUREX MEASUREY`
  `SET NOT AND OR XOR ADD SUB`; observables `EXPECT(u64 pauli)` (2 bits per
  qubit: I=00, X=01, Y=10, Z=11) and `SAVESTATE SAVEAMPS SAVEPROBS`.
- Parametric-gate angles live in a heap-free `f64` constant pool
  (`MAX_CONSTS` entries); ops reference angles by index.
- Control flow (`IFEQ`, `IFNE`, `SHOT` bodies) is encoded as `(offset,
  length)` ranges into the flat op array.
- `IF c == v` is lowered to `IFEQ(c, v, offset, length)`; `IF c != v` is
  lowered to the dedicated `IFNE(c, v, offset, length)` op.
- Multi-qubit `MEASURE` is lowered to one `MEASURE` op per qubit.
- Subroutines: `CALL(sub_id, a0, a1, a2)` jumps into `Program.subs[sub_id]`
  (`(off, len, nparams)` pointing into the sub-program region
  `ops[len .. sub_len)`), binding the body's qubit operands `0..nparams-1` to
  the call's arguments; unused argument slots are `NO_ARG = 0xFF`.
- The IR is fixed-capacity and heap-free.

## 6. Validation and limits

- Qubit operands must satisfy `0 <= q < num_qubits`.
- Classical operands must satisfy `c < MAX_CLASSICAL`.
- `MCX` masks are validated (nonzero, in range, disjoint from the target).
- Limits: `MAX_QUBITS = 28`, `MAX_OPS = 16384`, `MAX_TOKENS = 16384`,
  `MAX_CLASSICAL = 64`, `MAX_CONSTS = 1024`, `MAX_SUBS = 128`, nesting depth
  `<= 32`, gate names `<= 16` bytes.
- Violations produce diagnostics with line and column, never silent failure.

## 7. Backends and determinism

- **CPU**: reference interpreter (`src/sim.rs`), including a fused-execution
  path (`--fuse`) over `src/fusion.rs` output.
- **GPU**: PTX kernels loaded through the CUDA driver API
  (`src/gpu/`). Supports the full v0.2 op set; batchable programs use the
  pure-batch and shared-memory mega-kernel fast paths, and all other programs
  run through the GPU general executor.
- Determinism contract:
  - Given a program and a seed, output is reproducible run-to-run on the same
    backend (the GPU reduction order is fixed, never atomic-order dependent).
  - Exact (non-sampled) outputs agree across backends.
  - Sampled histograms are statistically identical across backends; where
    amplitudes are exactly representable they are bit-identical.
- All randomness comes from the host-side seeded RNG; the GPU is a pure
  linear-algebra coprocessor.
- **Noise models** (`--noise-*`, `CLI`): sampled-Kraus channels applied after
  each qubit gate on the (non-fused) CPU reference path, single-qubit
  depolarizing (`--noise-depolarizing p`), amplitude damping
  (`--noise-amp-damping p`, correct pure-state sampled form with
  renormalization), phase damping (`--noise-phase-damping p`), and readout
  error (`--noise-readout p`, the measurement report draws from the noisy
  distribution). Noise draws come from the seeded RNG in a fixed per-gate
  order, so noisy runs are bit-reproducible; noise is off by default and forces
  the CPU reference.
- **Readout mitigation** (`--mitigate-readout`): linear-inversion correction of
  the shot histogram using the single-qubit confusion matrix
  `[[1-p, p], [p, 1-p]]` (independent per qubit, tensor-inverse applied with a
  per-qubit transform), requiring `0 < --noise-readout < 0.5`.

## 8. Result model

The CLI reports the accumulated histogram (counts and percentages per basis
state, `|q_{n-1}...q_0>` ordering) and the observable results captured by
`EXPECT` and `SAVE_*` ops (expectation values, saved statevectors,
amplitudes, probabilities). Two output formats are supported: text
(default) and JSON (`--format json`), the latter being the
machine-readable form: `{"plasma": {...}, "histogram": {...}, "results": {...}}`.

## 9. The backward-compatibility rule (normative)

Plasma evolves by superset only. This is a rule of updating, not an
optional feature:

1. **Additive semantics.** Every program that runs under any version runs
   under every later version and produces the same output for the same seed.
   New constructs are additions; existing constructs are never reinterpreted.
   There is exactly one unified executor, no version forks. Old op encodings
   keep their exact meaning and behavior forever.
2. **Invariant A, exact results are bit-stable.** Mathematically-determined
   outputs, gate evolution, probabilities, expectation values, statevectors,
   deterministic-basis histograms, are bit-identical across all future
   versions. If the format of an output changes but the mathematical result
   is identical, it is correct. Golden-value tests enforce this.
3. **Invariant B, determinism.** Given a program and a seed, output is
   reproducible run-to-run on the same backend. A change to the RNG is
   permitted only if determinism is upheld and only if exact (non-sampled)
   results are unaffected. Histogram sampling is inherently random: sampled
   outputs may differ across versions under an approved RNG change; exact
   outputs never may.
4. **Golden enforcement.** The `tests/golden.py` harness compares values,
   not text, and treats format changes with identical math as correct.

## 10. Planned (non-normative)

These are specified for a future version and are not yet implemented:

- **Multi-GPU statevector** (see `PROJECT.md`; needs 2+ GPUs to verify).