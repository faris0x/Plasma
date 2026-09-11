// Copyright (c) 2026 Faris Alfarhan
// SPDX-License-Identifier: GPL-3.0-only

pub use core::num::NonZeroU32;

pub const MAX_OPS: usize = 16384;
pub const MAX_QUBITS: u8 = 28;
pub const MAX_CLASSICAL: u8 = 64;
/// Heap-free constant pool for parametric-gate angles (v0.2).
pub const MAX_CONSTS: usize = 1024;
/// Subprogram table size (jump-based subroutines).
pub const MAX_SUBS: usize = 128;
/// Weighted-Pauli term pool for the `ESTIMATE` (Estimator) op.
pub const MAX_ESTIMATE_TERMS: usize = 256;

/// Version of the flat IR encoding. Bump when the bytecode layout changes;
/// this is the wire contract between the frontend and any backend. v0.2 adds
/// variants additively, existing op encodings and semantics are unchanged.
pub const IR_VERSION: u8 = 2;

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
pub enum IrOp {
    /// Single-qubit Hadamard on |q>
    H(u8),
    /// Pauli-X (NOT) on |q>
    X(u8),
    /// CNOT with |control> controlling |target>
    CNOT(u8, u8),
    /// Toffoli with |c1> and |c2> controlling |target>
    Toff(u8, u8, u8),
    /// Measure |q>, store into |c>
    Measure(u8, u8),
    /// If classical bit |c == val|, execute block at (offset, len)
    IfEq(u8, u8, u16, u16),
    /// If classical bit |c != val|, execute block at (offset, len)
    IfNe(u8, u8, u16, u16),
    /// Repeat block at (offset, len) |count| times, resetting state each iteration.
    /// If len == 0, repeats the entire program.
    Shot(NonZeroU32, u16, u16),
    /// Print accumulated histogram
    Print,

    // v0.2 additions (appended so v0.1 discriminants are unchanged)
    /// RZ(θ) on |q>; θ from the constant pool at index |k|.
    RZ(u8, u16),
    /// RX(θ) on |q>.
    RX(u8, u16),
    /// RY(θ) on |q>.
    RY(u8, u16),
    /// Phase(θ) = diag(1, e^{i θ}) on |q>.
    Phase(u8, u16),
    /// S = diag(1, i) on |q>.
    S(u8),
    /// T = diag(1, e^{i π/4}) on |q>.
    T(u8),
    /// SX = sqrt(X) on |q>.
    SX(u8),
    /// SWAP |a> |b>.
    SWAP(u8, u8),
    /// iSWAP |a> |b> (swap with i phase on the exchanged term).
    ISWAP(u8, u8),
    /// Controlled-Z on |a> |b>.
    CZ(u8, u8),
    /// Controlled-phase e^{i θ} on |11>; θ from constant pool.
    CPHASE(u8, u8, u16),
    /// Fredkin: |a> controls the swap of |b> and |c>.
    CSWAP(u8, u8, u8),
    /// Multi-controlled-X: flip |t> when every bit of the control mask is set.
    MCX(u32, u8),
    /// Collapse |q> to |0> (measure then correct), not recorded to classical.
    Reset(u8),
    /// Measure |q> in the X basis into |c>.
    MeasureX(u8, u8),
    /// Measure |q> in the Y basis into |c>.
    MeasureY(u8, u8),
    /// Set classical bit |c| to |v|.
    Set(u8, u8),
    /// Flip classical bit |c|.
    Not(u8),
    /// c0 &= c1
    And(u8, u8),
    /// c0 |= c1
    Or(u8, u8),
    /// c0 ^= c1
    Xor(u8, u8),
    /// c0 += c1
    Add(u8, u8),
    /// c0 -= c1
    Sub(u8, u8),
    /// Direct <ψ|P|ψ> for a Pauli product P, encoded as 2 bits per qubit:
    /// I=00, X=01, Y=10, Z=11 (qubit q at bits 2q..2q+1). No sampling.
    Expect(u64),
    /// Save the current statevector (re, im interleaved) into the results.
    SaveState,
    /// Save the current amplitudes ((index, re, im)) into the results.
    SaveAmps,
    /// Save the current basis probabilities (|amp|^2) into the results.
    SaveProbs,
    /// Jump-based subroutine call. `sub_id` indexes `Program.subs`; the
    /// subprogram body occupies `ops[off..off+len]` in the sub region
    /// (`off >= Program.len`) and treats qubits 0..nparams-1 as formal
    /// parameters. `a0..a2` are the argument qubits (the parser validates
    /// arity); unused slots are `0xFF`.
    Call(u16, u8, u8, u8),
    /// Estimator: evaluate <H> = sum over the weighted-Pauli terms
    /// `estimate_terms[off..off+len]` of coef * <P>, pushing one value.
    Estimate(u16, u16),
}

/// A jump-based subprogram: its body is a contiguous range of the flat op
/// array in the sub region (offset >= `Program.len`), with `nparams` formal
/// qubit parameters (qubits 0..nparams-1 in the body, resolved to the call's
/// arguments at execution time).
#[derive(Clone, Copy, Debug)]
pub struct Sub {
    pub off: u16,
    pub len: u16,
    pub nparams: u8,
}

/// Sentinel for unused Call argument slots.
pub const NO_ARG: u8 = 0xFF;

#[derive(Clone, Debug)]
pub struct Program {
    pub version: u8,
    pub ops: [IrOp; MAX_OPS],
    pub len: usize,
    pub num_qubits: u8,
    pub num_classical: u8,
    pub has_explicit_shot: bool,
    /// v0.2 heap-free constant pool for parametric gate angles.
    pub consts: [f64; MAX_CONSTS],
    pub num_consts: usize,
    /// Jump-based subprograms (bodies live in `ops[len..sub_len)`).
    pub subs: [Sub; MAX_SUBS],
    pub num_subs: usize,
    /// Weighted-Pauli term pool for `ESTIMATE` (coef, pauli-code).
    pub estimate_terms: [(f64, u64); MAX_ESTIMATE_TERMS],
    pub num_estimate_terms: usize,
    /// Total ops including sub-program bodies (`sub_len >= len`; the main
    /// executable region is `ops[0..len)`, sub bodies are `ops[len..sub_len)`).
    pub sub_len: usize,
}

impl Program {
    pub fn new() -> Self {
        Self {
            version: IR_VERSION,
            ops: [IrOp::H(0); MAX_OPS],
            len: 0,
            num_qubits: 0,
            num_classical: 0,
            has_explicit_shot: false,
            consts: [0.0; MAX_CONSTS],
            num_consts: 0,
            subs: [Sub { off: 0, len: 0, nparams: 0 }; MAX_SUBS],
            num_subs: 0,
            estimate_terms: [(0.0, 0); MAX_ESTIMATE_TERMS],
            num_estimate_terms: 0,
            sub_len: 0,
        }
    }

    pub fn reset(&mut self) {
        self.len = 0;
        self.num_qubits = 0;
        self.num_classical = 0;
        self.has_explicit_shot = false;
        self.num_consts = 0;
        self.num_subs = 0;
        self.num_estimate_terms = 0;
        self.sub_len = 0;
    }

    pub fn emit(&mut self, op: IrOp) -> bool {
        if self.len >= MAX_OPS {
            return false;
        }
        self.ops[self.len] = op;
        self.len += 1;
        true
    }

    /// Append a weighted-Pauli term to the ESTIMATE pool.
    pub fn push_estimate_term(&mut self, coef: f64, pauli: u64) -> bool {
        if self.num_estimate_terms >= MAX_ESTIMATE_TERMS {
            return false;
        }
        self.estimate_terms[self.num_estimate_terms] = (coef, pauli);
        self.num_estimate_terms += 1;
        true
    }

    /// Emit an op into the sub-program region (`ops[len..sub_len)`), which the
    /// main executor never runs directly, sub bodies are reached by CALL.
    pub fn emit_sub(&mut self, op: IrOp) -> bool {
        if self.sub_len >= MAX_OPS {
            return false;
        }
        self.ops[self.sub_len] = op;
        self.sub_len += 1;
        true
    }

    /// Record a subprogram whose body occupies
/// `ops[sub_len-len .. sub_len)` (just emitted into the sub region).
    /// Returns the sub_id.
    pub fn add_sub(&mut self, len: usize, nparams: u8) -> Option<u16> {
        if self.num_subs >= MAX_SUBS {
            return None;
        }
        let id = self.num_subs as u16;
        self.subs[id as usize] = Sub {
            off: (self.sub_len - len) as u16,
            len: len as u16,
            nparams,
        };
        self.num_subs += 1;
        Some(id)
    }

    /// Append a constant to the pool, returning its index for use by a
    /// parametric op. Returns None if the pool is full.
    pub fn emit_const(&mut self, v: f64) -> Option<u16> {
        if self.num_consts >= MAX_CONSTS {
            return None;
        }
        self.consts[self.num_consts] = v;
        let idx = self.num_consts as u16;
        self.num_consts += 1;
        Some(idx)
    }
}
