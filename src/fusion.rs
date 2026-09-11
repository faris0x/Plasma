// Copyright (c) 2026 Faris Alfarhan
// SPDX-License-Identifier: GPL-3.0-only

//! Backend-agnostic gate-fusion analysis (v0.2).
//!
//! Walks the flat IR and collapses straight-line runs of single-qubit gates
//! on the same qubit into one 2×2 unitary (`U1`) and runs of gates on the
//! same two-qubit set into one 4×4 unitary (`U2`). Control flow (`IF`, `SHOT`)
//! is preserved: bodies are fused recursively and their `(offset, len)`
//! ranges are re-patched against the *fused* op list. Ops that cannot be
//! fused (measurements, multi-qubit gates, classical computation) pass
//! through unchanged as `FusedOp::Original`.
//!
//! The fused stream feeds the `--fuse` CPU executor (`sim.rs`) and is a
//! strictly-shrinking transform: it never increases the op count. The GPU
//! backend does not consume `FusedOp`; it packs `IrOp`s into its own
//! shared-memory mega-kernel encoding (`gpu/mod.rs`, `build_mega_ops`), so
//! GPU kernel-launch elimination is independent of this pass.
//!
//! Floating point: fusing changes the evaluation order, so fused results
//! agree with the unfused reference within float tolerance, not bit-for-bit.

use alloc::vec::Vec;

use super::ir::{IrOp, Program};
use super::math::{sin_cos, sqrt};

/// A single-qubit unitary on qubit `q`, complex 2×2 (row-major `re`/`im`).
#[derive(Clone, Copy, Debug)]
pub struct C2 {
    pub re: [[f64; 2]; 2],
    pub im: [[f64; 2]; 2],
}

/// A two-qubit unitary, complex 4×4 in the basis `s = bit(q0) + 2·bit(q1)`.
#[derive(Clone, Copy, Debug)]
pub struct C4 {
    pub re: [[f64; 4]; 4],
    pub im: [[f64; 4]; 4],
}

/// One entry of the fused op stream.
#[derive(Clone, Debug)]
pub enum FusedOp {
    /// A single-qubit unitary on `q`.
    U1(u8, C2),
    /// A two-qubit unitary on `(q0, q1)`.
    U2(u8, u8, C4),
    /// An op that could not be fused; re-executed as-is.
    Original(IrOp),
}

impl C2 {
    pub fn identity() -> Self {
        let mut m = C2 {
            re: [[0.0; 2]; 2],
            im: [[0.0; 2]; 2],
        };
        m.re[0][0] = 1.0;
        m.re[1][1] = 1.0;
        m
    }
}

impl C4 {
    pub fn identity() -> Self {
        let mut m = C4 {
            re: [[0.0; 4]; 4],
            im: [[0.0; 4]; 4],
        };
        for s in 0..4 {
            m.re[s][s] = 1.0;
        }
        m
    }
}

/// 2×2 complex matrix multiply: `a·b`.
fn mul2(a: &C2, b: &C2) -> C2 {
    let mut out = C2 {
        re: [[0.0; 2]; 2],
        im: [[0.0; 2]; 2],
    };
    for i in 0..2 {
        for j in 0..2 {
            let mut sre = 0.0;
            let mut sim = 0.0;
            for k in 0..2 {
                sre += a.re[i][k] * b.re[k][j] - a.im[i][k] * b.im[k][j];
                sim += a.re[i][k] * b.im[k][j] + a.im[i][k] * b.re[k][j];
            }
            out.re[i][j] = sre;
            out.im[i][j] = sim;
        }
    }
    out
}

/// 4×4 complex matrix multiply: `a·b`.
fn mul4(a: &C4, b: &C4) -> C4 {
    let mut out = C4 {
        re: [[0.0; 4]; 4],
        im: [[0.0; 4]; 4],
    };
    for i in 0..4 {
        for j in 0..4 {
            let mut sre = 0.0;
            let mut sim = 0.0;
            for k in 0..4 {
                sre += a.re[i][k] * b.re[k][j] - a.im[i][k] * b.im[k][j];
                sim += a.re[i][k] * b.im[k][j] + a.im[i][k] * b.re[k][j];
            }
            out.re[i][j] = sre;
            out.im[i][j] = sim;
        }
    }
    out
}

/// Expand a 2×2 unitary on `qubit` (∈ {q0, q1}) to a 4×4 in the `(q0, q1)`
/// basis by tensoring with identity on the other qubit.
fn expand2_to4(m: &C2, qubit: u8, q0: u8) -> C4 {
    let mut out = C4::identity();
    if qubit == q0 {
        // m acts on the low bit (q0): M[s][t] = m[b0][b0'] if b1 == b1'.
        for b0 in 0..2 {
            for b0p in 0..2 {
                for b1 in 0..2 {
                    let s = b0 + 2 * b1;
                    let t = b0p + 2 * b1;
                    out.re[s][t] = m.re[b0][b0p];
                    out.im[s][t] = m.im[b0][b0p];
                }
            }
        }
    } else {
        // m acts on the high bit (q1): M[s][t] = m[b1][b1'] if b0 == b0'.
        for b1 in 0..2 {
            for b1p in 0..2 {
                for b0 in 0..2 {
                    let s = b0 + 2 * b1;
                    let t = b0 + 2 * b1p;
                    out.re[s][t] = m.re[b1][b1p];
                    out.im[s][t] = m.im[b1][b1p];
                }
            }
        }
    }
    out
}

// Single-qubit gate matrices

fn h2() -> C2 {
    C2 {
        re: [
            [1.0 / sqrt(2.0), 1.0 / sqrt(2.0)],
            [1.0 / sqrt(2.0), -1.0 / sqrt(2.0)],
        ],
        im: [[0.0; 2]; 2],
    }
}

fn x2() -> C2 {
    C2 {
        re: [[0.0, 1.0], [1.0, 0.0]],
        im: [[0.0; 2]; 2],
    }
}

/// RZ(θ) = diag(e^{-iθ/2}, e^{iθ/2}).
fn rz2(theta: f64) -> C2 {
    let (s, c) = sin_cos(theta * 0.5);
    // e^{-iθ/2} = c - i s;  e^{iθ/2} = c + i s
    C2 {
        re: [[c, 0.0], [0.0, c]],
        im: [[-s, 0.0], [0.0, s]],
    }
}

/// RX(θ) = [[cos, -i sin],[-i sin, cos]] (θ/2 entries).
fn rx2(theta: f64) -> C2 {
    let (s, c) = sin_cos(theta * 0.5);
    C2 {
        re: [[c, 0.0], [0.0, c]],
        im: [[0.0, -s], [-s, 0.0]],
    }
}

/// RY(θ) = [[cos, -sin],[sin, cos]] (θ/2 entries).
fn ry2(theta: f64) -> C2 {
    let (s, c) = sin_cos(theta * 0.5);
    C2 {
        re: [[c, -s], [s, c]],
        im: [[0.0; 2]; 2],
    }
}

/// Phase(θ) = diag(1, e^{iθ}).
fn phase2(theta: f64) -> C2 {
    let (s, c) = sin_cos(theta);
    C2 {
        re: [[1.0, 0.0], [0.0, c]],
        im: [[0.0, 0.0], [0.0, s]],
    }
}

/// S = diag(1, i).
fn s2() -> C2 {
    C2 {
        re: [[1.0, 0.0], [0.0, 0.0]],
        im: [[0.0, 0.0], [0.0, 1.0]],
    }
}

/// T = diag(1, e^{iπ/4}).
fn t2() -> C2 {
    let inv = 1.0 / sqrt(2.0);
    C2 {
        re: [[1.0, 0.0], [0.0, inv]],
        im: [[0.0, 0.0], [0.0, inv]],
    }
}

/// SX = (1/2)[(1+i)I + (1-i)X].
fn sx2() -> C2 {
    C2 {
        re: [[0.5, 0.5], [0.5, 0.5]],
        im: [[0.5, -0.5], [-0.5, 0.5]],
    }
}

// Two-qubit gate matrices (built in the block's (q0,q1) basis)

/// Bit of `q` in the pair `(q0, q1)`, given b0 = bit(q0), b1 = bit(q1).
/// Any qubit other than `q0` is treated as the pair's high bit.
fn bit_of(q: u8, q0: u8, b0: bool, b1: bool) -> bool {
    if q == q0 {
        b0
    } else {
        b1
    }
}

/// CNOT: flip `target` when `control` is set. Matrix in the (q0,q1) basis.
fn cnot4(control: u8, target: u8, q0: u8) -> C4 {
    let mut m = C4 {
        re: [[0.0; 4]; 4],
        im: [[0.0; 4]; 4],
    };
    for s in 0..4 {
        let b0 = (s & 1) != 0;
        let b1 = (s & 2) != 0;
        let cbit = bit_of(control, q0, b0, b1);
        let mut tbit = bit_of(target, q0, b0, b1);
        if cbit {
            tbit = !tbit;
        }
        let (nb0, nb1) = if target == q0 { (tbit, b1) } else { (b0, tbit) };
        let t = (if nb0 { 1 } else { 0 }) + 2 * (if nb1 { 1 } else { 0 });
        m.re[s][t] = 1.0;
    }
    m
}

/// SWAP |a> |b> in the (q0,q1) basis.
fn swap4(a: u8, b: u8, q0: u8) -> C4 {
    let mut m = C4 {
        re: [[0.0; 4]; 4],
        im: [[0.0; 4]; 4],
    };
    for s in 0..4 {
        let b0 = (s & 1) != 0;
        let b1 = (s & 2) != 0;
        let va = bit_of(a, q0, b0, b1);
        let vb = bit_of(b, q0, b0, b1);
        // After swap: a holds vb, b holds va.
        let (nb0, nb1) = if a == q0 {
            (vb, va) // a is low bit, b is high bit
        } else {
            (va, vb) // a is high bit, b is low bit
        };
        let t = (if nb0 { 1 } else { 0 }) + 2 * (if nb1 { 1 } else { 0 });
        m.re[s][t] = 1.0;
    }
    m
}

/// iSWAP: swap with an i phase, in the (q0,q1) basis.
fn iswap4(a: u8, b: u8, q0: u8) -> C4 {
    let mut m = swap4(a, b, q0);
    // The exchanged entries |01> ↔ |10> pick up an i phase.
    for s in 0..4 {
        let b0 = (s & 1) != 0;
        let b1 = (s & 2) != 0;
        let va = bit_of(a, q0, b0, b1);
        let vb = bit_of(b, q0, b0, b1);
        if va != vb {
            // swapped term: M[s][s'] carries i, real part zero.
            m.re[s][s] = 0.0;
            m.im[s][s] = 0.0;
            let (nb0, nb1) = if a == q0 { (vb, va) } else { (va, vb) };
            let t = (if nb0 { 1 } else { 0 }) + 2 * (if nb1 { 1 } else { 0 });
            m.re[s][t] = 0.0;
            m.im[s][t] = 1.0;
        }
    }
    m
}

/// CZ: phase -1 on |11>, in the (q0,q1) basis.
fn cz4(a: u8, b: u8, q0: u8) -> C4 {
    let mut m = C4::identity();
    for s in 0..4 {
        let b0 = (s & 1) != 0;
        let b1 = (s & 2) != 0;
        if bit_of(a, q0, b0, b1) && bit_of(b, q0, b0, b1) {
            m.re[s][s] = -1.0;
        }
    }
    m
}

/// CPHASE(θ): phase e^{iθ} on |11>, in the (q0,q1) basis.
fn cphase4(a: u8, b: u8, theta: f64, q0: u8) -> C4 {
    let (s, c) = sin_cos(theta);
    let mut m = C4::identity();
    for i in 0..4 {
        let b0 = (i & 1) != 0;
        let b1 = (i & 2) != 0;
        if bit_of(a, q0, b0, b1) && bit_of(b, q0, b0, b1) {
            m.re[i][i] = c;
            m.im[i][i] = s;
        }
    }
    m
}

// Fusion block state

/// Streaming fusion state: at most two qubits and a running matrix.
struct Block {
    q0: Option<u8>,
    q1: Option<u8>,
    m2: C2,
    m4: C4,
}

impl Block {
    fn new() -> Self {
        Block {
            q0: None,
            q1: None,
            m2: C2::identity(),
            m4: C4::identity(),
        }
    }

    /// Emit the current block (if non-empty) into `out`.
    fn flush(&mut self, out: &mut Vec<FusedOp>) {
        if let Some(q0) = self.q0 {
            if let Some(q1) = self.q1 {
                out.push(FusedOp::U2(q0, q1, self.m4));
            } else {
                out.push(FusedOp::U1(q0, self.m2));
            }
        }
        *self = Block::new();
    }

    /// Merge a single-qubit gate `g2` on qubit `q` into the block.
    fn merge1(&mut self, q: u8, g2: &C2, out: &mut Vec<FusedOp>) {
        match (self.q0, self.q1) {
            (None, None) => {
                self.q0 = Some(q);
                self.m2 = *g2;
            }
            (Some(a), None) if a == q => {
                self.m2 = mul2(g2, &self.m2);
            }
            (Some(a), None) => {
                // Promote to a pair (a, q): expand m2 to 4×4, then apply g2.
                self.m4 = expand2_to4(&self.m2, a, a);
                let g4 = expand2_to4(g2, q, a);
                self.m4 = mul4(&g4, &self.m4);
                self.q1 = Some(q);
            }
            (Some(a), Some(b)) if a == q || b == q => {
                let g4 = expand2_to4(g2, q, a);
                self.m4 = mul4(&g4, &self.m4);
            }
            _ => {
                self.flush(out);
                self.q0 = Some(q);
                self.m2 = *g2;
            }
        }
    }

    /// Merge a two-qubit gate `g4` on (a, b) into the block.
    fn merge2(&mut self, a: u8, b: u8, g4: &C4, out: &mut Vec<FusedOp>) {
        match (self.q0, self.q1) {
            (None, None) => {
                self.q0 = Some(a);
                self.q1 = Some(b);
                self.m4 = *g4;
            }
            (Some(x), None) if x == a || x == b => {
                // Promote the single-qubit block to the pair (a, b).
                self.m4 = expand2_to4(&self.m2, x, a);
                self.m4 = mul4(g4, &self.m4);
                self.q0 = Some(a);
                self.q1 = Some(b);
            }
            (Some(x), Some(y)) if (x == a && y == b) || (x == b && y == a) => {
                self.m4 = mul4(g4, &self.m4);
            }
            _ => {
                self.flush(out);
                self.q0 = Some(a);
                self.q1 = Some(b);
                self.m4 = *g4;
            }
        }
    }
}

/// Build a two-qubit gate matrix in the block's (q0, q1) ordering.
fn gate_matrix4(op: IrOp, consts: &[f64], q0: u8) -> C4 {
    match op {
        IrOp::CNOT(c, t) => cnot4(c, t, q0),
        IrOp::SWAP(a, b) => swap4(a, b, q0),
        IrOp::ISWAP(a, b) => iswap4(a, b, q0),
        IrOp::CZ(a, b) => cz4(a, b, q0),
        IrOp::CPHASE(a, b, k) => cphase4(a, b, consts[k as usize], q0),
        _ => unreachable!("not a two-qubit fusible gate"),
    }
}

/// Fuse a straight-line range of ops into `out`, recursing through control
/// flow and re-patching body ranges against the fused list.
fn fuse_range(prog: &Program, offset: usize, end: usize, out: &mut Vec<FusedOp>) {
    let mut i = offset;
    let mut block = Block::new();

    while i < end {
        let op = prog.ops[i];
        match op {
            IrOp::IfEq(c, v, boff, blen) => {
                block.flush(out);
                let (b0, bl) = (boff as usize, blen as usize);
                let ph = out.len();
                out.push(FusedOp::Original(IrOp::IfEq(c, v, 0, 0)));
                let bstart = out.len();
                fuse_range(prog, b0, b0 + bl, out);
                let blen_f = out.len() - bstart;
                if let FusedOp::Original(IrOp::IfEq(_, _, po, pl)) = &mut out[ph] {
                    *po = bstart as u16;
                    *pl = blen_f as u16;
                }
                i = b0 + bl;
                continue;
            }
            IrOp::IfNe(c, v, boff, blen) => {
                block.flush(out);
                let (b0, bl) = (boff as usize, blen as usize);
                let ph = out.len();
                out.push(FusedOp::Original(IrOp::IfNe(c, v, 0, 0)));
                let bstart = out.len();
                fuse_range(prog, b0, b0 + bl, out);
                let blen_f = out.len() - bstart;
                if let FusedOp::Original(IrOp::IfNe(_, _, po, pl)) = &mut out[ph] {
                    *po = bstart as u16;
                    *pl = blen_f as u16;
                }
                i = b0 + bl;
                continue;
            }
            IrOp::Shot(count, boff, blen) => {
                block.flush(out);
                let (b0, bl) = (boff as usize, blen as usize);
                let ph = out.len();
                out.push(FusedOp::Original(IrOp::Shot(count, 0, 0)));
                if bl != 0 {
                    let bstart = out.len();
                    fuse_range(prog, b0, b0 + bl, out);
                    let blen_f = out.len() - bstart;
                    if let FusedOp::Original(IrOp::Shot(_, po, pl)) = &mut out[ph] {
                        *po = bstart as u16;
                        *pl = blen_f as u16;
                    }
                    i = b0 + bl;
                    continue;
                } else {
                    i += 1;
                    continue;
                }
            }
            // Ops that cannot enter a unitary block: flush and pass through.
            op @ (IrOp::Measure(..)
                | IrOp::MeasureX(..)
                | IrOp::MeasureY(..)
                | IrOp::Reset(..)
                | IrOp::Toff(..)
                | IrOp::CSWAP(..)
                | IrOp::MCX(..)
                | IrOp::Set(..)
                | IrOp::Not(..)
                | IrOp::And(..)
                | IrOp::Or(..)
                | IrOp::Xor(..)
                | IrOp::Add(..)
                | IrOp::Sub(..)
                | IrOp::Expect(..)
                | IrOp::Estimate(..)
                | IrOp::SaveState
                | IrOp::SaveAmps
                | IrOp::SaveProbs
                | IrOp::Call(..)
                | IrOp::Print) => {
                block.flush(out);
                out.push(FusedOp::Original(op));
                i += 1;
                continue;
            }
            // Single-qubit gates: fold into the running matrix.
            IrOp::H(q) => {
                block.merge1(q, &h2(), out);
                i += 1;
            }
            IrOp::X(q) => {
                block.merge1(q, &x2(), out);
                i += 1;
            }
            IrOp::RZ(q, k) => {
                block.merge1(q, &rz2(prog.consts[k as usize]), out);
                i += 1;
            }
            IrOp::RX(q, k) => {
                block.merge1(q, &rx2(prog.consts[k as usize]), out);
                i += 1;
            }
            IrOp::RY(q, k) => {
                block.merge1(q, &ry2(prog.consts[k as usize]), out);
                i += 1;
            }
            IrOp::Phase(q, k) => {
                block.merge1(q, &phase2(prog.consts[k as usize]), out);
                i += 1;
            }
            IrOp::S(q) => {
                block.merge1(q, &s2(), out);
                i += 1;
            }
            IrOp::T(q) => {
                block.merge1(q, &t2(), out);
                i += 1;
            }
            IrOp::SX(q) => {
                block.merge1(q, &sx2(), out);
                i += 1;
            }
            // Two-qubit fusible gates.
            op @ (IrOp::CNOT(..) | IrOp::SWAP(..) | IrOp::ISWAP(..) | IrOp::CZ(..)
                | IrOp::CPHASE(..)) => {
                let (a, b) = match op {
                    IrOp::CNOT(a, b) | IrOp::SWAP(a, b) | IrOp::ISWAP(a, b) | IrOp::CZ(a, b) => {
                        (a, b)
                    }
                    IrOp::CPHASE(a, b, _) => (a, b),
                    _ => unreachable!(),
                };
                // The gate matrix is built in the block's pair ordering: the low bit
                // is the block's q0 (or the gate's first qubit when the block
                // is empty/single and will become the pair (a, b)).
                let low = match (block.q0, block.q1) {
                    (Some(x), Some(y)) if (x == a && y == b) || (x == b && y == a) => x,
                    _ => a,
                };
                let g4 = gate_matrix4(op, &prog.consts, low);
                block.merge2(a, b, &g4, out);
                i += 1;
            }
        }
    }
    block.flush(out);
}

/// Fuse an entire program into a `FusedOp` stream (op count never increases).
pub fn fuse_program(prog: &Program) -> Vec<FusedOp> {
    let mut out = Vec::new();
    fuse_range(prog, 0, prog.len, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuse_identity_hh() {
        let mut prog = Program::new();
        prog.num_qubits = 1;
        prog.emit(IrOp::H(0));
        prog.emit(IrOp::H(0));
        let fused = fuse_program(&prog);
        // H·H = I should collapse to a single U1.
        assert_eq!(fused.len(), 1);
        match &fused[0] {
            FusedOp::U1(_, m) => {
                assert!((m.re[0][0] - 1.0).abs() < 1e-12);
                assert!((m.re[1][1] - 1.0).abs() < 1e-12);
            }
            _ => panic!("expected U1"),
        }
    }

    #[test]
    fn test_fuse_h_cnot_h() {
        let mut prog = Program::new();
        prog.num_qubits = 2;
        prog.emit(IrOp::H(0));
        prog.emit(IrOp::CNOT(0, 1));
        prog.emit(IrOp::H(1));
        let fused = fuse_program(&prog);
        assert_eq!(fused.len(), 1);
        match &fused[0] {
            FusedOp::U2(0, 1, _) => {}
            _ => panic!("expected single U2"),
        }
    }

    #[test]
    fn test_fuse_preserves_measure() {
        let mut prog = Program::new();
        prog.num_qubits = 2;
        prog.emit(IrOp::H(0));
        prog.emit(IrOp::Measure(0, 0));
        prog.emit(IrOp::X(1));
        let fused = fuse_program(&prog);
        // H(0) fuses alone -> U1; Measure passthrough; X(1) -> U1.
        assert_eq!(fused.len(), 3);
        assert!(matches!(fused[0], FusedOp::U1(0, _)));
        assert!(matches!(fused[1], FusedOp::Original(IrOp::Measure(0, 0))));
        assert!(matches!(fused[2], FusedOp::U1(1, _)));
    }

    #[test]
    fn test_fuse_patches_if_body() {
        let mut prog = Program::new();
        prog.num_qubits = 1;
        // If c0 == 1 then { H 0; X 0 } endif
        prog.emit(IrOp::IfEq(0, 1, 1, 2));
        prog.emit(IrOp::H(0));
        prog.emit(IrOp::X(0));
        let fused = fuse_program(&prog);
        // IfEq (patched) then one fused U1 for the body.
        assert_eq!(fused.len(), 2);
        match &fused[0] {
            FusedOp::Original(IrOp::IfEq(c, v, off, len)) => {
                assert_eq!(*c, 0);
                assert_eq!(*v, 1);
                assert_eq!(*off, 1);
                assert_eq!(*len, 1);
            }
            _ => panic!("expected patched IfEq"),
        }
        assert!(matches!(fused[1], FusedOp::U1(0, _)));
    }

    #[test]
    fn test_fuse_never_grows() {
        let mut prog = Program::new();
        prog.num_qubits = 3;
        for _ in 0..50 {
            prog.emit(IrOp::H(0));
            prog.emit(IrOp::CNOT(0, 1));
            prog.emit(IrOp::RZ(1, 0));
        }
        let fused = fuse_program(&prog);
        assert!(fused.len() <= 100);
        // The whole thing is one U2 on (0,1): gates only touch {0,1}.
        assert_eq!(fused.len(), 1);
    }

    #[test]
    fn test_gate_matrix_consistency() {
        // X(0); CNOT(0,1) must equal the direct state evolution: verify
        // the 4×4 product against a manual Bell-state application.
        let mut prog = Program::new();
        prog.num_qubits = 2;
        prog.emit(IrOp::H(0));
        prog.emit(IrOp::CNOT(0, 1));
        let fused = fuse_program(&prog);
        assert_eq!(fused.len(), 1);
        match &fused[0] {
            FusedOp::U2(0, 1, m) => {
                // M = CNOT·(H⊗I). Sub-index s = bit(q0) + 2·bit(q1), so
                // row 1 = |10>, row 2 = |01>, row 3 = |11>.
                //   |00> -> (|00>+|11>)/√2
                //   |11> -> (|01>-|10>)/√2  (row2 +1/√2, row1 -1/√2)
                assert!((m.re[0][0] - 1.0 / sqrt(2.0)).abs() < 1e-12);
                assert!((m.re[3][0] - 1.0 / sqrt(2.0)).abs() < 1e-12);
                assert!((m.re[1][3] + 1.0 / sqrt(2.0)).abs() < 1e-12);
                assert!((m.re[2][3] - 1.0 / sqrt(2.0)).abs() < 1e-12);
            }
            _ => panic!("expected U2"),
        }
    }
}
