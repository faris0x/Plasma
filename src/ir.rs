// Copyright (c) 2026 Faris Alfarhan
// SPDX-License-Identifier: GPL-3.0-only

pub use core::num::NonZeroU32;

pub const MAX_OPS: usize = 16384;
pub const MAX_QUBITS: u8 = 28;
pub const MAX_CLASSICAL: u8 = 10;

/// Version of the flat IR encoding. Bump when the bytecode layout changes;
/// this is the wire contract between the frontend and any backend.
pub const IR_VERSION: u8 = 1;

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
pub enum IrOp {
    /// Single-qubit Hadamard on |q⟩
    H(u8),
    /// Pauli-X (NOT) on |q⟩
    X(u8),
    /// CNOT with |control⟩ controlling |target⟩
    CNOT(u8, u8),
    /// Toffoli with |c1⟩ and |c2⟩ controlling |target⟩
    Toff(u8, u8, u8),
    /// Measure |q⟩, store into |c⟩
    Measure(u8, u8),
    /// If classical bit |c == val|, execute block at (offset, len)
    IfEq(u8, u8, u16, u16),
    /// Repeat block at (offset, len) |count| times, resetting state each iteration.
    /// If len == 0, repeats the entire program.
    Shot(NonZeroU32, u16, u16),
    /// Print accumulated histogram
    Print,
}

#[derive(Clone, Debug)]
pub struct Program {
    pub version: u8,
    pub ops: [IrOp; MAX_OPS],
    pub len: usize,
    pub num_qubits: u8,
    pub num_classical: u8,
    pub has_explicit_shot: bool,
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
        }
    }

    pub fn reset(&mut self) {
        self.len = 0;
        self.num_qubits = 0;
        self.num_classical = 0;
        self.has_explicit_shot = false;
    }

    pub fn emit(&mut self, op: IrOp) -> bool {
        if self.len >= MAX_OPS {
            return false;
        }
        self.ops[self.len] = op;
        self.len += 1;
        true
    }
}
