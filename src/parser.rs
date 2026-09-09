// Copyright (c) 2026 Faris Alfarhan
// SPDX-License-Identifier: GPL-3.0-only

use core::num::NonZeroU32;

use super::diag::Diag;
use super::ir::{IrOp, Program, MAX_CLASSICAL, MAX_OPS, MAX_QUBITS, MAX_SUBS, NO_ARG};
use super::lexer::{LexResult, Token};

pub const MAX_NESTING: u8 = 32;

/// Emission mode: main executable region (0) or sub-program region (1).
const MODE_MAIN: u8 = 0;
const MODE_SUB: u8 = 1;

/// Emit an op into the main or sub region depending on `mode`.
fn emit_op(prog: &mut Program, op: IrOp, mode: u8) -> bool {
    if mode == MODE_SUB {
        prog.emit_sub(op)
    } else {
        prog.emit(op)
    }
}

/// The current emission cursor (main or sub region).
fn cursor(prog: &Program, mode: u8) -> usize {
    if mode == MODE_SUB {
        prog.sub_len
    } else {
        prog.len
    }
}

/// A recorded `GATE` definition awaiting body emission after the main pass.
#[derive(Clone, Copy)]
struct GateDef {
    name: [u8; 16],
    name_len: u8,
    nparams: u8,
    body_start: u16,
    body_end: u16,
}

/// The gate-definition table (name -> sub_id; sub_id is the index).
struct Gates {
    defs: [GateDef; MAX_SUBS],
    count: usize,
}

impl Gates {
    fn new() -> Self {
        Gates {
            defs: [GateDef {
                name: [0; 16],
                name_len: 0,
                nparams: 0,
                body_start: 0,
                body_end: 0,
            }; MAX_SUBS],
            count: 0,
        }
    }

    fn lookup(&self, name: &[u8]) -> Option<u16> {
        for (i, g) in self.defs.iter().take(self.count).enumerate() {
            if g.name_len as usize == name.len()
                && g.name[..g.name_len as usize] == name[..]
            {
                return Some(i as u16);
            }
        }
        None
    }
}

/// Record a parser error at the current token position and bail.
macro_rules! fail {
    ($diag:expr, $lex:expr, $pos:expr, $msg:expr) => {{
        let p = (*$pos).min($lex.count.saturating_sub(1));
        $diag.set($lex.token_lines[p] as u32, $lex.token_cols[p] as u32, $msg);
        return false;
    }};
}

/// Parse a qubit operand, validating it against the declared qubit count.
macro_rules! qubit {
    ($prog:expr, $tokens:expr, $pos:expr, $lex:expr, $diag:expr) => {{
        match expect_number($tokens, $pos) {
            Some(n) => {
                if n >= $prog.num_qubits as u32 {
                    fail!($diag, $lex, $pos, "qubit index out of range");
                }
                n as u8
            }
            None => fail!($diag, $lex, $pos, "expected a qubit index"),
        }
    }};
}

/// Parse a parametric angle (decimal float or integer) and push it into the
/// constant pool, returning its index.
macro_rules! angle {
    ($prog:expr, $tokens:expr, $pos:expr, $lex:expr, $diag:expr) => {{
        match $tokens.get(*$pos).copied().unwrap_or(Token::Error) {
            Token::Float(v) => {
                *$pos += 1;
                match $prog.emit_const(v) {
                    Some(k) => k,
                    None => fail!($diag, $lex, $pos, "constant pool full"),
                }
            }
            Token::Number(n) => {
                *$pos += 1;
                match $prog.emit_const(n as f64) {
                    Some(k) => k,
                    None => fail!($diag, $lex, $pos, "constant pool full"),
                }
            }
            _ => fail!($diag, $lex, $pos, "expected an angle"),
        }
    }};
}

/// Parse a classical-bit reference, validating the index.
macro_rules! cref {
    ($tokens:expr, $pos:expr, $lex:expr, $diag:expr) => {{
        match $tokens.get(*$pos).copied().unwrap_or(Token::Error) {
            Token::ClassicalRef(c) => {
                if c >= MAX_CLASSICAL as u32 {
                    fail!($diag, $lex, $pos, "classical bit index too large");
                }
                *$pos += 1;
                c as u8
            }
            _ => fail!($diag, $lex, $pos, "expected a classical bit"),
        }
    }};
}

/// Measurement basis for the multi-qubit `MEASURE`/`MEASURE_X`/`MEASURE_Y`.
#[derive(Clone, Copy)]
enum MeasureKind {
    Z,
    X,
    Y,
}

/// Parse `MEASURE[|_X|_Y] q1 [q2 ...] [-> c1 [c2 ...]]` into one measure op
/// per qubit. A missing arrow means "measure into the same-numbered bit".
fn parse_measure(
    lex: &LexResult,
    tokens: &[Token],
    pos: &mut usize,
    prog: &mut Program,
    diag: &mut Diag,
    kind: MeasureKind,
    mode: u8,
) -> bool {
    let mut qs = [0u8; MAX_CLASSICAL as usize];
    let mut nq = 0usize;
    let mut cs = [0u8; MAX_CLASSICAL as usize];
    let mut nc = 0usize;
    let mut has_arrow = false;

    let first = expect_number(tokens, pos);
    let Some(first) = first else {
        fail!(diag, lex, pos, "expected a qubit index");
    };
    if first >= prog.num_qubits as u32 {
        fail!(diag, lex, pos, "qubit index out of range");
    }
    qs[nq] = first as u8;
    nq += 1;

    loop {
        match tokens.get(*pos).copied() {
            Some(Token::Number(n)) => {
                if n >= prog.num_qubits as u32 {
                    fail!(diag, lex, pos, "qubit index out of range");
                }
                if nq >= qs.len() {
                    fail!(diag, lex, pos, "too many measurements");
                }
                qs[nq] = n as u8;
                nq += 1;
                *pos += 1;
            }
            Some(Token::Arrow) => {
                has_arrow = true;
                *pos += 1;
                while let Some(Token::ClassicalRef(c)) = tokens.get(*pos).copied() {
                    if c >= MAX_CLASSICAL as u32 {
                        fail!(diag, lex, pos, "classical bit index too large");
                    }
                    if nc >= cs.len() {
                        fail!(diag, lex, pos, "too many classical targets");
                    }
                    cs[nc] = c as u8;
                    nc += 1;
                    *pos += 1;
                }
                if nc == 0 {
                    fail!(diag, lex, pos, "expected a classical bit after '->'");
                }
                break;
            }
            _ => break,
        }
    }

    let emit_one = |prog: &mut Program, q: u8, c: u8| -> bool {
        let op = match kind {
            MeasureKind::Z => IrOp::Measure(q, c),
            MeasureKind::X => IrOp::MeasureX(q, c),
            MeasureKind::Y => IrOp::MeasureY(q, c),
        };
        emit_op(prog, op, mode)
    };

    if has_arrow {
        if nc != nq {
            fail!(diag, lex, pos, "measurement qubits and classical targets must match");
        }
        for i in 0..nq {
            if cs[i] >= prog.num_classical {
                prog.num_classical = cs[i] + 1;
            }
            if !emit_one(prog, qs[i], cs[i]) {
                fail!(diag, lex, pos, "too many operations");
            }
        }
    } else {
        for i in 0..nq {
            if qs[i] >= prog.num_classical {
                prog.num_classical = qs[i] + 1;
            }
            if !emit_one(prog, qs[i], qs[i]) {
                fail!(diag, lex, pos, "too many operations");
            }
        }
    }
    true
}

/// Reserve an op slot at the current position (main or sub cursor) and
/// return its index. The caller will patch it later once the body length
/// is known.
fn reserve(prog: &mut Program, mode: u8) -> Option<usize> {
    let idx = cursor(prog, mode);
    if idx >= MAX_OPS {
        return None;
    }
    if mode == MODE_SUB {
        prog.sub_len += 1;
    } else {
        prog.len += 1;
    }
    Some(idx)
}

fn patch(prog: &mut Program, idx: usize, op: IrOp) {
    prog.ops[idx] = op;
}

pub fn parse(lex: &LexResult, prog: &mut Program, diag: &mut Diag) -> bool {
    diag.clear();
    let tokens = &lex.tokens[..lex.count];
    let mut pos = 0usize;

    prog.reset();

    // Allow leading comments / blank lines before the QUBITS declaration.
    skip_newlines(tokens, &mut pos);

    // Expect QUBITS <n>
    if tokens.get(pos).copied() != Some(Token::Qubits) {
        fail!(diag, lex, &mut pos, "expected QUBITS declaration");
    }
    pos += 1;
    let Token::Number(n_val) = tokens.get(pos).copied().unwrap_or(Token::Error) else {
        fail!(diag, lex, &mut pos, "expected qubit count after QUBITS");
    };
    pos += 1;
    // Validate the u32 before truncating to u8 (avoids wrap-around).
    if n_val == 0 || n_val > MAX_QUBITS as u32 {
        fail!(diag, lex, &mut pos, "qubit count must be in 1..=28");
    }
    let n = n_val as u8;
    prog.num_qubits = n;
    prog.num_classical = n;

    // Skip first newline after QUBITS line
    skip_newlines(tokens, &mut pos);

    let mut gates = Gates::new();
    if !parse_body(lex, tokens, &mut pos, prog, 0, diag, MODE_MAIN, &mut gates) {
        return false;
    }

    // After body, expect EOF / END / HALT
    skip_newlines(tokens, &mut pos);
    let end = tokens.get(pos).copied().unwrap_or(Token::Eof);
    match end {
        Token::Eof | Token::End | Token::Halt => {}
        _ => fail!(diag, lex, &mut pos, "expected END or HALT at end of program"),
    }

    // Step 2: emit each GATE body into the sub region (jump-based: the main
    // executor runs ops[0..len), sub bodies live in ops[len..sub_len) and are
    // reached only by CALL). Bodies are re-parsed in sub mode so nested calls
    // resolve names from the same table.
    prog.sub_len = prog.len;
    for i in 0..gates.count {
        let g = gates.defs[i];
        let before = prog.sub_len;
        let mut bpos = g.body_start as usize;
        if !parse_body(lex, tokens, &mut bpos, prog, 0, diag, MODE_SUB, &mut gates) {
            return false;
        }
        if tokens.get(bpos).copied() != Some(Token::Endgate) {
            fail!(diag, lex, &mut bpos, "expected ENDGATE");
        }
        let blen = prog.sub_len - before;
        if prog.add_sub(blen, g.nparams).is_none() {
            fail!(diag, lex, &mut bpos, "too many gates");
        }
    }
    true
}

fn parse_body(
    lex: &LexResult,
    tokens: &[Token],
    pos: &mut usize,
    prog: &mut Program,
    depth: u8,
    diag: &mut Diag,
    mode: u8,
    gates: &mut Gates,
) -> bool {
    if depth > MAX_NESTING {
        fail!(diag, lex, pos, "nesting too deep");
    }
    while let Some(token) = tokens.get(*pos).copied() {
        match token {
            Token::Newline => {
                *pos += 1;
                continue;
            }
            // These tokens end the current block
            Token::Endif | Token::Endshot | Token::End | Token::Halt | Token::Eof
            | Token::Endgate => {
                return true;
            }
            _ => {}
        }
        if !parse_line(lex, tokens, pos, prog, depth, diag, mode, gates) {
            return false;
        }
    }
    true
}

#[inline]
fn skip_newlines(tokens: &[Token], pos: &mut usize) {
    while tokens.get(*pos).copied() == Some(Token::Newline) {
        *pos += 1;
    }
}

fn parse_line(
    lex: &LexResult,
    tokens: &[Token],
    pos: &mut usize,
    prog: &mut Program,
    depth: u8,
    diag: &mut Diag,
    mode: u8,
    gates: &mut Gates,
) -> bool {
    let token = tokens.get(*pos).copied().unwrap_or(Token::Error);
    *pos += 1;

    match token {
        Token::H => {
            let n = qubit!(prog, tokens, pos, lex, diag);
            if !emit_op(prog, IrOp::H(n), mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::X => {
            let n = qubit!(prog, tokens, pos, lex, diag);
            if !emit_op(prog, IrOp::X(n), mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::Cnot => {
            let c = qubit!(prog, tokens, pos, lex, diag);
            let t = qubit!(prog, tokens, pos, lex, diag);
            if !emit_op(prog, IrOp::CNOT(c, t), mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::Toff => {
            let c1 = qubit!(prog, tokens, pos, lex, diag);
            let c2 = qubit!(prog, tokens, pos, lex, diag);
            let t = qubit!(prog, tokens, pos, lex, diag);
            if !emit_op(prog, IrOp::Toff(c1, c2, t), mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::Rz => {
            let q = qubit!(prog, tokens, pos, lex, diag);
            let k = angle!(prog, tokens, pos, lex, diag);
            if !emit_op(prog, IrOp::RZ(q, k), mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::Rx => {
            let q = qubit!(prog, tokens, pos, lex, diag);
            let k = angle!(prog, tokens, pos, lex, diag);
            if !emit_op(prog, IrOp::RX(q, k), mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::Ry => {
            let q = qubit!(prog, tokens, pos, lex, diag);
            let k = angle!(prog, tokens, pos, lex, diag);
            if !emit_op(prog, IrOp::RY(q, k), mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::Phase => {
            let q = qubit!(prog, tokens, pos, lex, diag);
            let k = angle!(prog, tokens, pos, lex, diag);
            if !emit_op(prog, IrOp::Phase(q, k), mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::S => {
            let q = qubit!(prog, tokens, pos, lex, diag);
            if !emit_op(prog, IrOp::S(q), mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::T => {
            let q = qubit!(prog, tokens, pos, lex, diag);
            if !emit_op(prog, IrOp::T(q), mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::Sx => {
            let q = qubit!(prog, tokens, pos, lex, diag);
            if !emit_op(prog, IrOp::SX(q), mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::Swap => {
            let a = qubit!(prog, tokens, pos, lex, diag);
            let b = qubit!(prog, tokens, pos, lex, diag);
            if !emit_op(prog, IrOp::SWAP(a, b), mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::Iswap => {
            let a = qubit!(prog, tokens, pos, lex, diag);
            let b = qubit!(prog, tokens, pos, lex, diag);
            if !emit_op(prog, IrOp::ISWAP(a, b), mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::Cz => {
            let a = qubit!(prog, tokens, pos, lex, diag);
            let b = qubit!(prog, tokens, pos, lex, diag);
            if !emit_op(prog, IrOp::CZ(a, b), mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::Cphase => {
            let a = qubit!(prog, tokens, pos, lex, diag);
            let b = qubit!(prog, tokens, pos, lex, diag);
            let k = angle!(prog, tokens, pos, lex, diag);
            if !emit_op(prog, IrOp::CPHASE(a, b, k), mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::Cswap => {
            let a = qubit!(prog, tokens, pos, lex, diag);
            let b = qubit!(prog, tokens, pos, lex, diag);
            let c = qubit!(prog, tokens, pos, lex, diag);
            if !emit_op(prog, IrOp::CSWAP(a, b, c), mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::Mcx => {
            let mask = match expect_number(tokens, pos) {
                Some(m) => m,
                None => fail!(diag, lex, pos, "expected a control mask"),
            };
            if mask == 0 {
                fail!(diag, lex, pos, "MCX control mask must be nonzero");
            }
            if mask >= (1u32 << prog.num_qubits) {
                fail!(diag, lex, pos, "MCX control mask out of range");
            }
            let t = qubit!(prog, tokens, pos, lex, diag);
            if ((mask >> t) & 1) == 1 {
                fail!(diag, lex, pos, "MCX target overlaps the control mask");
            }
            if !emit_op(prog, IrOp::MCX(mask, t), mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::Reset => {
            let q = qubit!(prog, tokens, pos, lex, diag);
            if !emit_op(prog, IrOp::Reset(q), mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::Set => {
            let c = cref!(tokens, pos, lex, diag);
            let v = match expect_number(tokens, pos) {
                Some(n) if n <= 255 => n as u8,
                Some(_) => fail!(diag, lex, pos, "SET value must fit in a byte"),
                None => fail!(diag, lex, pos, "expected a value after SET"),
            };
            if !emit_op(prog, IrOp::Set(c, v), mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::Not => {
            let c = cref!(tokens, pos, lex, diag);
            if !emit_op(prog, IrOp::Not(c), mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::And => {
            let a = cref!(tokens, pos, lex, diag);
            let b = cref!(tokens, pos, lex, diag);
            if !emit_op(prog, IrOp::And(a, b), mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::Or => {
            let a = cref!(tokens, pos, lex, diag);
            let b = cref!(tokens, pos, lex, diag);
            if !emit_op(prog, IrOp::Or(a, b), mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::Xor => {
            let a = cref!(tokens, pos, lex, diag);
            let b = cref!(tokens, pos, lex, diag);
            if !emit_op(prog, IrOp::Xor(a, b), mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::Add => {
            let a = cref!(tokens, pos, lex, diag);
            let b = cref!(tokens, pos, lex, diag);
            if !emit_op(prog, IrOp::Add(a, b), mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::Sub => {
            let a = cref!(tokens, pos, lex, diag);
            let b = cref!(tokens, pos, lex, diag);
            if !emit_op(prog, IrOp::Sub(a, b), mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::Expect => {
            let Token::Pauli(code) = tokens.get(*pos).copied().unwrap_or(Token::Error) else {
                fail!(diag, lex, pos, "expected a Pauli product after EXPECT");
            };
            *pos += 1;
            // Every non-identity term must reference a declared qubit.
            for q in 0..32u64 {
                if ((code >> (2 * q)) & 3) != 0 && q >= prog.num_qubits as u64 {
                    fail!(diag, lex, pos, "Pauli qubit index out of range");
                }
            }
            if !emit_op(prog, IrOp::Expect(code), mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::Estimate => {
            // ESTIMATE c0 P0 c1 P1 ...  ->  <H> = sum c_i <P_i>.
            let off = prog.num_estimate_terms;
            let mut n = 0usize;
            loop {
                let (neg, num) = match tokens.get(*pos).copied() {
                    Some(Token::Minus) => {
                        let t = tokens.get(*pos + 1).copied().unwrap_or(Token::Error);
                        *pos += 1;
                        (true, t)
                    }
                    t => (false, t.unwrap_or(Token::Error)),
                };
                let coef = match num {
                    Token::Number(x) => {
                        if neg {
                            -(x as f64)
                        } else {
                            x as f64
                        }
                    }
                    Token::Float(x) => {
                        if neg {
                            -x
                        } else {
                            x
                        }
                    }
                    _ => {
                        if neg {
                            fail!(diag, lex, pos, "expected a coefficient after '-' in ESTIMATE");
                        }
                        break;
                    }
                };
                *pos += 1;
                let Token::Pauli(code) = tokens.get(*pos).copied().unwrap_or(Token::Error) else {
                    fail!(diag, lex, pos, "expected a Pauli product after the coefficient");
                };
                *pos += 1;
                for q in 0..32u64 {
                    if ((code >> (2 * q)) & 3) != 0 && q >= prog.num_qubits as u64 {
                        fail!(diag, lex, pos, "Pauli qubit index out of range");
                    }
                }
                if !prog.push_estimate_term(coef, code as u32) {
                    fail!(diag, lex, pos, "too many ESTIMATE terms");
                }
                n += 1;
            }
            if n == 0 {
                fail!(diag, lex, pos, "expected at least one coefficient/Pauli pair after ESTIMATE");
            }
            if !emit_op(prog, IrOp::Estimate(off as u16, n as u16), mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::SaveState => {
            if !emit_op(prog, IrOp::SaveState, mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::SaveAmps => {
            if !emit_op(prog, IrOp::SaveAmps, mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::SaveProbs => {
            if !emit_op(prog, IrOp::SaveProbs, mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::Gate => {
            // Only at the top level; sub-program bodies cannot define gates.
            if mode == MODE_SUB {
                fail!(diag, lex, pos, "GATE definitions cannot be nested");
            }
            let name_pos = *pos;
            if tokens.get(*pos).copied() != Some(Token::Ident) {
                fail!(diag, lex, pos, "expected a gate name after GATE");
            }
            *pos += 1;
            let np = match expect_number(tokens, pos) {
                Some(n) if (1..=3).contains(&n) => n as u8,
                Some(_) => fail!(diag, lex, pos, "gate parameter count must be 1..=3"),
                None => fail!(diag, lex, pos, "expected a parameter count after the gate name"),
            };
            if gates.count >= MAX_SUBS {
                fail!(diag, lex, pos, "too many gates");
            }
            let name = lex.ident_at(name_pos);
            if name.len() > 16 {
                fail!(diag, lex, pos, "gate name too long");
            }
            if gates.lookup(name).is_some() {
                fail!(diag, lex, pos, "duplicate gate definition");
            }
            skip_newlines(tokens, pos);
            let body_start = *pos;
            // Skip the body statements up to ENDGATE (validated in step 2).
            loop {
                match tokens.get(*pos).copied() {
                    Some(Token::Endgate) => break,
                    Some(Token::End) | Some(Token::Halt) | Some(Token::Eof) => {
                        fail!(diag, lex, pos, "expected ENDGATE");
                    }
                    _ => *pos += 1,
                }
            }
            let body_end = *pos;
            *pos += 1; // consume ENDGATE
            let d = &mut gates.defs[gates.count];
            d.name[..name.len()].copy_from_slice(name);
            d.name_len = name.len() as u8;
            d.nparams = np;
            d.body_start = body_start as u16;
            d.body_end = body_end as u16;
            gates.count += 1;
            true
        }
        Token::Ident => {
            // A call to a user-defined gate: `name q1 q2 q3`.
            let name_pos = *pos - 1;
            let name = lex.ident_at(name_pos);
            let Some(sub_id) = gates.lookup(name) else {
                fail!(diag, lex, pos, "undefined gate");
            };
            let np = gates.defs[sub_id as usize].nparams;
            let mut args = [NO_ARG; 3];
            for k in 0..np {
                let q = qubit!(prog, tokens, pos, lex, diag);
                args[k as usize] = q;
            }
            if !emit_op(prog, IrOp::Call(sub_id, args[0], args[1], args[2]), mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::Measure => parse_measure(lex, tokens, pos, prog, diag, MeasureKind::Z, mode),
        Token::MeasureX => parse_measure(lex, tokens, pos, prog, diag, MeasureKind::X, mode),
        Token::MeasureY => parse_measure(lex, tokens, pos, prog, diag, MeasureKind::Y, mode),
        Token::If => {
            let Token::ClassicalRef(c_val) = tokens.get(*pos).copied().unwrap_or(Token::Error) else {
                fail!(diag, lex, pos, "expected a classical bit after IF");
            };
            *pos += 1;
            if c_val >= MAX_CLASSICAL as u32 {
                fail!(diag, lex, pos, "classical bit index too large");
            }
            let c = c_val as u8;
            let negate = match tokens.get(*pos).copied() {
                Some(Token::Equals) => {
                    *pos += 1;
                    false
                }
                Some(Token::Ne) => {
                    *pos += 1;
                    true
                }
                _ => fail!(diag, lex, pos, "expected '==' or '!=' after IF condition"),
            };
            let Token::Number(v_val) = tokens.get(*pos).copied().unwrap_or(Token::Error) else {
                fail!(diag, lex, pos, "expected a value after the comparison");
            };
            *pos += 1;
            if v_val > 255 {
                fail!(diag, lex, pos, "IF value must fit in a byte");
            }
            let v = v_val as u8;
            if tokens.get(*pos).copied() != Some(Token::Then) {
                fail!(diag, lex, pos, "expected THEN");
            }
            *pos += 1;
            skip_newlines(tokens, pos);

            let Some(placeholder) = reserve(prog, mode) else {
                fail!(diag, lex, pos, "too many operations");
            };
            let body_start = cursor(prog, mode);

            if !parse_body(lex, tokens, pos, prog, depth + 1, diag, mode, gates) {
                return false;
            }

            let body_len = cursor(prog, mode) - body_start;
            if body_len > u16::MAX as usize {
                fail!(diag, lex, pos, "IF body too large");
            }

            if tokens.get(*pos).copied() != Some(Token::Endif) {
                fail!(diag, lex, pos, "expected ENDIF");
            }
            *pos += 1;

            let op = if negate {
                IrOp::IfNe(c, v, body_start as u16, body_len as u16)
            } else {
                IrOp::IfEq(c, v, body_start as u16, body_len as u16)
            };
            patch(prog, placeholder, op);
            true
        }
        Token::Shot => {
            let Token::Number(n) = tokens.get(*pos).copied().unwrap_or(Token::Error) else {
                fail!(diag, lex, pos, "expected shot count");
            };
            *pos += 1;
            let count = NonZeroU32::new(n).unwrap_or(NonZeroU32::new(1).unwrap());

            skip_newlines(tokens, pos);

            // Determine whether this SHOT has a body block.
            let next = tokens.get(*pos).copied().unwrap_or(Token::Eof);
            if next == Token::Endshot || !is_body_start(next) {
                // Shot without body: runs entire program
                if next == Token::Endshot {
                    *pos += 1;
                }
                prog.has_explicit_shot = true;
                if !emit_op(prog, IrOp::Shot(count, 0, 0), mode) {
                    fail!(diag, lex, pos, "too many operations");
                }
                true
            } else {
                // Shot with explicit body
                let Some(placeholder) = reserve(prog, mode) else {
                    fail!(diag, lex, pos, "too many operations");
                };
                let body_start = cursor(prog, mode);

                if !parse_body(lex, tokens, pos, prog, depth + 1, diag, mode, gates) {
                    return false;
                }

                let body_len = cursor(prog, mode) - body_start;
                if body_len > u16::MAX as usize {
                    fail!(diag, lex, pos, "SHOT body too large");
                }

                if tokens.get(*pos).copied() != Some(Token::Endshot) {
                    fail!(diag, lex, pos, "expected ENDSHOT");
                }
                *pos += 1;

                prog.has_explicit_shot = true;
                patch(
                    prog,
                    placeholder,
                    IrOp::Shot(count, body_start as u16, body_len as u16),
                );
                true
            }
        }
        Token::Print => {
            if !emit_op(prog, IrOp::Print, mode) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        _ => fail!(diag, lex, pos, "unexpected token"),
    }
}

fn expect_number(tokens: &[Token], pos: &mut usize) -> Option<u32> {
    match tokens.get(*pos).copied()? {
        Token::Number(n) => {
            *pos += 1;
            Some(n)
        }
        _ => None,
    }
}

fn is_body_start(t: Token) -> bool {
    matches!(
        t,
        Token::H | Token::X | Token::Cnot | Token::Toff
            | Token::Measure | Token::If | Token::Shot
            | Token::Rz | Token::Rx | Token::Ry | Token::Phase
            | Token::S | Token::T | Token::Sx
            | Token::Swap | Token::Iswap | Token::Cz | Token::Cphase
            | Token::Cswap | Token::Mcx | Token::Reset
            | Token::MeasureX | Token::MeasureY
            | Token::Set | Token::Not | Token::And | Token::Or
            | Token::Xor | Token::Add | Token::Sub
            | Token::Expect | Token::Estimate | Token::SaveState | Token::SaveAmps
            | Token::SaveProbs | Token::Gate | Token::Ident
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer;
    use crate::diag::Diag;

    fn parse_str(input: &[u8]) -> Option<Program> {
        let mut lex = LexResult::new();
        lexer::tokenise(input, &mut lex);
        if lex.error {
            return None;
        }
        let mut prog = Program::new();
        let mut diag = Diag::new();
        if parse(&lex, &mut prog, &mut diag) {
            Some(prog)
        } else {
            None
        }
    }

    fn parse_diag(input: &[u8]) -> (bool, Diag) {
        let mut lex = LexResult::new();
        lexer::tokenise(input, &mut lex);
        let mut prog = Program::new();
        let mut diag = Diag::new();
        let ok = !lex.error && parse(&lex, &mut prog, &mut diag);
        (ok, diag)
    }

    #[test]
    fn test_simple_circuit() {
        let prog = parse_str(b"QUBITS 2\nH 0\nCNOT 0 1\nMEASURE 0 -> c0\n").unwrap();
        assert_eq!(prog.num_qubits, 2);
        assert_eq!(prog.num_classical, 2);
        assert_eq!(prog.ops[0], IrOp::H(0));
        assert_eq!(prog.ops[1], IrOp::CNOT(0, 1));
        assert_eq!(prog.ops[2], IrOp::Measure(0, 0));
    }

    #[test]
    fn test_measure_implicit_classical() {
        let prog = parse_str(b"QUBITS 2\nMEASURE 0\n").unwrap();
        assert_eq!(prog.ops[0], IrOp::Measure(0, 0));
    }

    #[test]
    fn test_if_statement() {
        let prog = parse_str(b"QUBITS 1\nIF c0 == 1 THEN\nX 0\nENDIF\n").unwrap();
        assert_eq!(prog.ops[0], IrOp::IfEq(0, 1, 1, 1));
        assert_eq!(prog.ops[1], IrOp::X(0));
    }

    #[test]
    fn test_shot_with_body() {
        let prog = parse_str(b"QUBITS 1\nSHOT 10\nH 0\nENDSHOT\n").unwrap();
        assert!(prog.has_explicit_shot);
        let IrOp::Shot(count, off, len) = prog.ops[0] else {
            panic!("expected Shot");
        };
        assert_eq!(count.get(), 10);
        assert_eq!(off, 1);
        assert_eq!(len, 1);
        assert_eq!(prog.ops[1], IrOp::H(0));
    }

    #[test]
    fn test_shot_without_body() {
        let prog = parse_str(b"QUBITS 1\nH 0\nSHOT 100\nPRINT\n").unwrap();
        assert!(prog.has_explicit_shot);
        assert_eq!(prog.ops[1], IrOp::Shot(NonZeroU32::new(100).unwrap(), 0, 0));
    }

    #[test]
    fn test_toffoli() {
        let prog = parse_str(b"QUBITS 3\nTOFF 0 1 2\n").unwrap();
        assert_eq!(prog.ops[0], IrOp::Toff(0, 1, 2));
    }

    #[test]
    fn test_invalid_nesting_too_deep() {
        let mut input = b"QUBITS 1\n".to_vec();
        // Generate 33 nested IFs
        for _ in 0..33 {
            input.extend_from_slice(b"IF c0 == 1 THEN\n");
        }
        for _ in 0..33 {
            input.extend_from_slice(b"ENDIF\n");
        }
        assert!(parse_str(&input).is_none());
    }

    #[test]
    fn test_print_instruction() {
        let prog = parse_str(b"QUBITS 1\nH 0\nPRINT\n").unwrap();
        assert_eq!(prog.ops[1], IrOp::Print);
    }

    #[test]
    fn test_out_of_range_qubit_rejected() {
        // H on qubit 2 of a 1-qubit circuit must be rejected.
        assert!(parse_str(b"QUBITS 1\nH 2\n").is_none());
        assert!(parse_str(b"QUBITS 2\nCNOT 0 3\n").is_none());
        assert!(parse_str(b"QUBITS 3\nTOFF 0 1 7\n").is_none());
    }

    #[test]
    fn test_truncating_large_operand_rejected() {
        // 266 as u8 wraps to 10; must not silently become qubit 10.
        assert!(parse_str(b"QUBITS 1\nH 266\n").is_none());
        assert!(parse_str(b"QUBITS 2\nMEASURE 300\n").is_none());
    }

    #[test]
    fn test_qubit_count_overflow_rejected() {
        // 266 as u8 == 10; must not wrap into an accepted qubit count.
        assert!(parse_str(b"QUBITS 266\nH 0\n").is_none());
    }

    #[test]
    fn test_diag_reports_position() {
        let (ok, diag) = parse_diag(b"QUBITS 2\nH 0\nCNOT 0 9\n");
        assert!(!ok);
        assert_eq!(diag.line, 3); // error is on line 3
        assert_eq!(diag.message(), "qubit index out of range");
    }

    #[test]
    fn test_diag_missing_qubits() {
        let (ok, diag) = parse_diag(b"H 0\n");
        assert!(!ok);
        assert_eq!(diag.message(), "expected QUBITS declaration");
    }

    // v0.2

    #[test]
    fn test_rotations_and_angles() {
        let prog = parse_str(b"QUBITS 1\nRZ 0 1.5707963267948966\nRX 0 0.5\nRY 0 3\n").unwrap();
        assert_eq!(prog.ops[0], IrOp::RZ(0, 0));
        assert_eq!(prog.ops[1], IrOp::RX(0, 1));
        assert_eq!(prog.ops[2], IrOp::RY(0, 2));
        assert_eq!(prog.consts[0], 1.5707963267948966);
        assert_eq!(prog.consts[1], 0.5);
        assert_eq!(prog.consts[2], 3.0);
        assert_eq!(prog.num_consts, 3);
    }

    #[test]
    fn test_fixed_gates() {
        let prog = parse_str(b"QUBITS 2\nS 0\nT 1\nSX 0\nSWAP 0 1\nISWAP 0 1\nCZ 0 1\n").unwrap();
        assert_eq!(prog.ops[0], IrOp::S(0));
        assert_eq!(prog.ops[1], IrOp::T(1));
        assert_eq!(prog.ops[2], IrOp::SX(0));
        assert_eq!(prog.ops[3], IrOp::SWAP(0, 1));
        assert_eq!(prog.ops[4], IrOp::ISWAP(0, 1));
        assert_eq!(prog.ops[5], IrOp::CZ(0, 1));
    }

    #[test]
    fn test_mcx_and_cswap() {
        let prog = parse_str(b"QUBITS 4\nCSWAP 0 1 2\nMCX 7 3\nCPHASE 0 1 0.25\n").unwrap();
        assert_eq!(prog.ops[0], IrOp::CSWAP(0, 1, 2));
        assert_eq!(prog.ops[1], IrOp::MCX(7, 3));
        assert_eq!(prog.ops[2], IrOp::CPHASE(0, 1, 0));
        assert_eq!(prog.consts[0], 0.25);
    }

    #[test]
    fn test_mcx_validation() {
        // Target overlapping the mask must be rejected.
        assert!(parse_str(b"QUBITS 4\nMCX 7 2\n").is_none()); // mask 0b111 includes target 2
        assert!(parse_str(b"QUBITS 2\nMCX 0 1\n").is_none()); // zero mask
        assert!(parse_str(b"QUBITS 2\nMCX 8 1\n").is_none()); // mask bit out of range
    }

    #[test]
    fn test_reset_and_measure_x_y() {
        let prog = parse_str(b"QUBITS 2\nRESET 0\nMEASURE_X 0 -> c0\nMEASURE_Y 1\n").unwrap();
        assert_eq!(prog.ops[0], IrOp::Reset(0));
        assert_eq!(prog.ops[1], IrOp::MeasureX(0, 0));
        assert_eq!(prog.ops[2], IrOp::MeasureY(1, 1));
    }

    #[test]
    fn test_classical_ops() {
        let prog = parse_str(b"QUBITS 1\nSET c0 5\nNOT c0\nAND c0 c1\nOR c0 c1\nXOR c0 c1\nADD c0 c1\nSUB c0 c1\n").unwrap();
        assert_eq!(prog.ops[0], IrOp::Set(0, 5));
        assert_eq!(prog.ops[1], IrOp::Not(0));
        assert_eq!(prog.ops[2], IrOp::And(0, 1));
        assert_eq!(prog.ops[3], IrOp::Or(0, 1));
        assert_eq!(prog.ops[4], IrOp::Xor(0, 1));
        assert_eq!(prog.ops[5], IrOp::Add(0, 1));
        assert_eq!(prog.ops[6], IrOp::Sub(0, 1));
    }

    #[test]
    fn test_multi_measure() {
        let prog = parse_str(b"QUBITS 3\nMEASURE 0 1 2 -> c0 c1 c2\n").unwrap();
        assert_eq!(prog.ops[0], IrOp::Measure(0, 0));
        assert_eq!(prog.ops[1], IrOp::Measure(1, 1));
        assert_eq!(prog.ops[2], IrOp::Measure(2, 2));
    }

    #[test]
    fn test_multi_measure_implicit() {
        let prog = parse_str(b"QUBITS 3\nMEASURE 0 2\n").unwrap();
        assert_eq!(prog.ops[0], IrOp::Measure(0, 0));
        assert_eq!(prog.ops[1], IrOp::Measure(2, 2));
    }

    #[test]
    fn test_multi_measure_mismatch_rejected() {
        assert!(parse_str(b"QUBITS 3\nMEASURE 0 1 2 -> c0 c1\n").is_none());
    }

    #[test]
    fn test_if_ne_lowered() {
        // IF c0 != 1 THEN ... lowers to IfNe(c0, 1, ...).
        let prog = parse_str(b"QUBITS 1\nIF c0 != 1 THEN\nX 0\nENDIF\n").unwrap();
        assert_eq!(prog.ops[0], IrOp::IfNe(0, 1, 1, 1));
        assert_eq!(prog.ops[1], IrOp::X(0));
    }

    #[test]
    fn test_if_byte_value() {
        // IF compares the full byte; any value 0..=255 is allowed.
        let prog = parse_str(b"QUBITS 1\nIF c0 == 5 THEN\nX 0\nENDIF\n").unwrap();
        assert_eq!(prog.ops[0], IrOp::IfEq(0, 5, 1, 1));
        assert!(parse_str(b"QUBITS 1\nIF c0 == 256 THEN\nX 0\nENDIF\n").is_none());
    }

    #[test]
    fn test_float_lexing() {
        let mut lex = LexResult::new();
        crate::lexer::tokenise(b"QUBITS 1\nRZ 0 3.14159\n", &mut lex);
        assert!(!lex.error);
        assert_eq!(lex.tokens[5], Token::Float(3.14159));
    }

    #[test]
    fn test_pauli_lexing() {
        // Z0X1 -> Z on q0 (3) | X on q1 (1 << 2) = 7.
        let mut lex = LexResult::new();
        crate::lexer::tokenise(b"QUBITS 2\nEXPECT Z0X1\n", &mut lex);
        assert!(!lex.error);
        assert_eq!(lex.tokens[3], Token::Expect);
        assert_eq!(lex.tokens[4], Token::Pauli(7));
    }

    #[test]
    fn test_expect_parse() {
        let prog = parse_str(b"QUBITS 2\nEXPECT Z0Z1\nEXPECT Y0\n").unwrap();
        assert_eq!(prog.ops[0], IrOp::Expect(15)); // Z0Z1
        assert_eq!(prog.ops[1], IrOp::Expect(2)); // Y0
    }

    #[test]
    fn test_expect_qubit_range_validated() {
        // Z2 on a 2-qubit program must be rejected.
        assert!(parse_str(b"QUBITS 2\nEXPECT Z2\n").is_none());
        // Z0Z3 on 2 qubits: Z3 out of range.
        assert!(parse_str(b"QUBITS 2\nEXPECT Z0Z3\n").is_none());
    }

    #[test]
    fn test_save_parse() {
        let prog = parse_str(
            b"QUBITS 1\nH 0\nSAVE_STATEVECTOR\nSAVE_AMPLITUDES\nSAVE_PROBABILITIES\n",
        )
        .unwrap();
        assert_eq!(prog.ops[1], IrOp::SaveState);
        assert_eq!(prog.ops[2], IrOp::SaveAmps);
        assert_eq!(prog.ops[3], IrOp::SaveProbs);
    }
}
