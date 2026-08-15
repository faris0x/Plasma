// Copyright (c) 2026 Faris Alfarhan
// SPDX-License-Identifier: GPL-3.0-only

use core::num::NonZeroU32;

use super::diag::Diag;
use super::ir::{IrOp, Program, MAX_CLASSICAL, MAX_OPS, MAX_QUBITS};
use super::lexer::{LexResult, Token};

pub const MAX_NESTING: u8 = 32;

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

/// Reserve an op slot at the current position and return its index.
/// The caller will patch it later once the body length is known.
fn reserve(prog: &mut Program) -> Option<usize> {
    let idx = prog.len;
    if idx >= MAX_OPS {
        return None;
    }
    prog.len += 1;
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
        fail!(diag, lex, &mut pos, "qubit count must be in 1..=10");
    }
    let n = n_val as u8;
    prog.num_qubits = n;
    prog.num_classical = n;

    // Skip first newline after QUBITS line
    skip_newlines(tokens, &mut pos);

    if !parse_body(lex, tokens, &mut pos, prog, 0, diag) {
        return false;
    }

    // After body, expect EOF / END / HALT
    skip_newlines(tokens, &mut pos);
    let end = tokens.get(pos).copied().unwrap_or(Token::Eof);
    match end {
        Token::Eof | Token::End | Token::Halt => true,
        _ => fail!(diag, lex, &mut pos, "expected END or HALT at end of program"),
    }
}

fn parse_body(
    lex: &LexResult,
    tokens: &[Token],
    pos: &mut usize,
    prog: &mut Program,
    depth: u8,
    diag: &mut Diag,
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
            Token::Endif | Token::Endshot | Token::End | Token::Halt | Token::Eof => {
                return true;
            }
            _ => {}
        }
        if !parse_line(lex, tokens, pos, prog, depth, diag) {
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
) -> bool {
    let token = tokens.get(*pos).copied().unwrap_or(Token::Error);
    *pos += 1;

    match token {
        Token::H => {
            let n = qubit!(prog, tokens, pos, lex, diag);
            if !prog.emit(IrOp::H(n)) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::X => {
            let n = qubit!(prog, tokens, pos, lex, diag);
            if !prog.emit(IrOp::X(n)) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::Cnot => {
            let c = qubit!(prog, tokens, pos, lex, diag);
            let t = qubit!(prog, tokens, pos, lex, diag);
            if !prog.emit(IrOp::CNOT(c, t)) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::Toff => {
            let c1 = qubit!(prog, tokens, pos, lex, diag);
            let c2 = qubit!(prog, tokens, pos, lex, diag);
            let t = qubit!(prog, tokens, pos, lex, diag);
            if !prog.emit(IrOp::Toff(c1, c2, t)) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::Measure => {
            let q = qubit!(prog, tokens, pos, lex, diag);
            let c = if tokens.get(*pos).copied() == Some(Token::Arrow) {
                *pos += 1;
                match tokens.get(*pos).copied().unwrap_or(Token::Error) {
                    Token::ClassicalRef(c_val) => {
                        *pos += 1;
                        if c_val >= MAX_CLASSICAL as u32 {
                            fail!(diag, lex, pos, "classical bit index too large");
                        }
                        let c = c_val as u8;
                        if c >= prog.num_classical {
                            prog.num_classical = c + 1;
                        }
                        c
                    }
                    _ => fail!(diag, lex, pos, "expected a classical bit after '->'"),
                }
            } else {
                // Implicit: measure into same-numbered classical bit
                if q >= prog.num_classical {
                    prog.num_classical = q + 1;
                }
                q
            };
            if !prog.emit(IrOp::Measure(q, c)) {
                fail!(diag, lex, pos, "too many operations");
            }
            true
        }
        Token::If => {
            let Token::ClassicalRef(c_val) = tokens.get(*pos).copied().unwrap_or(Token::Error) else {
                fail!(diag, lex, pos, "expected a classical bit after IF");
            };
            *pos += 1;
            if c_val >= MAX_CLASSICAL as u32 {
                fail!(diag, lex, pos, "classical bit index too large");
            }
            let c = c_val as u8;
            if tokens.get(*pos).copied() != Some(Token::Equals) {
                fail!(diag, lex, pos, "expected '==' after IF condition");
            }
            *pos += 1;
            let Token::Number(v_val) = tokens.get(*pos).copied().unwrap_or(Token::Error) else {
                fail!(diag, lex, pos, "expected 0 or 1 after '=='");
            };
            *pos += 1;
            if v_val != 0 && v_val != 1 {
                fail!(diag, lex, pos, "IF value must be 0 or 1");
            }
            let v = v_val as u8;
            if tokens.get(*pos).copied() != Some(Token::Then) {
                fail!(diag, lex, pos, "expected THEN");
            }
            *pos += 1;
            skip_newlines(tokens, pos);

            let Some(placeholder) = reserve(prog) else {
                fail!(diag, lex, pos, "too many operations");
            };
            let body_start = prog.len;

            if !parse_body(lex, tokens, pos, prog, depth + 1, diag) {
                return false;
            }

            let body_len = prog.len - body_start;
            if body_len > u16::MAX as usize {
                fail!(diag, lex, pos, "IF body too large");
            }

            if tokens.get(*pos).copied() != Some(Token::Endif) {
                fail!(diag, lex, pos, "expected ENDIF");
            }
            *pos += 1;

            patch(
                prog,
                placeholder,
                IrOp::IfEq(c, v, body_start as u16, body_len as u16),
            );
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
                if !prog.emit(IrOp::Shot(count, 0, 0)) {
                    fail!(diag, lex, pos, "too many operations");
                }
                true
            } else {
                // Shot with explicit body
                let Some(placeholder) = reserve(prog) else {
                    fail!(diag, lex, pos, "too many operations");
                };
                let body_start = prog.len;

                if !parse_body(lex, tokens, pos, prog, depth + 1, diag) {
                    return false;
                }

                let body_len = prog.len - body_start;
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
            if !prog.emit(IrOp::Print) {
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
}
