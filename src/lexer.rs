// Copyright (c) 2026 Faris Alfarhan
// SPDX-License-Identifier: GPL-3.0-only

pub const MAX_TOKENS: usize = 16384;

use super::diag::Diag;

#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(u8)]
pub enum Token {
    // Keywords
    Qubits,
    H,
    X,
    Cnot,
    Toff,
    Measure,
    If,
    Then,
    Endif,
    Shot,
    Endshot,
    Print,
    End,
    Halt,
    // v0.2 keywords (appended; v0.1 discriminants unchanged)
    Rz,
    Rx,
    Ry,
    Phase,
    S,
    T,
    Sx,
    Swap,
    Iswap,
    Cz,
    Cphase,
    Cswap,
    Mcx,
    Reset,
    MeasureX,
    MeasureY,
    Set,
    Not,
    And,
    Or,
    Xor,
    Add,
    Sub,
    // v0.2 observables
    Expect,
    Estimate,
    SaveState,
    SaveAmps,
    SaveProbs,
    // Subroutines
    Gate,
    Endgate,
    /// A user-defined gate name (call site): any word that is not a keyword
    /// and not a Pauli product.
    Ident,
    // Values and symbols
    Number(u32),
    /// v0.2 decimal angle literal.
    Float(f64),
    /// v0.2 Pauli product like `Z0X1`, encoded as 2 bits per qubit
    /// (I=00, X=01, Y=10, Z=11; qubit q at bits 2q..2q+1).
    Pauli(u64),
    ClassicalRef(u32),
    Arrow,
    /// Unary minus for negative ESTIMATE coefficients.
    Minus,
    Equals,
    /// v0.2 `!=` (not-equals).
    Ne,
    Newline,
    Eof,
    Error,
}

#[derive(Clone, Debug)]
pub struct LexResult {
    pub tokens: [Token; MAX_TOKENS],
    pub count: usize,
    pub error: bool,
    /// Source line (1-based) of each token, for parser diagnostics.
    pub token_lines: [u16; MAX_TOKENS],
    /// Source column (1-based) of each token, for parser diagnostics.
    pub token_cols: [u16; MAX_TOKENS],
    /// Details of the lexer error, if `error` is set.
    pub diag: Diag,
    /// Identifier (gate name) storage: for each `Token::Ident` at token index
    /// `p`, the name lives in `ident_buf[ident_offs[p] .. ident_offs[p]+ident_lens[p]]`.
    /// Needed because the parser defers sub-program bodies to a second pass.
    pub ident_offs: [u32; MAX_TOKENS],
    pub ident_lens: [u16; MAX_TOKENS],
    pub ident_buf: [u8; 4096],
    pub ident_len: usize,
}

impl LexResult {
    pub fn new() -> Self {
        Self {
            tokens: [Token::Error; MAX_TOKENS],
            count: 0,
            error: false,
            token_lines: [1; MAX_TOKENS],
            token_cols: [1; MAX_TOKENS],
            diag: Diag::new(),
            ident_offs: [0; MAX_TOKENS],
            ident_lens: [0; MAX_TOKENS],
            ident_buf: [0; 4096],
            ident_len: 0,
        }
    }

    pub fn reset(&mut self) {
        self.count = 0;
        self.error = false;
        self.diag.clear();
        self.ident_len = 0;
    }

    /// The identifier (gate name) at token index `p`, if it is `Token::Ident`.
    pub fn ident_at(&self, p: usize) -> &[u8] {
        let off = self.ident_offs[p] as usize;
        let len = self.ident_lens[p] as usize;
        &self.ident_buf[off..off + len]
    }
}

/// Tokenise Plasma source. Handles comments (`//` to end of line) and skips
/// whitespace.
pub fn tokenise(input: &[u8], result: &mut LexResult) {
    result.reset();
    let mut cursor = 0usize;
    let mut line: u32 = 1;
    let mut col: u32 = 1;

    while cursor < input.len() && result.count < MAX_TOKENS {
        let byte = input[cursor];

        // Skip spaces and tabs (but not newlines)
        if byte == b' ' || byte == b'\t' || byte == b'\r' {
            cursor += 1;
            col += 1;
            continue;
        }

        // Comments: // to end of line
        if byte == b'/' && cursor + 1 < input.len() && input[cursor + 1] == b'/' {
            while cursor < input.len() && input[cursor] != b'\n' {
                cursor += 1;
                col += 1;
            }
            continue;
        }

        // Newline
        if byte == b'\n' {
            append(result, Token::Newline, line, col);
            cursor += 1;
            line += 1;
            col = 1;
            continue;
        }

        // Digits (and v0.2 decimals: digits '.' digits)
        if byte.is_ascii_digit() {
            let start_col = col;
            let mut int: u64 = 0;
            while cursor < input.len() && input[cursor].is_ascii_digit() {
                int = int.wrapping_mul(10).wrapping_add((input[cursor] - b'0') as u64);
                cursor += 1;
                col += 1;
            }
            // Decimal point: lex as Float.
            if cursor + 1 < input.len() && input[cursor] == b'.'
                && input[cursor + 1].is_ascii_digit()
            {
                cursor += 1;
                col += 1;
                let mut frac = 0.0f64;
                let mut scale = 0.1f64;
                while cursor < input.len() && input[cursor].is_ascii_digit() {
                    frac += (input[cursor] - b'0') as f64 * scale;
                    scale *= 0.1;
                    cursor += 1;
                    col += 1;
                }
                append(result, Token::Float(int as f64 + frac), line, start_col);
            } else {
                append(result, Token::Number(int as u32), line, start_col);
            }
            continue;
        }

        // Classical reference (c0, c1, ...), must precede identifier check
        // because 'c' is alphabetic and would be caught as a keyword.
        if byte == b'c' && cursor + 1 < input.len() && input[cursor + 1].is_ascii_digit() {
            let start_col = col;
            cursor += 1;
            col += 1;
            let mut val: u32 = 0;
            while cursor < input.len() && input[cursor].is_ascii_digit() {
                val = val.wrapping_mul(10).wrapping_add((input[cursor] - b'0') as u32);
                cursor += 1;
                col += 1;
            }
            append(result, Token::ClassicalRef(val), line, start_col);
            continue;
        }

        // Identifiers (keywords); `_` allowed so multi-word keywords like
        // MEASURE_X lex as a single token.
        if byte.is_ascii_alphabetic() {
            let start_col = col;
            let start = cursor;
            while cursor < input.len()
                && (input[cursor].is_ascii_alphanumeric() || input[cursor] == b'_')
            {
                cursor += 1;
                col += 1;
            }
            let word = &input[start..cursor];
            // A Pauli product (e.g. `Z0X1`) lexes as a single token. This must
            // be checked before keyword matching, but a bare `X` (a gate) is
            // not a Pauli term (a term needs a qubit digit), so gates are
            // unaffected.
            if let Some(code) = parse_pauli(word) {
                append(result, Token::Pauli(code), line, start_col);
                continue;
            }
            let token = match_keyword(word);
            if token == Token::Error {
                // Not a keyword: it is either a user-defined gate name
                // (call site) or a genuinely unknown word (parser reports
                // "undefined gate"). Store the name for the deferred
                // sub-program pass.
                let ident = &input[start..cursor];
                if result.ident_len + ident.len() <= 4096 {
                    result.ident_offs[result.count] = result.ident_len as u32;
                    result.ident_lens[result.count] = ident.len() as u16;
                    result.ident_buf[result.ident_len..result.ident_len + ident.len()]
                        .copy_from_slice(ident);
                    result.ident_len += ident.len();
                }
                append(result, Token::Ident, line, start_col);
                continue;
            }
            append(result, token, line, start_col);
            continue;
        }

        // Symbols
        match byte {
            b'-' if cursor + 1 < input.len() && input[cursor + 1] == b'>' => {
                append(result, Token::Arrow, line, col);
                cursor += 2;
                col += 2;
            }
            b'-' => {
                append(result, Token::Minus, line, col);
                cursor += 1;
                col += 1;
            }
            b'=' => {
                append(result, Token::Equals, line, col);
                cursor += 1;
                col += 1;
                // Skip second '=' in == comparison
                if cursor < input.len() && input[cursor] == b'=' {
                    cursor += 1;
                    col += 1;
                }
            }
            b'!' => {
                if cursor + 1 < input.len() && input[cursor + 1] == b'=' {
                    append(result, Token::Ne, line, col);
                    cursor += 2;
                    col += 2;
                } else {
                    result.error = true;
                    result.diag.set(line, col, "unexpected character");
                    return;
                }
            }
            _ => {
                result.error = true;
                result.diag.set(line, col, "unexpected character");
                return;
            }
        }
    }

    // If the input is exhausted without a newline terminator, that's fine
    // (EOF). But if we ran out of token capacity while input remains, the
    // program is too large to lex: report it rather than silently truncate.
    if result.count >= MAX_TOKENS {
        result.error = true;
        result.diag.set(line, col, "source exceeds token limit");
    }
}

/// Parse a Pauli product word like `Z0X1` into a 2-bits-per-qubit code
/// (I=00, X=01, Y=10, Z=11). Every term must be `[IXYZ]<qubit>`. Returns
/// None if the word is not a valid Pauli product (e.g. a bare gate keyword).
fn parse_pauli(word: &[u8]) -> Option<u64> {
    let mut code = 0u64;
    let mut i = 0;
    let mut any = false;
    while i < word.len() {
        let pv = match word[i] {
            b'I' => 0u64,
            b'X' => 1u64,
            b'Y' => 2u64,
            b'Z' => 3u64,
            _ => return None,
        };
        i += 1;
        let dstart = i;
        let mut q: u64 = 0;
        while i < word.len() && word[i].is_ascii_digit() {
            q = q.wrapping_mul(10).wrapping_add((word[i] - b'0') as u64);
            i += 1;
        }
        if i == dstart || q > 31 {
            return None; // term without a qubit, or a qubit that overflows
        }
        code |= pv << (2 * q);
        any = true;
    }
    if any {
        Some(code)
    } else {
        None
    }
}

fn match_keyword(word: &[u8]) -> Token {
    match word {
        b"QUBITS" => Token::Qubits,
        b"H" => Token::H,
        b"X" => Token::X,
        b"CNOT" => Token::Cnot,
        b"TOFF" => Token::Toff,
        b"MEASURE" => Token::Measure,
        b"IF" => Token::If,
        b"THEN" => Token::Then,
        b"ENDIF" => Token::Endif,
        b"SHOT" => Token::Shot,
        b"ENDSHOT" => Token::Endshot,
        b"PRINT" => Token::Print,
        b"END" => Token::End,
        b"HALT" => Token::Halt,
        // v0.2
        b"RZ" => Token::Rz,
        b"RX" => Token::Rx,
        b"RY" => Token::Ry,
        b"PHASE" => Token::Phase,
        b"S" => Token::S,
        b"T" => Token::T,
        b"SX" => Token::Sx,
        b"SWAP" => Token::Swap,
        b"ISWAP" => Token::Iswap,
        b"CZ" => Token::Cz,
        b"CPHASE" => Token::Cphase,
        b"CSWAP" => Token::Cswap,
        b"MCX" => Token::Mcx,
        b"RESET" => Token::Reset,
        b"MEASURE_X" => Token::MeasureX,
        b"MEASURE_Y" => Token::MeasureY,
        b"SET" => Token::Set,
        b"NOT" => Token::Not,
        b"AND" => Token::And,
        b"OR" => Token::Or,
        b"XOR" => Token::Xor,
        b"ADD" => Token::Add,
        b"SUB" => Token::Sub,
        // v0.2 observables
        b"EXPECT" => Token::Expect,
        b"ESTIMATE" => Token::Estimate,
        b"SAVE_STATEVECTOR" => Token::SaveState,
        b"SAVE_AMPLITUDES" => Token::SaveAmps,
        b"SAVE_PROBABILITIES" => Token::SaveProbs,
        // subroutines
        b"GATE" => Token::Gate,
        b"ENDGATE" => Token::Endgate,
        _ => Token::Error,
    }
}

fn append(result: &mut LexResult, token: Token, line: u32, col: u32) {
    if result.count < MAX_TOKENS {
        result.tokens[result.count] = token;
        result.token_lines[result.count] = line as u16;
        result.token_cols[result.count] = col as u16;
        result.count += 1;
    } else {
        result.error = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_gate() {
        let mut lex = LexResult::new();
        tokenise(b"H 0\nCNOT 0 1\n", &mut lex);
        assert!(!lex.error);
        assert_eq!(lex.tokens[0], Token::H);
        assert_eq!(lex.tokens[1], Token::Number(0));
        assert_eq!(lex.tokens[2], Token::Newline);
        assert_eq!(lex.tokens[3], Token::Cnot);
        assert_eq!(lex.tokens[4], Token::Number(0));
        assert_eq!(lex.tokens[5], Token::Number(1));
    }

    #[test]
    fn test_comment_skipped() {
        let mut lex = LexResult::new();
        tokenise(b"H 0 // comment\nX 1\n", &mut lex);
        assert!(!lex.error);
        assert_eq!(lex.tokens[0], Token::H);
        assert_eq!(lex.tokens[1], Token::Number(0));
        assert_eq!(lex.tokens[2], Token::Newline);
        assert_eq!(lex.tokens[3], Token::X);
        assert_eq!(lex.tokens[4], Token::Number(1));
    }

    #[test]
    fn test_classical_ref() {
        let mut lex = LexResult::new();
        tokenise(b"MEASURE 0 -> c0\n", &mut lex);
        assert!(!lex.error);
        assert_eq!(lex.tokens[0], Token::Measure);
        assert_eq!(lex.tokens[1], Token::Number(0));
        assert_eq!(lex.tokens[2], Token::Arrow);
        assert_eq!(lex.tokens[3], Token::ClassicalRef(0));
    }

    #[test]
    fn test_if_statement() {
        let mut lex = LexResult::new();
        tokenise(b"IF c0 == 1 THEN\nH 0\nENDIF\n", &mut lex);
        assert!(!lex.error);
        assert_eq!(lex.tokens[0], Token::If);
        assert_eq!(lex.tokens[1], Token::ClassicalRef(0));
        assert_eq!(lex.tokens[2], Token::Equals);
        assert_eq!(lex.tokens[3], Token::Number(1));
        assert_eq!(lex.tokens[4], Token::Then);
        assert_eq!(lex.tokens[5], Token::Newline);
        assert_eq!(lex.tokens[6], Token::H);
        assert_eq!(lex.tokens[7], Token::Number(0));
        assert_eq!(lex.tokens[8], Token::Newline);
        assert_eq!(lex.tokens[9], Token::Endif);
    }

    #[test]
    fn test_unknown_word_is_gate_name() {
        // An unknown word is a user-defined gate name (call site), not a
        // lexer error; the parser reports "undefined gate" if never defined.
        let mut lex = LexResult::new();
        tokenise(b"FOO 0\n", &mut lex);
        assert!(!lex.error);
        assert_eq!(lex.tokens[0], Token::Ident);
        assert_eq!(lex.ident_at(0), b"FOO");
    }
}
