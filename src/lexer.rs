// Copyright (c) 2026 Faris Alfarhan
// SPDX-License-Identifier: GPL-3.0-only

pub const MAX_TOKENS: usize = 16384;

use super::diag::Diag;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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
    // Values and symbols
    Number(u32),
    ClassicalRef(u32),
    Arrow,
    Equals,
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
        }
    }

    pub fn reset(&mut self) {
        self.count = 0;
        self.error = false;
        self.diag.clear();
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

        // Digits
        if byte.is_ascii_digit() {
            let start_col = col;
            let mut val: u32 = 0;
            while cursor < input.len() && input[cursor].is_ascii_digit() {
                val = val.wrapping_mul(10).wrapping_add((input[cursor] - b'0') as u32);
                cursor += 1;
                col += 1;
            }
            append(result, Token::Number(val), line, start_col);
            continue;
        }

        // Classical reference (c0, c1, ...) — must precede identifier check
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

        // Identifiers (keywords)
        if byte.is_ascii_alphabetic() {
            let start_col = col;
            let start = cursor;
            while cursor < input.len() && input[cursor].is_ascii_alphanumeric() {
                cursor += 1;
                col += 1;
            }
            let word = &input[start..cursor];
            let token = match_keyword(word);
            if token == Token::Error {
                result.error = true;
                result.diag.set(line, start_col, "unknown keyword");
                return;
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
    fn test_unknown_keyword_is_error() {
        let mut lex = LexResult::new();
        tokenise(b"FOO 0\n", &mut lex);
        assert!(lex.error);
    }
}
