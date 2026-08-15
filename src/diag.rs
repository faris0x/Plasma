// Copyright (c) 2026 Faris Alfarhan
// SPDX-License-Identifier: GPL-3.0-only

//! Heap-free error diagnostics shared by the lexer and parser.

pub const MAX_MSG: usize = 80;

#[derive(Clone, Copy, Debug)]
pub struct Diag {
    pub line: u32,
    pub col: u32,
    pub msg: [u8; MAX_MSG],
    pub len: usize,
}

impl Diag {
    pub const fn new() -> Self {
        Diag { line: 1, col: 1, msg: [0; MAX_MSG], len: 0 }
    }

    pub fn clear(&mut self) {
        self.line = 1;
        self.col = 1;
        self.len = 0;
    }

    pub fn set(&mut self, line: u32, col: u32, msg: &str) {
        self.line = line;
        self.col = col;
        let n = msg.len().min(MAX_MSG);
        self.msg[..n].copy_from_slice(&msg.as_bytes()[..n]);
        self.len = n;
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn message(&self) -> &str {
        core::str::from_utf8(&self.msg[..self.len]).unwrap_or("?")
    }
}
