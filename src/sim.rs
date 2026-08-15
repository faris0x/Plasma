// Copyright (c) 2026 Faris Alfarhan
// SPDX-License-Identifier: GPL-3.0-only

use alloc::vec;
use alloc::vec::Vec;

use super::ir::{IrOp, Program};
use super::rng::Lcg64;

/// Normalisation factor 1/√2 for Hadamard gates.
const INV_SQRT_2: f64 = 0.7071067811865476;

/// Seed for initial state: all amplitude in |0...0⟩.
const GROUND_STATE: usize = 0;

#[derive(Clone, Debug)]
pub struct State {
    pub re: Vec<f64>,
    pub im: Vec<f64>,
}

impl State {
    /// Allocate a zeroed state of `n` amplitudes and set |0...0⟩.
    pub fn new(n: usize) -> Self {
        let mut s = Self {
            re: vec![0.0; n],
            im: vec![0.0; n],
        };
        s.re[GROUND_STATE] = 1.0;
        s
    }

    pub fn reset(&mut self) {
        for v in self.re.iter_mut() {
            *v = 0.0;
        }
        for v in self.im.iter_mut() {
            *v = 0.0;
        }
        self.re[GROUND_STATE] = 1.0;
    }
}

pub struct CpuBackend {
    state: State,
}

impl CpuBackend {
    pub fn new(num_qubits: u8) -> Self {
        let n = 1usize << num_qubits;
        let backend = Self {
            state: State::new(n),
        };
        backend
    }

    pub fn reset(&mut self) {
        self.state.reset();
    }

    // ─── Single-qubit gates ────────────────────────────────────────

    pub fn apply_h(&mut self, target: u8) {
        let bit = 1 << target;
        let n = self.state.re.len();
        for i in 0..n {
            if i & bit != 0 {
                continue;
            }
            let j = i ^ bit;
            if j > i {
                let re_i = self.state.re[i];
                let im_i = self.state.im[i];
                let re_j = self.state.re[j];
                let im_j = self.state.im[j];
                self.state.re[i] = INV_SQRT_2 * (re_i + re_j);
                self.state.im[i] = INV_SQRT_2 * (im_i + im_j);
                self.state.re[j] = INV_SQRT_2 * (re_i - re_j);
                self.state.im[j] = INV_SQRT_2 * (im_i - im_j);
            }
        }
    }

    pub fn apply_x(&mut self, target: u8) {
        let bit = 1 << target;
        let n = self.state.re.len();
        for i in 0..n {
            if i & bit != 0 {
                continue;
            }
            let j = i ^ bit;
            self.state.re.swap(i, j);
            self.state.im.swap(i, j);
        }
    }

    // ─── Multi-qubit gates ──────────────────────────────────────────

    pub fn apply_cnot(&mut self, control: u8, target: u8) {
        let cbit = 1 << control;
        let tbit = 1 << target;
        let n = self.state.re.len();
        for i in 0..n {
            if i & cbit == 0 {
                continue;
            }
            if i & tbit != 0 {
                continue;
            }
            let j = i ^ tbit;
            self.state.re.swap(i, j);
            self.state.im.swap(i, j);
        }
    }

    pub fn apply_toff(&mut self, c1: u8, c2: u8, target: u8) {
        let c1bit = 1 << c1;
        let c2bit = 1 << c2;
        let tbit = 1 << target;
        let n = self.state.re.len();
        for i in 0..n {
            if i & (c1bit | c2bit) != (c1bit | c2bit) {
                continue;
            }
            if i & tbit != 0 {
                continue;
            }
            let j = i ^ tbit;
            self.state.re.swap(i, j);
            self.state.im.swap(i, j);
        }
    }

    // ─── Measurement ────────────────────────────────────────────────

    pub fn probability(&self, qubit: u8) -> f64 {
        let bit = 1 << qubit;
        let mut sum = 0.0;
        for i in 0..self.state.re.len() {
            if i & bit != 0 {
                sum += self.state.re[i] * self.state.re[i]
                    + self.state.im[i] * self.state.im[i];
            }
        }
        sum
    }

    pub fn measure(&mut self, qubit: u8, rng: &mut Lcg64) -> u8 {
        let p = self.probability(qubit);
        let outcome = if rng.next_f64() < p { 1 } else { 0 };
        self.collapse(qubit, outcome);
        outcome
    }

    fn collapse(&mut self, qubit: u8, outcome: u8) {
        let n = self.state.re.len();
        let mut norm_sq = 0.0;
        for i in 0..n {
            if ((i >> qubit) & 1) as u8 != outcome {
                self.state.re[i] = 0.0;
                self.state.im[i] = 0.0;
            } else {
                norm_sq += self.state.re[i] * self.state.re[i]
                    + self.state.im[i] * self.state.im[i];
            }
        }
        if norm_sq > 0.0 {
            let norm = f64_sqrt(norm_sq);
            for i in 0..n {
                if ((i >> qubit) & 1) as u8 == outcome {
                    self.state.re[i] /= norm;
                    self.state.im[i] /= norm;
                }
            }
        }
    }

    // ─── Sampling ──────────────────────────────────────────────────

    /// Sample a single basis state according to the Born rule.
    pub fn sample(&self, rng: &mut Lcg64) -> usize {
        let r = rng.next_f64();
        let n = self.state.re.len();
        let mut cumulative = 0.0;
        for i in 0..n {
            cumulative += self.state.re[i] * self.state.re[i]
                + self.state.im[i] * self.state.im[i];
            if r < cumulative {
                return i;
            }
        }
        n - 1
    }
}

// ─── Top-level execution ───────────────────────────────────────────

/// Execute a parsed program and fill the histogram.
pub fn run_program(
    prog: &Program,
    classical: &mut [u8],
    histogram: &mut [u32],
    rng: &mut Lcg64,
) {
    let mut backend = CpuBackend::new(prog.num_qubits);

    for c in classical.iter_mut() { *c = 0; }
    for h in histogram.iter_mut() { *h = 0; }

    exec_ops(prog, 0, prog.len, &mut backend, classical, histogram, rng);
    // Without an explicit SHOT, execute once and add a single sample.
    if !prog.has_explicit_shot {
        let sample = backend.sample(rng);
        histogram[sample] += 1;
    }
}

/// Execute `prog` as a bare-shot run: reset the state each iteration and
/// sample, accumulating into `histogram`. Meaningful for programs without
/// their own SHOT ops; classical bits persist across iterations.
pub fn run_program_shots(
    prog: &Program,
    shots: u32,
    classical: &mut [u8],
    histogram: &mut [u32],
    rng: &mut Lcg64,
) {
    let mut backend = CpuBackend::new(prog.num_qubits);
    for c in classical.iter_mut() { *c = 0; }
    for h in histogram.iter_mut() { *h = 0; }
    for _ in 0..shots {
        backend.reset();
        exec_ops(prog, 0, prog.len, &mut backend, classical, histogram, rng);
        let sample = backend.sample(rng);
        histogram[sample] += 1;
    }
}

fn exec_ops(
    prog: &Program,
    offset: usize,
    end: usize,
    backend: &mut CpuBackend,
    classical: &mut [u8],
    histogram: &mut [u32],
    rng: &mut Lcg64,
) {
    let mut i = offset;
    while i < end {
        match prog.ops[i] {
            IrOp::H(q) => backend.apply_h(q),
            IrOp::X(q) => backend.apply_x(q),
            IrOp::CNOT(c, t) => backend.apply_cnot(c, t),
            IrOp::Toff(c1, c2, t) => backend.apply_toff(c1, c2, t),
            IrOp::Measure(q, c) => {
                let result = backend.measure(q, rng);
                classical[c as usize] = result;
            }
            IrOp::IfEq(c, val, body_off, body_len) => {
                let boff = body_off as usize;
                let blen = body_len as usize;
                if classical[c as usize] == val {
                    exec_ops(prog, boff, boff + blen, backend, classical, histogram, rng);
                }
                // Body ops are inline in the flat array — skip past them
                // to avoid executing them again in the sequential loop.
                i = boff + blen;
                continue;
            }
            IrOp::Shot(count, body_off, body_len) => {
                let (boff, blen) = (body_off as usize, body_len as usize);
                if blen == 0 {
                    // Shot without body: body is all ops before this Shot
                    let shot_pos = i;
                    for _ in 0..count.get() {
                        backend.reset();
                        exec_ops(prog, 0, shot_pos, backend, classical, histogram, rng);
                        let sample = backend.sample(rng);
                        histogram[sample] += 1;
                    }
                } else {
                    // Shot with explicit body block
                    for _ in 0..count.get() {
                        backend.reset();
                        exec_ops(prog, boff, boff + blen, backend, classical, histogram, rng);
                        let sample = backend.sample(rng);
                        histogram[sample] += 1;
                    }
                    i = boff + blen;
                    continue;
                }
            }
            IrOp::Print => {
                // Handled at the REPL level
            }
        }
        i += 1;
    }
}

/// Fast f64 sqrt via Newton–Raphson iteration.
fn f64_sqrt(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut y = x;
    for _ in 0..5 {
        y = (y + x / y) * 0.5;
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NonZeroU32;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn test_h_gate() {
        let mut sim = CpuBackend::new(1);
        sim.apply_h(0);
        assert!(approx_eq(sim.state.re[0], INV_SQRT_2));
        assert!(approx_eq(sim.state.re[1], INV_SQRT_2));
        assert!(approx_eq(sim.state.im[0], 0.0));
        assert!(approx_eq(sim.state.im[1], 0.0));
    }

    #[test]
    fn test_x_gate() {
        let mut sim = CpuBackend::new(1);
        sim.apply_x(0);
        assert!(approx_eq(sim.state.re[0], 0.0));
        assert!(approx_eq(sim.state.re[1], 1.0));
    }

    #[test]
    fn test_h_then_x() {
        let mut sim = CpuBackend::new(1);
        sim.apply_h(0);
        sim.apply_x(0);
        // X(H|0⟩) = (|1⟩ + |0⟩)/√2
        assert!(approx_eq(sim.state.re[0], INV_SQRT_2));
        assert!(approx_eq(sim.state.re[1], INV_SQRT_2));
    }

    #[test]
    fn test_cnot_bell_state() {
        let mut sim = CpuBackend::new(2);
        sim.apply_h(0);
        sim.apply_cnot(0, 1);
        // Should be (|00⟩ + |11⟩)/√2
        assert!(approx_eq(sim.state.re[0], INV_SQRT_2));
        assert!(approx_eq(sim.state.re[3], INV_SQRT_2));
        assert!(approx_eq(sim.state.re[1], 0.0));
        assert!(approx_eq(sim.state.re[2], 0.0));
    }

    #[test]
    fn test_toffoli() {
        let mut sim = CpuBackend::new(3);
        // Set |110⟩ (qubits 0 and 1 = 1, qubit 2 = 0)
        sim.apply_x(0);
        sim.apply_x(1);
        // |110⟩ → index 3 (binary 011 on qubits 2,1,0 = 110 = 3)
        assert!(approx_eq(sim.state.re[3], 1.0));
        sim.apply_toff(0, 1, 2);
        // |110⟩ → |111⟩
        assert!(approx_eq(sim.state.re[7], 1.0));
        assert!(approx_eq(sim.state.re[3], 0.0));
    }

    #[test]
    fn test_measurement_conserves_probability() {
        let mut sim = CpuBackend::new(1);
        sim.apply_h(0);
        let p0 = sim.probability(0);
        assert!(approx_eq(p0, 0.5));
    }

    #[test]
    fn test_sample_distribution() {
        let mut sim = CpuBackend::new(1);
        sim.apply_h(0);
        let mut rng = Lcg64::new(42);
        let mut counts = [0u32; 2];
        for _ in 0..10_000 {
            let sample = sim.sample(&mut rng);
            counts[sample] += 1;
        }
        // ~5000 each, within 5% tolerance
        assert!((counts[0] as f64 / 10000.0 - 0.5).abs() < 0.05);
        assert!((counts[1] as f64 / 10000.0 - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_full_program_execution() {
        let mut classical = [0u8; 10];
        let mut histogram = vec![0u32; 4];
        let mut rng = Lcg64::new(42);

        let mut prog = Program::new();
        prog.num_qubits = 2;
        prog.num_classical = 2;
        prog.emit(IrOp::H(0));
        prog.emit(IrOp::CNOT(0, 1));
        // No SHOT → single execution + sample

        run_program(&prog, &mut classical, &mut histogram, &mut rng);

        // With 2 qubit state, only |00⟩ + |11⟩, so histogram[0] and histogram[3]
        let total: u32 = histogram.iter().sum();
        assert_eq!(total, 1); // single shot
        assert!(histogram[0] == 1 || histogram[3] == 1);
    }

    #[test]
    fn test_shot_with_body() {
        let mut classical = [0u8; 10];
        let mut histogram = vec![0u32; 2];
        let mut rng = Lcg64::new(42);

        let mut prog = Program::new();
        prog.num_qubits = 1;
        prog.num_classical = 1;
        prog.has_explicit_shot = true;
        prog.emit(IrOp::Shot(NonZeroU32::new(100).unwrap(), 1, 1));
        prog.emit(IrOp::H(0));

        run_program(&prog, &mut classical, &mut histogram, &mut rng);

        let total: u32 = histogram.iter().sum();
        assert_eq!(total, 100);
        // ~50/50 split within tolerance
        assert!((histogram[0] as f64 / 100.0 - 0.5).abs() < 0.15);
        assert!((histogram[1] as f64 / 100.0 - 0.5).abs() < 0.15);
    }

    #[test]
    fn test_run_program_shots() {
        let mut classical = [0u8; 10];
        let mut histogram = vec![0u32; 2];
        let mut rng = Lcg64::new(7);

        let mut prog = Program::new();
        prog.num_qubits = 1;
        prog.num_classical = 1;
        prog.emit(IrOp::H(0));

        run_program_shots(&prog, 5000, &mut classical, &mut histogram, &mut rng);

        let total: u32 = histogram.iter().sum();
        assert_eq!(total, 5000);
        assert!((histogram[0] as f64 / 5000.0 - 0.5).abs() < 0.05);
        assert!((histogram[1] as f64 / 5000.0 - 0.5).abs() < 0.05);
    }
}
