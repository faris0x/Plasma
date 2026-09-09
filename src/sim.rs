// Copyright (c) 2026 Faris Alfarhan
// SPDX-License-Identifier: GPL-3.0-only

use alloc::vec;
use alloc::vec::Vec;

use super::fusion::{C2, C4, FusedOp};
use super::ir::{IrOp, Program};
use super::math::{sin_cos, sqrt};
use super::rng::Lcg64;
use crate::stabilizer::StabilizerBackend;
use crate::mps::MpsBackend;

/// Normalisation factor 1/√2 for Hadamard gates.
const INV_SQRT_2: f64 = 0.7071067811865476;

/// Seed for initial state: all amplitude in |0...0>.
const GROUND_STATE: usize = 0;

/// Jump-based subroutine remap: `remap[k]` is the actual qubit bound to the
/// subprogram's formal parameter `k` (qubit operand `k` inside the body).
/// `[None; 3]` means "not inside a subprogram".
pub type Remap = [Option<u8>; 3];

/// Translate a qubit operand through the current remap (parameters 0..2 only).
pub fn translate_q(q: u8, remap: &Remap) -> u8 {
    if (q as usize) < remap.len() {
        remap[q as usize].unwrap_or(q)
    } else {
        q
    }
}

/// Rebuild a qubit-bearing op with operands translated through the remap.
/// Non-qubit ops pass through unchanged.
pub fn translate_op(op: IrOp, remap: &Remap) -> IrOp {
    match op {
        IrOp::H(q) => IrOp::H(translate_q(q, remap)),
        IrOp::X(q) => IrOp::X(translate_q(q, remap)),
        IrOp::CNOT(c, t) => IrOp::CNOT(translate_q(c, remap), translate_q(t, remap)),
        IrOp::Toff(a, b, t) => IrOp::Toff(
            translate_q(a, remap),
            translate_q(b, remap),
            translate_q(t, remap),
        ),
        IrOp::RZ(q, k) => IrOp::RZ(translate_q(q, remap), k),
        IrOp::RX(q, k) => IrOp::RX(translate_q(q, remap), k),
        IrOp::RY(q, k) => IrOp::RY(translate_q(q, remap), k),
        IrOp::Phase(q, k) => IrOp::Phase(translate_q(q, remap), k),
        IrOp::S(q) => IrOp::S(translate_q(q, remap)),
        IrOp::T(q) => IrOp::T(translate_q(q, remap)),
        IrOp::SX(q) => IrOp::SX(translate_q(q, remap)),
        IrOp::SWAP(a, b) => IrOp::SWAP(translate_q(a, remap), translate_q(b, remap)),
        IrOp::ISWAP(a, b) => IrOp::ISWAP(translate_q(a, remap), translate_q(b, remap)),
        IrOp::CZ(a, b) => IrOp::CZ(translate_q(a, remap), translate_q(b, remap)),
        IrOp::CPHASE(a, b, k) => IrOp::CPHASE(translate_q(a, remap), translate_q(b, remap), k),
        IrOp::CSWAP(a, b, c) => IrOp::CSWAP(
            translate_q(a, remap),
            translate_q(b, remap),
            translate_q(c, remap),
        ),
        IrOp::MCX(mask, t) => IrOp::MCX(mask, translate_q(t, remap)),
        IrOp::Reset(q) => IrOp::Reset(translate_q(q, remap)),
        IrOp::Measure(q, c) => IrOp::Measure(translate_q(q, remap), c),
        IrOp::MeasureX(q, c) => IrOp::MeasureX(translate_q(q, remap), c),
        IrOp::MeasureY(q, c) => IrOp::MeasureY(translate_q(q, remap), c),
        IrOp::Call(id, a0, a1, a2) => IrOp::Call(
            id,
            translate_q(a0, remap),
            translate_q(a1, remap),
            translate_q(a2, remap),
        ),
        other => other,
    }
}


/// Readout-error mitigation via linear inversion of the single-qubit
/// confusion matrix M1 = [[1-p, p], [p, 1-p]] (independent per qubit, so the
/// joint confusion is the tensor product). c_true = M^-1 c_obs is computed
/// with a per-qubit butterfly transform, then rounded back to integer counts.
/// Requires p < 0.5 (M invertible).
pub fn mitigate_readout(hist: &mut [u32], num_qubits: u8, p: f64) {
    if p <= 0.0 || p >= 0.5 {
        return;
    }
    let a = (1.0 - p) / (1.0 - 2.0 * p);
    let b = -p / (1.0 - 2.0 * p);
    let n = hist.len();
    let mut v: alloc::vec::Vec<f64> = hist.iter().map(|&c| c as f64).collect();
    for bit in 0..num_qubits {
        let step = 1usize << bit;
        for i in (0..n).step_by(2 * step) {
            for j in 0..step {
                let lo = i + j;
                let hi = lo + step;
                let c_lo = v[lo];
                let c_hi = v[hi];
                v[lo] = a * c_lo + b * c_hi;
                v[hi] = b * c_lo + a * c_hi;
            }
        }
    }
    for (h, &x) in hist.iter_mut().zip(v.iter()) {
        *h = round_count(x);
    }
}

/// Round a non-negative f64 to u32 (half up); `round` is unavailable in the
/// crate's no_std core, so truncate via integer cast and add the half.
fn round_count(x: f64) -> u32 {
    if x <= 0.0 {
        return 0;
    }
    let t = x as u64;
    if x - t as f64 >= 0.5 {
        (t + 1).min(u32::MAX as u64) as u32
    } else {
        t as u32
    }
}
/// True when the remap is active (executing inside a subprogram body).
pub fn remap_active(remap: &Remap) -> bool {
    remap[0].is_some() || remap[1].is_some() || remap[2].is_some()
}

/// Observable results captured by `EXPECT` and `SAVE_*` ops during a run.
#[derive(Clone, Debug, Default)]
pub struct Results {
    /// `<ψ|P|ψ>` per `EXPECT`, in program order.
    pub expectations: Vec<f64>,
    /// `<H>` per `ESTIMATE` (weighted Pauli sum), in program order.
    pub estimates: Vec<f64>,
    /// One statevector (re, im interleaved) per `SAVE_STATEVECTOR`.
    pub saved_states: Vec<Vec<f64>>,
    /// One set of `(index, re, im)` triples per `SAVE_AMPLITUDES`.
    pub saved_amplitudes: Vec<Vec<(usize, f64, f64)>>,
    /// One set of basis probabilities per `SAVE_PROBABILITIES`.
    pub saved_probs: Vec<Vec<f64>>,
}

/// Predict the accumulated *relative* amplitude error of the f32 GPU path
/// from the number of gate applications, using a random-walk accumulation
/// model: each gate rounds every amplitude it touches by ~f32-ε at each
/// of a handful of operations, and errors add in quadrature over gates.
///
///   err ~ c * ε_f32 * sqrt(gates),   c ~ 4,  ε_f32 ~ 1.19e-7
///
/// The constant is calibrated against observed CPU/GPU divergence on deep
/// circuits and validated by `tests/golden.py`. Used by `--precision auto`
/// to decide whether the fast f32 GPU path is accurate enough.
pub fn estimate_f32_amp_error(gates: usize) -> f64 {
    const F32_EPS: f64 = 1.19e-7;
    const FLOP_FACTOR: f64 = 4.0;
    FLOP_FACTOR * F32_EPS * super::math::sqrt(gates.max(1) as f64)
}

#[derive(Clone, Debug)]
pub struct State {
    pub re: Vec<f64>,
    pub im: Vec<f64>,
}

impl State {
    /// Allocate a zeroed state of `n` amplitudes and set |0...0>.
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

/// Sampled-Kraus noise model. All draws come from the seeded host RNG in a
/// fixed per-gate order (depolarizing, then amplitude damping, then phase
/// damping), so noisy runs are bit-reproducible for a given seed. Applied
/// after each qubit gate on the (non-fused) CPU reference path; the GPU and
/// fused paths do not apply noise (`--noise` forces the CPU reference).
#[derive(Clone, Copy, Debug, Default)]
pub struct NoiseModel {
    /// Per single-qubit-gate depolarizing rate: with prob `depolarizing` the
    /// qubit gets a random Pauli (X/Y/Z) after the gate.
    pub depolarizing: f64,
    /// Readout error: a measurement result flips with prob `readout`.
    pub readout: f64,
    /// Amplitude damping: after each single-qubit gate, with prob
    /// `amp_damping` the qubit's |1> population relaxes to |0> (jump), else
    /// the |1> amplitudes scale by sqrt(1-amp_damping).
    pub amp_damping: f64,
    /// Phase damping (dephasing): after each single-qubit gate, with prob
    /// `phase_damping` a Z is applied.
    pub phase_damping: f64,
}

impl NoiseModel {
    pub fn is_active(&self) -> bool {
        self.depolarizing > 0.0
            || self.readout > 0.0
            || self.amp_damping > 0.0
            || self.phase_damping > 0.0
    }
}

/// Backend interface used by the generic executor: implemented by the
/// statevector `CpuBackend` and the stabilizer (Clifford) engine.
pub trait SimBackend {
    fn reset_backend(num_qubits: u8, noise: NoiseModel) -> Self;
    fn reset(&mut self);
    fn apply_h(&mut self, q: u8);
    fn apply_x(&mut self, q: u8);
    fn apply_cnot(&mut self, c: u8, t: u8);
    fn apply_toff(&mut self, c1: u8, c2: u8, t: u8);
    fn apply_rz(&mut self, q: u8, theta: f64);
    fn apply_rx(&mut self, q: u8, theta: f64);
    fn apply_ry(&mut self, q: u8, theta: f64);
    fn apply_phase(&mut self, q: u8, theta: f64);
    fn apply_s(&mut self, q: u8);
    fn apply_t(&mut self, q: u8);
    fn apply_sx(&mut self, q: u8);
    fn apply_swap(&mut self, a: u8, b: u8);
    fn apply_iswap(&mut self, a: u8, b: u8);
    fn apply_cz(&mut self, a: u8, b: u8);
    fn apply_cphase(&mut self, a: u8, b: u8, theta: f64);
    fn apply_cswap(&mut self, c: u8, b: u8, t: u8);
    fn apply_mcx(&mut self, mask: u32, t: u8);
    fn reset_qubit(&mut self, q: u8, rng: &mut Lcg64);
    fn measure(&mut self, q: u8, rng: &mut Lcg64) -> u8;
    fn measure_x(&mut self, q: u8, rng: &mut Lcg64) -> u8;
    fn measure_y(&mut self, q: u8, rng: &mut Lcg64) -> u8;
    fn sample(&mut self, rng: &mut Lcg64) -> usize;
    fn expect_value(&self, pauli: u64) -> f64;
    fn save_state(&mut self, results: &mut Results);
    fn save_amplitudes(&mut self, results: &mut Results);
    fn save_probabilities(&mut self, results: &mut Results);
    /// Post-gate noise (no-op for the stabilizer engine).
    fn noise_after_single(&mut self, _q: u8, _rng: &mut Lcg64) {}
    fn noise_after_pair(&mut self, _a: u8, _b: u8, _rng: &mut Lcg64) {}
    fn noise_after_triple(&mut self, _a: u8, _b: u8, _c: u8, _rng: &mut Lcg64) {}
}

pub struct CpuBackend {
    state: State,
    noise: NoiseModel,
}

impl CpuBackend {
    pub fn new(num_qubits: u8) -> Self {
        let n = 1usize << num_qubits;
        let backend = Self {
            state: State::new(n),
            noise: NoiseModel::default(),
        };
        backend
    }

    /// Construct with a sampled-Kraus noise model applied after each gate.
    pub fn with_noise(num_qubits: u8, noise: NoiseModel) -> Self {
        let mut b = Self::new(num_qubits);
        b.noise = noise;
        b
    }

    pub fn reset(&mut self) {
        self.state.reset();
    }

    // Noise (sampled-Kraus)

    /// Apply a single-qubit Pauli to the state: 1=X, 2=Y, 3=Z.
    fn apply_pauli(&mut self, target: u8, pauli: u8) {
        match pauli {
            1 => self.apply_x(target),
            2 => {
                // Y = i X Z up to global phase on a single qubit; for a
                // Pauli channel the global phase is irrelevant.
                self.apply_z(target);
                self.apply_x(target);
            }
            _ => self.apply_z(target),
        }
    }

    /// Z gate (Z|1> = -|1>) used by the Pauli/phase-damping channels.
    pub fn apply_z(&mut self, target: u8) {
        self.for_each_pair(target, |sim, _i, j| {
            sim.state.re[j] = -sim.state.re[j];
            sim.state.im[j] = -sim.state.im[j];
        });
    }

    /// Amplitude-damping sampled Kraus on qubit `target` (rate `γ`).
    ///
    /// Correct pure-state sampling: jump with probability `γ * P(|1>)`;
    /// on a jump the |1> population moves to |0> (K1); otherwise the |1>
    /// amplitudes scale by sqrt(1-γ) (K0). Both branches renormalize so
    /// the state stays a valid pure state for subsequent gates.
    fn apply_amp_damp(&mut self, target: u8, gamma: f64, rng: &mut Lcg64) {
        let p1 = self.probability(target);
        let pj = gamma * p1;
        if pj <= 0.0 {
            return;
        }
        let jump = rng.next_f64() < pj;
        if jump {
            // K1: |1> -> sqrt(γ)|0>, then normalize by sqrt(pj).
            let scale = crate::math::sqrt(gamma / pj);
            self.for_each_pair(target, |sim, i, j| {
                sim.state.re[i] += sim.state.re[j] * scale;
                sim.state.im[i] += sim.state.im[j] * scale;
                sim.state.re[j] = 0.0;
                sim.state.im[j] = 0.0;
            });
        } else {
            // K0: |1> -> sqrt(1-γ)|1>, normalize by sqrt(1 - pj).
            let s = crate::math::sqrt(1.0 - gamma);
            let norm = crate::math::sqrt(1.0 - pj);
            self.for_each_pair(target, |sim, _i, j| {
                sim.state.re[j] *= s / norm;
                sim.state.im[j] *= s / norm;
            });
            self.for_each_pair(target, |sim, i, _j| {
                sim.state.re[i] /= norm;
                sim.state.im[i] /= norm;
            });
        }
    }

    /// Apply the post-gate noise channels for a single-qubit gate. Draw order
    /// is fixed (depolarizing, amplitude damping, phase damping) so a given
    /// seed reproduces the same noise trajectory.
    pub fn noise_after_single(&mut self, q: u8, rng: &mut Lcg64) {
        if !self.noise.is_active() {
            return;
        }
        let nd = self.noise.depolarizing;
        if nd > 0.0 && rng.next_f64() < nd {
            // Random Pauli: X, Y, or Z with equal probability.
            let k = (rng.next_f64() * 3.0) as u64;
            self.apply_pauli(q, (1 + k) as u8);
        }
        let na = self.noise.amp_damping;
        if na > 0.0 {
            self.apply_amp_damp(q, na, rng);
        }
        let np = self.noise.phase_damping;
        if np > 0.0 && rng.next_f64() < np {
            self.apply_z(q);
        }
    }

    /// Post-gate noise for a two-qubit gate: apply to each qubit in a fixed
    /// order (control then target) for reproducibility.
    pub fn noise_after_pair(&mut self, a: u8, b: u8, rng: &mut Lcg64) {
        self.noise_after_single(a, rng);
        self.noise_after_single(b, rng);
    }

    /// Three-qubit gate noise (Toffoli/CSWAP): fixed order.
    pub fn noise_after_triple(&mut self, a: u8, b: u8, c: u8, rng: &mut Lcg64) {
        self.noise_after_single(a, rng);
        self.noise_after_single(b, rng);
        self.noise_after_single(c, rng);
    }




// Single-qubit gates

    /// Visit every amplitude pair `(i, j)` with `j = i ^ bit(target)`, where
    /// `i` is the low member (bit clear). Uses a blocked, branch-free inner
    /// loop for `bit >= 8` (vectorization-friendly for mid/high qubits) and a
    /// simple single loop for low qubits, where the blocked form's outer-loop
    /// overhead outweighs the win. Bit-identical arithmetic either way.
    #[inline]
    fn for_each_pair(&mut self, target: u8, f: impl Fn(&mut Self, usize, usize)) {
        let bit = 1usize << target;
        let n = self.state.re.len();
        if bit >= 8 {
            let block = bit << 1;
            for base in (0..n).step_by(block) {
                for i in base..base + bit {
                    f(self, i, i + bit);
                }
            }
        } else {
            for i in 0..n {
                if i & bit != 0 {
                    continue;
                }
                f(self, i, i ^ bit);
            }
        }
    }

    pub fn apply_h(&mut self, target: u8) {
        self.for_each_pair(target, |sim, i, j| {
            let re_i = sim.state.re[i];
            let im_i = sim.state.im[i];
            let re_j = sim.state.re[j];
            let im_j = sim.state.im[j];
            sim.state.re[i] = INV_SQRT_2 * (re_i + re_j);
            sim.state.im[i] = INV_SQRT_2 * (im_i + im_j);
            sim.state.re[j] = INV_SQRT_2 * (re_i - re_j);
            sim.state.im[j] = INV_SQRT_2 * (im_i - im_j);
        });
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

    // Multi-qubit gates

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

    // v0.2 gates

    /// RZ(θ) = diag(e^{-iθ/2}, e^{iθ/2}).
    pub fn apply_rz(&mut self, target: u8, theta: f64) {
        let (sn, cs) = sin_cos(theta * 0.5);
        self.for_each_pair(target, |sim, i, j| {
            let (re_i, im_i) = (sim.state.re[i], sim.state.im[i]);
            let (re_j, im_j) = (sim.state.re[j], sim.state.im[j]);
            sim.state.re[i] = re_i * cs + im_i * sn;
            sim.state.im[i] = im_i * cs - re_i * sn;
            sim.state.re[j] = re_j * cs - im_j * sn;
            sim.state.im[j] = im_j * cs + re_j * sn;
        });
    }

    /// RX(θ) = [[cos, -i sin],[-i sin, cos]] (θ/2 entries).
    pub fn apply_rx(&mut self, target: u8, theta: f64) {
        let (sn, cs) = sin_cos(theta * 0.5);
        self.for_each_pair(target, |sim, i, j| {
            let (re_i, im_i) = (sim.state.re[i], sim.state.im[i]);
            let (re_j, im_j) = (sim.state.re[j], sim.state.im[j]);
            sim.state.re[i] = cs * re_i + sn * im_j;
            sim.state.im[i] = cs * im_i - sn * re_j;
            sim.state.re[j] = cs * re_j + sn * im_i;
            sim.state.im[j] = cs * im_j - sn * re_i;
        });
    }

    /// RY(θ) = [[cos, -sin],[sin, cos]] (θ/2 entries).
    pub fn apply_ry(&mut self, target: u8, theta: f64) {
        let (sn, cs) = sin_cos(theta * 0.5);
        self.for_each_pair(target, |sim, i, j| {
            let (re_i, im_i) = (sim.state.re[i], sim.state.im[i]);
            let (re_j, im_j) = (sim.state.re[j], sim.state.im[j]);
            sim.state.re[i] = cs * re_i - sn * re_j;
            sim.state.im[i] = cs * im_i - sn * im_j;
            sim.state.re[j] = sn * re_i + cs * re_j;
            sim.state.im[j] = sn * im_i + cs * im_j;
        });
    }

    /// Phase(θ) = diag(1, e^{iθ}).
    pub fn apply_phase(&mut self, target: u8, theta: f64) {
        let (s, c) = sin_cos(theta);
        self.for_each_pair(target, |sim, _i, j| {
            let (re_j, im_j) = (sim.state.re[j], sim.state.im[j]);
            sim.state.re[j] = re_j * c - im_j * s;
            sim.state.im[j] = re_j * s + im_j * c;
        });
    }

    /// S = diag(1, i): multiply |1> amplitudes by i.
    pub fn apply_s(&mut self, target: u8) {
        self.for_each_pair(target, |sim, _i, j| {
            let (re_j, im_j) = (sim.state.re[j], sim.state.im[j]);
            sim.state.re[j] = -im_j;
            sim.state.im[j] = re_j;
        });
    }

    /// T = diag(1, e^{iπ/4}).
    pub fn apply_t(&mut self, target: u8) {
        self.for_each_pair(target, |sim, _i, j| {
            let (re_j, im_j) = (sim.state.re[j], sim.state.im[j]);
            sim.state.re[j] = INV_SQRT_2 * (re_j - im_j);
            sim.state.im[j] = INV_SQRT_2 * (re_j + im_j);
        });
    }

    /// SX = sqrt(X) = (1/2)[(1+i)I + (1-i)X].
    pub fn apply_sx(&mut self, target: u8) {
        self.for_each_pair(target, |sim, i, j| {
            let (re_i, im_i) = (sim.state.re[i], sim.state.im[i]);
            let (re_j, im_j) = (sim.state.re[j], sim.state.im[j]);
            // (1+i)a = (re-im) + i(re+im);  (1-i)b = (re+im) + i(im-re)
            sim.state.re[i] = ((re_i - im_i) + (re_j + im_j)) * 0.5;
            sim.state.im[i] = ((re_i + im_i) + (im_j - re_j)) * 0.5;
            sim.state.re[j] = ((re_i + im_i) + (re_j - im_j)) * 0.5;
            sim.state.im[j] = ((im_i - re_i) + (re_j + im_j)) * 0.5;
        });
    }

    /// SWAP |a> |b> (single-writer: only the a=1, b=0 member acts).
    pub fn apply_swap(&mut self, a: u8, b: u8) {
        let abit = 1 << a;
        let bbit = 1 << b;
        let n = self.state.re.len();
        for i in 0..n {
            if i & abit == 0 || i & bbit != 0 {
                continue;
            }
            let j = i ^ abit ^ bbit;
            self.state.re.swap(i, j);
            self.state.im.swap(i, j);
        }
    }

    /// iSWAP: swap with an i phase on the exchanged term.
    pub fn apply_iswap(&mut self, a: u8, b: u8) {
        let abit = 1 << a;
        let bbit = 1 << b;
        let n = self.state.re.len();
        for i in 0..n {
            if i & abit == 0 || i & bbit != 0 {
                continue;
            }
            let j = i ^ abit ^ bbit;
            let (re_i, im_i) = (self.state.re[i], self.state.im[i]);
            let (re_j, im_j) = (self.state.re[j], self.state.im[j]);
            // a_i' = i·a_j, a_j' = i·a_i  (i·z = -im + i·re)
            self.state.re[i] = -im_j;
            self.state.im[i] = re_j;
            self.state.re[j] = -im_i;
            self.state.im[j] = re_i;
        }
    }

    /// CZ: phase -1 on |11>.
    pub fn apply_cz(&mut self, a: u8, b: u8) {
        let mask = (1 << a) | (1 << b);
        let n = self.state.re.len();
        for i in 0..n {
            if i & mask == mask {
                self.state.re[i] = -self.state.re[i];
                self.state.im[i] = -self.state.im[i];
            }
        }
    }

    /// CPHASE(θ): phase e^{iθ} on |11>.
    pub fn apply_cphase(&mut self, a: u8, b: u8, theta: f64) {
        let (s, c) = sin_cos(theta);
        let mask = (1 << a) | (1 << b);
        let n = self.state.re.len();
        for i in 0..n {
            if i & mask == mask {
                let (re, im) = (self.state.re[i], self.state.im[i]);
                self.state.re[i] = re * c - im * s;
                self.state.im[i] = re * s + im * c;
            }
        }
    }

    /// CSWAP (Fredkin): |ctl> controls the swap of |b> and |c>.
    pub fn apply_cswap(&mut self, ctl: u8, b: u8, c: u8) {
        let cbit = 1 << ctl;
        let bbit = 1 << b;
        let cbit2 = 1 << c;
        let n = self.state.re.len();
        for i in 0..n {
            if i & cbit == 0 || i & bbit == 0 || i & cbit2 != 0 {
                continue;
            }
            let j = i ^ bbit ^ cbit2;
            self.state.re.swap(i, j);
            self.state.im.swap(i, j);
        }
    }

    /// MCX: flip |t> when every bit of the control mask is set.
    pub fn apply_mcx(&mut self, mask: u32, target: u8) {
        let tbit = 1 << target;
        let n = self.state.re.len();
        for i in 0..n {
            if (i as u32) & mask != mask {
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

    /// Apply an arbitrary 2×2 unitary (from gate fusion) to qubit `q`.
    pub fn apply_u1(&mut self, q: u8, m: &C2) {
        self.for_each_pair(q, |sim, i, j| {
            let (re_i, im_i) = (sim.state.re[i], sim.state.im[i]);
            let (re_j, im_j) = (sim.state.re[j], sim.state.im[j]);
            sim.state.re[i] = m.re[0][0] * re_i - m.im[0][0] * im_i
                + m.re[0][1] * re_j - m.im[0][1] * im_j;
            sim.state.im[i] = m.re[0][0] * im_i + m.im[0][0] * re_i
                + m.re[0][1] * im_j + m.im[0][1] * re_j;
            sim.state.re[j] = m.re[1][0] * re_i - m.im[1][0] * im_i
                + m.re[1][1] * re_j - m.im[1][1] * im_j;
            sim.state.im[j] = m.re[1][0] * im_i + m.im[1][0] * re_i
                + m.re[1][1] * im_j + m.im[1][1] * re_j;
        });
    }

    /// Apply an arbitrary 4×4 unitary (from gate fusion) to qubits (q0, q1),
    /// in the basis `s = bit(q0) + 2·bit(q1)`.
    pub fn apply_u2(&mut self, q0: u8, q1: u8, m: &C4) {
        let bit0 = 1 << q0;
        let bit1 = 1 << q1;
        let bits = bit0 | bit1;
        let n = self.state.re.len();
        for base in 0..n {
            if base & bits != 0 {
                continue;
            }
            let idx = [base, base | bit0, base | bit1, base | bit0 | bit1];
            let src = [
                (self.state.re[idx[0]], self.state.im[idx[0]]),
                (self.state.re[idx[1]], self.state.im[idx[1]]),
                (self.state.re[idx[2]], self.state.im[idx[2]]),
                (self.state.re[idx[3]], self.state.im[idx[3]]),
            ];
            for s in 0..4 {
                let mut nre = 0.0;
                let mut nim = 0.0;
                for t in 0..4 {
                    nre += m.re[s][t] * src[t].0 - m.im[s][t] * src[t].1;
                    nim += m.re[s][t] * src[t].1 + m.im[s][t] * src[t].0;
                }
                self.state.re[idx[s]] = nre;
                self.state.im[idx[s]] = nim;
            }
        }
    }

    // Measurement

    /// S† = diag(1, -i): multiply |1> amplitudes by -i.
    pub fn apply_sdg(&mut self, target: u8) {
        self.for_each_pair(target, |sim, _i, j| {
            let (re_j, im_j) = (sim.state.re[j], sim.state.im[j]);
            sim.state.re[j] = im_j;
            sim.state.im[j] = -re_j;
        });
    }

    /// RESET q: collapse to |0> (measure, then correct with X if needed).
    pub fn reset_qubit(&mut self, qubit: u8, rng: &mut Lcg64) {
        let outcome = self.measure(qubit, rng);
        if outcome == 1 {
            self.apply_x(qubit);
        }
    }

    /// Measure |q> in the X basis (rotate with H, measure Z, rotate back).
    pub fn measure_x(&mut self, qubit: u8, rng: &mut Lcg64) -> u8 {
        self.apply_h(qubit);
        let outcome = self.measure(qubit, rng);
        self.apply_h(qubit);
        outcome
    }

    /// Measure |q> in the Y basis (rotate with S† then H, measure Z, rotate
    /// back with H then S).
    pub fn measure_y(&mut self, qubit: u8, rng: &mut Lcg64) -> u8 {
        self.apply_sdg(qubit);
        self.apply_h(qubit);
        let outcome = self.measure(qubit, rng);
        self.apply_h(qubit);
        self.apply_s(qubit);
        outcome
    }

    pub fn probability(&self, qubit: u8) -> f64 {
        // Plain f64 summation: at MAX_QUBITS = 28 the drift is ~1e-12
        // relative, so no compensation is needed here. The GPU path uses
        // Kahan summation because it accumulates in f32; a hypothetical f32
        // CPU path would need the same treatment.
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
        let p_true = self.probability(qubit);
        // Readout error: the detector's report-1 probability is the noisy
        // mixture (1-r)*P1 + r*P0, and the state collapses to the reported
        // outcome (a consistent back-action model). With r=0 this reduces to
        // the ideal measurement, keeping the deterministic contract.
        let p = if self.noise.readout > 0.0 {
            p_true * (1.0 - self.noise.readout) + (1.0 - p_true) * self.noise.readout
        } else {
            p_true
        };
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
            let norm = sqrt(norm_sq);
            for i in 0..n {
                if ((i >> qubit) & 1) as u8 == outcome {
                    self.state.re[i] /= norm;
                    self.state.im[i] /= norm;
                }
            }
        }
    }

    // Sampling

    /// Sample a single basis state according to the Born rule.
    pub fn sample(&mut self, rng: &mut Lcg64) -> usize {
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

    // Observables (M5)

    /// `<ψ|P|ψ>` for a Pauli product P encoded as 2 bits per qubit
    /// (I=00, X=01, Y=10, Z=11). Computed directly from the amplitudes
    /// (no sampling). Mathematically exact up to f64 rounding.
    ///
    /// For P acting with flip set F (qubits with X/Y) and phase qubits
    /// Z (sign) and Y (`±i`): <ψ|P|ψ> = Σ_i conj(a_i)·c(i)·a_{i^F}
    /// where c(i) = (-1)^{popcount(i&Z)} · (-i)^{|Y|} · (-1)^{popcount(i&Y)}.
    pub fn expect_value(&self, pauli: u64) -> f64 {
        let mut flip_mask = 0u64;
        let mut z_mask = 0u64;
        let mut y_mask = 0u64;
        let mut ny = 0u32;
        for q in 0..32u64 {
            match (pauli >> (2 * q)) & 3 {
                1 => flip_mask |= 1 << q,               // X: flip
                2 => {
                    flip_mask |= 1 << q;
                    y_mask |= 1 << q;
                    ny += 1;
                }                                       // Y: flip + phase
                3 => z_mask |= 1 << q,                  // Z: sign
                _ => {}                                 // I
            }
        }
        // (-i)^|Y|
        let (yb_re, yb_im) = match ny & 3 {
            0 => (1.0, 0.0),
            1 => (0.0, -1.0),
            2 => (-1.0, 0.0),
            _ => (0.0, 1.0),
        };
        let n = self.state.re.len();
        let zm = z_mask as usize;
        let ym = y_mask as usize;
        let mut acc_re = 0.0;
        let mut acc_im = 0.0;
        for i in 0..n {
            let j = (i as u64 ^ flip_mask) as usize;
            let sign = if (i & zm).count_ones() & 1 == 1 { -1.0 } else { 1.0 };
            let ysign = if (i & ym).count_ones() & 1 == 1 { -1.0 } else { 1.0 };
            let c_re = sign * yb_re * ysign;
            let c_im = sign * yb_im * ysign;
            let re_i = self.state.re[i];
            let im_i = self.state.im[i];
            let re_j = self.state.re[j];
            let im_j = self.state.im[j];
            // conj(a_i) * a_j
            let prod_re = re_i * re_j + im_i * im_j;
            let prod_im = re_i * im_j - im_i * re_j;
            acc_re += c_re * prod_re - c_im * prod_im;
            acc_im += c_re * prod_im + c_im * prod_re;
        }
        // <ψ|P|ψ> is real for Hermitian P; take the real part.
        let _ = acc_im;
        acc_re
    }

    /// Capture the current statevector into `results` (re, im interleaved).
    pub fn save_state(&self, results: &mut Results) {
        let mut v = Vec::with_capacity(2 * self.state.re.len());
        for i in 0..self.state.re.len() {
            v.push(self.state.re[i]);
            v.push(self.state.im[i]);
        }
        results.saved_states.push(v);
    }

    /// Capture the current amplitudes into `results` as (index, re, im).
    pub fn save_amplitudes(&self, results: &mut Results) {
        let mut v = Vec::with_capacity(self.state.re.len());
        for i in 0..self.state.re.len() {
            v.push((i, self.state.re[i], self.state.im[i]));
        }
        results.saved_amplitudes.push(v);
    }

    /// Capture the current basis probabilities into `results`.
    pub fn save_probabilities(&self, results: &mut Results) {
        let mut v = Vec::with_capacity(self.state.re.len());
        for i in 0..self.state.re.len() {
            v.push(self.state.re[i] * self.state.re[i] + self.state.im[i] * self.state.im[i]);
        }
        results.saved_probs.push(v);
    }
}

// Top-level execution

/// Execute a parsed program and fill the histogram and observable results.
pub fn run_program(
    prog: &Program,
    results: &mut Results,
    classical: &mut [u8],
    histogram: &mut [u32],
    rng: &mut Lcg64,
    noise: NoiseModel,
) {
    let mut backend = CpuBackend::with_noise(prog.num_qubits, noise);

    for c in classical.iter_mut() { *c = 0; }
    for h in histogram.iter_mut() { *h = 0; }

    exec_ops(prog, 0, prog.len, &mut backend, results, classical, histogram, rng, &[None; 3]);
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
    results: &mut Results,
    classical: &mut [u8],
    histogram: &mut [u32],
    rng: &mut Lcg64,
    noise: NoiseModel,
) {
    let mut backend = CpuBackend::with_noise(prog.num_qubits, noise);
    for c in classical.iter_mut() { *c = 0; }
    for h in histogram.iter_mut() { *h = 0; }
    for _ in 0..shots {
        backend.reset();
        exec_ops(prog, 0, prog.len, &mut backend, results, classical, histogram, rng, &[None; 3]);
        let sample = backend.sample(rng);
        histogram[sample] += 1;
    }
}

/// Execute a program through its fused gate stream (see `fusion`), producing
/// the same histogram as `run_program` within float tolerance. Used to verify
/// the fusion pass and as the reference for the GPU fused executor.
/// Stabilizer-engine run (Clifford programs only; validated by the adaptive
/// dispatcher before calling). Exact: for the same seed, measurement
/// histograms, classical bits, and EXPECT/ESTIMATE values reproduce the
/// statevector reference bit-for-bit.
pub fn run_stabilizer_program(
    prog: &Program,
    results: &mut Results,
    classical: &mut [u8],
    histogram: &mut [u32],
    rng: &mut Lcg64,
) {
    run_stabilizer_generic(prog, None, results, classical, histogram, rng);
}

/// MPS-engine run with a given bond dimension (approximate when the
/// entanglement exceeds `dmax`; the accumulated truncation error is reported
/// by the caller). Validated against the statevector on low-entanglement
/// circuits.
/// Returns the accumulated SVD truncation error (0 for exact runs).
pub fn run_mps_program(
    prog: &Program,
    dmax: usize,
    results: &mut Results,
    classical: &mut [u8],
    histogram: &mut [u32],
    rng: &mut Lcg64,
) -> f64 {
    let mut backend = MpsBackend::new(prog.num_qubits, dmax);
    for c in classical.iter_mut() {
        *c = 0;
    }
    for h in histogram.iter_mut() {
        *h = 0;
    }
    exec_ops(&prog, 0, prog.len, &mut backend, results, classical, histogram, rng, &[None; 3]);
    if !prog.has_explicit_shot {
        let sample = backend.sample(rng);
        histogram[sample] += 1;
    }
    backend.trunc_err
}

/// Returns the accumulated SVD truncation error across all shots.
pub fn run_mps_program_shots(
    prog: &Program,
    dmax: usize,
    shots: u32,
    results: &mut Results,
    classical: &mut [u8],
    histogram: &mut [u32],
    rng: &mut Lcg64,
) -> f64 {
    let mut err = 0.0;
    let mut backend = MpsBackend::new(prog.num_qubits, dmax);
    for c in classical.iter_mut() {
        *c = 0;
    }
    for h in histogram.iter_mut() {
        *h = 0;
    }
    for _ in 0..shots {
        backend.reset();
        exec_ops(&prog, 0, prog.len, &mut backend, results, classical, histogram, rng, &[None; 3]);
        let sample = backend.sample(rng);
        histogram[sample] += 1;
        err += backend.trunc_err;
    }
    err
}

/// Whether the MPS engine can run the program (no 3-qubit gates, no SAVE_*).
pub fn is_mps_suitable(prog: &Program) -> bool {
    let end = prog.sub_len;
    for i in 0..end {
        match prog.ops[i] {
            IrOp::Toff(..) | IrOp::CSWAP(..) | IrOp::MCX(..) | IrOp::SaveState
            | IrOp::SaveAmps | IrOp::SaveProbs => return false,
            _ => {}
        }
    }
    true
}

/// Stabilizer-engine bare-shot loop.
pub fn run_stabilizer_program_shots(
    prog: &Program,
    shots: u32,
    results: &mut Results,
    classical: &mut [u8],
    histogram: &mut [u32],
    rng: &mut Lcg64,
) {
    run_stabilizer_generic(prog, Some(shots), results, classical, histogram, rng);
}

fn run_stabilizer_generic(
    prog: &Program,
    shots: Option<u32>,
    results: &mut Results,
    classical: &mut [u8],
    histogram: &mut [u32],
    rng: &mut Lcg64,
) {
    let mut backend = StabilizerBackend::new(prog.num_qubits);
    for c in classical.iter_mut() {
        *c = 0;
    }
    for h in histogram.iter_mut() {
        *h = 0;
    }
    match shots {
        Some(n) => {
            for _ in 0..n {
                backend.reset();
                exec_ops(&prog, 0, prog.len, &mut backend, results, classical, histogram, rng, &[None; 3]);
                let sample = backend.sample(rng);
                histogram[sample] += 1;
            }
        }
        None => {
            exec_ops(&prog, 0, prog.len, &mut backend, results, classical, histogram, rng, &[None; 3]);
            if !prog.has_explicit_shot {
                let sample = backend.sample(rng);
                histogram[sample] += 1;
            }
        }
    }
}

/// Whether the executable main region contains any subroutine calls.
/// (Sub bodies are only reachable via CALL, so scanning the main region
/// suffices.) Programs with calls are not fused, the fused stream cannot
/// host sub bodies, so the regular executor handles them.
impl SimBackend for CpuBackend {
fn reset_backend(num_qubits: u8, noise: NoiseModel) -> Self {
    CpuBackend::with_noise(num_qubits, noise)
}
fn reset(&mut self) { self.reset(); }
fn apply_h(&mut self, q: u8) { self.apply_h(q); }
fn apply_x(&mut self, q: u8) { self.apply_x(q); }
fn apply_cnot(&mut self, c: u8, t: u8) { self.apply_cnot(c, t); }
fn apply_toff(&mut self, c1: u8, c2: u8, t: u8) { self.apply_toff(c1, c2, t); }
fn apply_rz(&mut self, q: u8, theta: f64) { self.apply_rz(q, theta); }
fn apply_rx(&mut self, q: u8, theta: f64) { self.apply_rx(q, theta); }
fn apply_ry(&mut self, q: u8, theta: f64) { self.apply_ry(q, theta); }
fn apply_phase(&mut self, q: u8, theta: f64) { self.apply_phase(q, theta); }
fn apply_s(&mut self, q: u8) { self.apply_s(q); }
fn apply_t(&mut self, q: u8) { self.apply_t(q); }
fn apply_sx(&mut self, q: u8) { self.apply_sx(q); }
fn apply_swap(&mut self, a: u8, b: u8) { self.apply_swap(a, b); }
fn apply_iswap(&mut self, a: u8, b: u8) { self.apply_iswap(a, b); }
fn apply_cz(&mut self, a: u8, b: u8) { self.apply_cz(a, b); }
fn apply_cphase(&mut self, a: u8, b: u8, theta: f64) { self.apply_cphase(a, b, theta); }
fn apply_cswap(&mut self, c: u8, b: u8, t: u8) { self.apply_cswap(c, b, t); }
fn apply_mcx(&mut self, mask: u32, t: u8) { self.apply_mcx(mask, t); }
fn reset_qubit(&mut self, q: u8, rng: &mut Lcg64) { self.reset_qubit(q, rng); }
fn measure(&mut self, q: u8, rng: &mut Lcg64) -> u8 { self.measure(q, rng) }
fn measure_x(&mut self, q: u8, rng: &mut Lcg64) -> u8 { self.measure_x(q, rng) }
fn measure_y(&mut self, q: u8, rng: &mut Lcg64) -> u8 { self.measure_y(q, rng) }
fn sample(&mut self, rng: &mut Lcg64) -> usize { self.sample(rng) }
fn expect_value(&self, pauli: u64) -> f64 { self.expect_value(pauli) }
fn save_state(&mut self, results: &mut Results) { CpuBackend::save_state(self, results); }
fn save_amplitudes(&mut self, results: &mut Results) { CpuBackend::save_amplitudes(self, results); }
fn save_probabilities(&mut self, results: &mut Results) { CpuBackend::save_probabilities(self, results); }
fn noise_after_single(&mut self, q: u8, rng: &mut Lcg64) { self.noise_after_single(q, rng); }
fn noise_after_pair(&mut self, a: u8, b: u8, rng: &mut Lcg64) { self.noise_after_pair(a, b, rng); }
fn noise_after_triple(&mut self, a: u8, b: u8, c: u8, rng: &mut Lcg64) { self.noise_after_triple(a, b, c, rng); }
}


/// Whether the program (main region and all sub bodies) is Clifford-only and
/// therefore simulatable exactly by the stabilizer engine. Excluded: rotations
/// and non-Clifford gates (RZ/RX/RY/PHASE/T/SX/TOFF/MCX/CPHASE), the
/// exponential-state ops (SAVE_*), and ISWAP/CSWAP (Clifford but not yet in
/// the engine's gate set).
pub fn is_clifford_program(prog: &Program) -> bool {
    let end = prog.sub_len;
    for i in 0..end {
        match prog.ops[i] {
            IrOp::RZ(..) | IrOp::RX(..) | IrOp::RY(..) | IrOp::Phase(..) | IrOp::T(..)
            | IrOp::SX(..) | IrOp::Toff(..) | IrOp::MCX(..) | IrOp::CPHASE(..)
            | IrOp::ISWAP(..) | IrOp::CSWAP(..) | IrOp::SaveState | IrOp::SaveAmps
            | IrOp::SaveProbs => return false,
            _ => {}
        }
    }
    true
}

pub fn program_has_calls(prog: &Program) -> bool {
    (0..prog.len).any(|i| matches!(prog.ops[i], IrOp::Call(..)))
}

/// Execute a fused program (falls back to the regular executor when the
/// program contains subroutine calls).
pub fn run_fused_program(
    prog: &Program,
    fused: &[FusedOp],
    results: &mut Results,
    classical: &mut [u8],
    histogram: &mut [u32],
    rng: &mut Lcg64,
    noise: NoiseModel,
) {
    if program_has_calls(prog) {
        run_program(prog, results, classical, histogram, rng, noise);
        return;
    }
    let mut backend = CpuBackend::with_noise(prog.num_qubits, noise);
    for c in classical.iter_mut() {
        *c = 0;
    }
    for h in histogram.iter_mut() {
        *h = 0;
    }
    exec_fused_ops(prog, fused, 0, fused.len(), &mut backend, results, classical, histogram, rng, &[None; 3]);
    if !prog.has_explicit_shot {
        let sample = backend.sample(rng);
        histogram[sample] += 1;
    }
}

/// Execute a fused program as a bare-shot loop (mirror of `run_program_shots`;
/// falls back to the regular executor when the program contains calls).
pub fn run_fused_program_shots(
    prog: &Program,
    fused: &[FusedOp],
    shots: u32,
    results: &mut Results,
    classical: &mut [u8],
    histogram: &mut [u32],
    rng: &mut Lcg64,
    noise: NoiseModel,
) {
    if program_has_calls(prog) {
        run_program_shots(prog, shots, results, classical, histogram, rng, noise);
        return;
    }
    let mut backend = CpuBackend::with_noise(prog.num_qubits, noise);
    for c in classical.iter_mut() {
        *c = 0;
    }
    for h in histogram.iter_mut() {
        *h = 0;
    }
    for _ in 0..shots {
        backend.reset();
        exec_fused_ops(prog, fused, 0, fused.len(), &mut backend, results, classical, histogram, rng, &[None; 3]);
        let sample = backend.sample(rng);
        histogram[sample] += 1;
    }
}

/// Execute a fused op stream (the mirror of `exec_ops`).
fn exec_fused_ops(
    prog: &Program,
    fused: &[FusedOp],
    offset: usize,
    end: usize,
    backend: &mut CpuBackend,
    results: &mut Results,
    classical: &mut [u8],
    histogram: &mut [u32],
    rng: &mut Lcg64,
    remap: &Remap,
) {
    let mut i = offset;
    while i < end {
        match &fused[i] {
            FusedOp::U1(q, m) => backend.apply_u1(*q, m),
            FusedOp::U2(q0, q1, m) => backend.apply_u2(*q0, *q1, m),
            FusedOp::Original(op) => {
                let raw = *op;
                let op = if remap_active(remap) {
                    translate_op(raw, remap)
                } else {
                    raw
                };
                match op {
                IrOp::Expect(pauli) => results.expectations.push(backend.expect_value(pauli)),
                IrOp::Estimate(off, len) => {
                    let mut h = 0.0;
                    for k in off as usize..(off as usize + len as usize) {
                        let (c, p) = prog.estimate_terms[k];
                        h += c * backend.expect_value(p as u64);
                    }
                    results.estimates.push(h);
                }
                IrOp::SaveState => backend.save_state(results),
                IrOp::SaveAmps => backend.save_amplitudes(results),
                IrOp::SaveProbs => backend.save_probabilities(results),
                IrOp::Call(sub_id, a0, a1, a2) => {
                    let sub = prog.subs[sub_id as usize];
                    let np = sub.nparams;
                    let mut nremap = [None; 3];
                    let cargs = [a0, a1, a2];
                    for k in 0..np {
                        nremap[k as usize] = Some(cargs[k as usize]);
                    }
                    let (off, len) = (sub.off as usize, sub.len as usize);
                    exec_fused_ops(
                        prog, fused, off, off + len, backend, results, classical,
                        histogram, rng, &nremap,
                    );
                }
                IrOp::Measure(q, c) => {
                    let r = backend.measure(q, rng);
                    classical[c as usize] = r;
                }
                IrOp::MeasureX(q, c) => {
                    let r = backend.measure_x(q, rng);
                    classical[c as usize] = r;
                }
                IrOp::MeasureY(q, c) => {
                    let r = backend.measure_y(q, rng);
                    classical[c as usize] = r;
                }
                IrOp::Reset(q) => backend.reset_qubit(q, rng),
                IrOp::Toff(c1, c2, t) => { backend.apply_toff(c1, c2, t); backend.noise_after_triple(c1, c2, t, rng); }
                IrOp::CSWAP(c, b, t) => { backend.apply_cswap(c, b, t); backend.noise_after_triple(c, b, t, rng); }
                IrOp::MCX(mask, t) => { backend.apply_mcx(mask, t); backend.noise_after_single(t, rng); }
                IrOp::Set(c, v) => classical[c as usize] = v,
                IrOp::Not(c) => classical[c as usize] ^= 1,
                IrOp::And(a, b) => classical[a as usize] &= classical[b as usize],
                IrOp::Or(a, b) => classical[a as usize] |= classical[b as usize],
                IrOp::Xor(a, b) => classical[a as usize] ^= classical[b as usize],
                IrOp::Add(a, b) => {
                    classical[a as usize] = classical[a as usize].wrapping_add(classical[b as usize])
                }
                IrOp::Sub(a, b) => {
                    classical[a as usize] = classical[a as usize].wrapping_sub(classical[b as usize])
                }
                IrOp::IfEq(c, v, boff, blen) => {
                    let (b0, bl) = (boff as usize, blen as usize);
                    if classical[c as usize] == v {
                        exec_fused_ops(prog, fused, b0, b0 + bl, backend, results, classical, histogram, rng, remap);
                    }
                    i = b0 + bl;
                    continue;
                }
                IrOp::IfNe(c, v, boff, blen) => {
                    let (b0, bl) = (boff as usize, blen as usize);
                    if classical[c as usize] != v {
                        exec_fused_ops(prog, fused, b0, b0 + bl, backend, results, classical, histogram, rng, remap);
                    }
                    i = b0 + bl;
                    continue;
                }
                IrOp::Shot(count, boff, blen) => {
                    let (b0, bl) = (boff as usize, blen as usize);
                    if bl == 0 {
                        let shot_pos = i;
                        for _ in 0..count.get() {
                            backend.reset();
                            exec_fused_ops(
                                prog,
                                fused,
                                0,
                                shot_pos,
                                backend,
                                results,
                                classical,
                                histogram,
                                rng,
                                remap,
                            );
                            let sample = backend.sample(rng);
                            histogram[sample] += 1;
                        }
                    } else {
                        for _ in 0..count.get() {
                            backend.reset();
                            exec_fused_ops(
                                prog,
                                fused,
                                b0,
                                b0 + bl,
                                backend,
                                results,
                                classical,
                                histogram,
                                rng,
                                remap,
                            );
                            let sample = backend.sample(rng);
                            histogram[sample] += 1;
                        }
                        i = b0 + bl;
                        continue;
                    }
                }
                IrOp::Print => {}
                // Unitary gates never appear as Original (fusion always
                // collapses them into U1/U2).
                IrOp::H(..) | IrOp::X(..) | IrOp::CNOT(..) | IrOp::RZ(..) | IrOp::RX(..)
                | IrOp::RY(..) | IrOp::Phase(..) | IrOp::S(..) | IrOp::T(..) | IrOp::SX(..)
                | IrOp::SWAP(..) | IrOp::ISWAP(..) | IrOp::CZ(..) | IrOp::CPHASE(..) => {
                    unreachable!("unitary op cannot appear as FusedOp::Original")
                }
                }
            },
        }
        i += 1;
    }
}

fn exec_ops<B: SimBackend>(
    prog: &Program,
    offset: usize,
    end: usize,
    backend: &mut B,
    results: &mut Results,
    classical: &mut [u8],
    histogram: &mut [u32],
    rng: &mut Lcg64,
    remap: &Remap,
) {
    let mut i = offset;
    while i < end {
        let raw = prog.ops[i];
        let op = if remap_active(remap) {
            translate_op(raw, remap)
        } else {
            raw
        };
        match op {
            IrOp::H(q) => { backend.apply_h(q); backend.noise_after_single(q, rng); }
            IrOp::X(q) => { backend.apply_x(q); backend.noise_after_single(q, rng); }
            IrOp::CNOT(c, t) => { backend.apply_cnot(c, t); backend.noise_after_pair(c, t, rng); }
            IrOp::Toff(c1, c2, t) => { backend.apply_toff(c1, c2, t); backend.noise_after_triple(c1, c2, t, rng); }
            IrOp::RZ(q, k) => { backend.apply_rz(q, prog.consts[k as usize]); backend.noise_after_single(q, rng); }
            IrOp::RX(q, k) => { backend.apply_rx(q, prog.consts[k as usize]); backend.noise_after_single(q, rng); }
            IrOp::RY(q, k) => { backend.apply_ry(q, prog.consts[k as usize]); backend.noise_after_single(q, rng); }
            IrOp::Phase(q, k) => { backend.apply_phase(q, prog.consts[k as usize]); backend.noise_after_single(q, rng); }
            IrOp::S(q) => { backend.apply_s(q); backend.noise_after_single(q, rng); }
            IrOp::T(q) => { backend.apply_t(q); backend.noise_after_single(q, rng); }
            IrOp::SX(q) => { backend.apply_sx(q); backend.noise_after_single(q, rng); }
            IrOp::SWAP(a, b) => { backend.apply_swap(a, b); backend.noise_after_pair(a, b, rng); }
            IrOp::ISWAP(a, b) => { backend.apply_iswap(a, b); backend.noise_after_pair(a, b, rng); }
            IrOp::CZ(a, b) => { backend.apply_cz(a, b); backend.noise_after_pair(a, b, rng); }
            IrOp::CPHASE(a, b, k) => { backend.apply_cphase(a, b, prog.consts[k as usize]); backend.noise_after_pair(a, b, rng); }
            IrOp::CSWAP(c, b, t) => { backend.apply_cswap(c, b, t); backend.noise_after_triple(c, b, t, rng); }
            IrOp::MCX(mask, t) => { backend.apply_mcx(mask, t); backend.noise_after_single(t, rng); }
            IrOp::Reset(q) => backend.reset_qubit(q, rng),
            IrOp::Set(c, v) => classical[c as usize] = v,
            IrOp::Not(c) => classical[c as usize] ^= 1,
            IrOp::And(a, b) => classical[a as usize] &= classical[b as usize],
            IrOp::Or(a, b) => classical[a as usize] |= classical[b as usize],
            IrOp::Xor(a, b) => classical[a as usize] ^= classical[b as usize],
            IrOp::Add(a, b) => {
                classical[a as usize] = classical[a as usize].wrapping_add(classical[b as usize])
            }
            IrOp::Sub(a, b) => {
                classical[a as usize] = classical[a as usize].wrapping_sub(classical[b as usize])
            }
            IrOp::Expect(pauli) => results.expectations.push(backend.expect_value(pauli)),
            IrOp::Estimate(off, len) => {
                let mut h = 0.0;
                for k in off as usize..(off as usize + len as usize) {
                    let (c, p) = prog.estimate_terms[k];
                    h += c * backend.expect_value(p as u64);
                }
                results.estimates.push(h);
            }
            IrOp::SaveState => backend.save_state(results),
            IrOp::SaveAmps => backend.save_amplitudes(results),
            IrOp::SaveProbs => backend.save_probabilities(results),
            IrOp::Call(sub_id, a0, a1, a2) => {
                let sub = prog.subs[sub_id as usize];
                let np = sub.nparams;
                // The op's args were already translated through the current
                // remap (translate_op); bind them as the callee's parameters.
                let mut nremap = [None; 3];
                let cargs = [a0, a1, a2];
                for k in 0..np {
                    nremap[k as usize] = Some(cargs[k as usize]);
                }
                let (off, len) = (sub.off as usize, sub.len as usize);
                exec_ops(
                    prog,
                    off,
                    off + len,
                    backend,
                    results,
                    classical,
                    histogram,
                    rng,
                    &nremap,
                );
            }
            IrOp::Measure(q, c) => {
                let result = backend.measure(q, rng);
                classical[c as usize] = result;
            }
            IrOp::MeasureX(q, c) => {
                let result = backend.measure_x(q, rng);
                classical[c as usize] = result;
            }
            IrOp::MeasureY(q, c) => {
                let result = backend.measure_y(q, rng);
                classical[c as usize] = result;
            }
IrOp::IfEq(c, val, body_off, body_len) => {
                let boff = body_off as usize;
                let blen = body_len as usize;
                if classical[c as usize] == val {
                    exec_ops(
                        prog,
                        boff,
                        boff + blen,
                        backend,
                        results,
                        classical,
                        histogram,
                        rng,
                        remap,
                    );
                }
                // Body ops are inline in the flat array, skip past them
                // to avoid executing them again in the sequential loop.
                i = boff + blen;
                continue;
            }
            IrOp::IfNe(c, val, body_off, body_len) => {
                let boff = body_off as usize;
                let blen = body_len as usize;
                if classical[c as usize] != val {
                    exec_ops(
                        prog,
                        boff,
                        boff + blen,
                        backend,
                        results,
                        classical,
                        histogram,
                        rng,
                        remap,
                    );
                }
                // Body ops are inline in the flat array, skip past them
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
                        exec_ops(
                            prog,
                            0,
                            shot_pos,
                            backend,
                            results,
                            classical,
                            histogram,
                            rng,
                            remap,
                        );
                        let sample = backend.sample(rng);
                        histogram[sample] += 1;
                    }
                } else {
                    // Shot with explicit body block
                    for _ in 0..count.get() {
                        backend.reset();
                        exec_ops(
                            prog,
                            boff,
                            boff + blen,
                            backend,
                            results,
                            classical,
                            histogram,
                            rng,
                            remap,
                        );
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
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
        // X(H|0>) = (|1> + |0>)/√2
        assert!(approx_eq(sim.state.re[0], INV_SQRT_2));
        assert!(approx_eq(sim.state.re[1], INV_SQRT_2));
    }

    #[test]
    fn test_cnot_bell_state() {
        let mut sim = CpuBackend::new(2);
        sim.apply_h(0);
        sim.apply_cnot(0, 1);
        // Should be (|00> + |11>)/√2
        assert!(approx_eq(sim.state.re[0], INV_SQRT_2));
        assert!(approx_eq(sim.state.re[3], INV_SQRT_2));
        assert!(approx_eq(sim.state.re[1], 0.0));
        assert!(approx_eq(sim.state.re[2], 0.0));
    }

    #[test]
    fn test_toffoli() {
        let mut sim = CpuBackend::new(3);
        // Set |110> (qubits 0 and 1 = 1, qubit 2 = 0)
        sim.apply_x(0);
        sim.apply_x(1);
        // |110> → index 3 (binary 011 on qubits 2,1,0 = 110 = 3)
        assert!(approx_eq(sim.state.re[3], 1.0));
        sim.apply_toff(0, 1, 2);
        // |110> → |111>
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

        run_program(&prog, &mut Results::default(), &mut classical, &mut histogram, &mut rng, NoiseModel::default());

        // With 2 qubit state, only |00> + |11>, so histogram[0] and histogram[3]
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

        run_program(&prog, &mut Results::default(), &mut classical, &mut histogram, &mut rng, NoiseModel::default());

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

        run_program_shots(&prog, 5000, &mut Results::default(), &mut classical, &mut histogram, &mut rng, NoiseModel::default());

        let total: u32 = histogram.iter().sum();
        assert_eq!(total, 5000);
        assert!((histogram[0] as f64 / 5000.0 - 0.5).abs() < 0.05);
        assert!((histogram[1] as f64 / 5000.0 - 0.5).abs() < 0.05);
    }

    // v0.2 gates

    const PI: f64 = 3.141592653589793;
    const HALF_PI: f64 = 1.5707963267948966;
    const PI_4: f64 = 0.7853981633974483;

    fn re(sim: &CpuBackend, i: usize) -> f64 {
        sim.state.re[i]
    }
    fn im(sim: &CpuBackend, i: usize) -> f64 {
        sim.state.im[i]
    }

    #[test]
    fn test_rz_pi() {
        let mut sim = CpuBackend::new(1);
        sim.apply_rz(0, PI);
        // RZ(π)|0> = e^{-iπ/2}|0> = -i|0>
        assert!(approx_eq(re(&sim, 0), 0.0), "re0 = {}", sim.state.re[0]);
        assert!(approx_eq(im(&sim, 0), -1.0), "im0 = {}", sim.state.im[0]);
        let mut sim2 = CpuBackend::new(1);
        sim2.apply_x(0);
        sim2.apply_rz(0, PI);
        // RZ(π)|1> = e^{iπ/2}|1> = i|1>
        assert!(approx_eq(re(&sim2, 1), 0.0));
        assert!(approx_eq(im(&sim2, 1), 1.0));
    }

    #[test]
    fn test_rx_pi() {
        let mut sim = CpuBackend::new(1);
        sim.apply_rx(0, PI);
        // RX(π)|0> = -i|1>
        assert!(approx_eq(re(&sim, 0), 0.0));
        assert!(approx_eq(im(&sim, 0), 0.0));
        assert!(approx_eq(re(&sim, 1), 0.0));
        assert!(approx_eq(im(&sim, 1), -1.0));
    }

    #[test]
    fn test_ry_pi() {
        let mut sim = CpuBackend::new(1);
        sim.apply_ry(0, PI);
        // RY(π)|0> = |1>
        assert!(approx_eq(re(&sim, 1), 1.0));
        assert!(approx_eq(im(&sim, 1), 0.0));
    }

    #[test]
    fn test_phase_pi_over_2() {
        let mut sim = CpuBackend::new(1);
        sim.apply_x(0);
        sim.apply_phase(0, HALF_PI);
        // |1> -> e^{iπ/2}|1> = i|1>
        assert!(approx_eq(re(&sim, 1), 0.0));
        assert!(approx_eq(im(&sim, 1), 1.0));
    }

    #[test]
    fn test_s_gate() {
        let mut sim = CpuBackend::new(1);
        sim.apply_x(0);
        sim.apply_s(0);
        assert!(approx_eq(re(&sim, 1), 0.0));
        assert!(approx_eq(im(&sim, 1), 1.0));
    }

    #[test]
    fn test_t_gate() {
        let mut sim = CpuBackend::new(1);
        sim.apply_x(0);
        sim.apply_t(0);
        // T|1> = e^{iπ/4}|1> = (√2/2)(1+i)
        assert!(approx_eq(re(&sim, 1), INV_SQRT_2));
        assert!(approx_eq(im(&sim, 1), INV_SQRT_2));
    }

    #[test]
    fn test_sx_twice_is_x() {
        let mut sim = CpuBackend::new(1);
        sim.apply_sx(0);
        sim.apply_sx(0);
        // SX2 = X: |0> -> |1>
        assert!(approx_eq(re(&sim, 0), 0.0));
        assert!(approx_eq(re(&sim, 1), 1.0));
        assert!(approx_eq(im(&sim, 0), 0.0));
    }

    #[test]
    fn test_swap_gate() {
        let mut sim = CpuBackend::new(2);
        sim.apply_x(0); // |01> (index 1, qubit 0 set)
        sim.apply_swap(0, 1);
        // |10> (index 2, qubit 1 set)
        assert!(approx_eq(re(&sim, 2), 1.0));
        assert!(approx_eq(re(&sim, 1), 0.0));
    }

    #[test]
    fn test_iswap_gate() {
        let mut sim = CpuBackend::new(2);
        sim.apply_x(0); // |01>
        sim.apply_iswap(0, 1);
        // iSWAP|01> = i|10>
        assert!(approx_eq(re(&sim, 2), 0.0));
        assert!(approx_eq(im(&sim, 2), 1.0));
    }

    #[test]
    fn test_cz_gate() {
        let mut sim = CpuBackend::new(2);
        sim.apply_x(0);
        sim.apply_x(1);
        sim.apply_cz(0, 1);
        // |11> -> -|11>
        assert!(approx_eq(re(&sim, 3), -1.0));
    }

    #[test]
    fn test_cphase_gate() {
        let mut sim = CpuBackend::new(2);
        sim.apply_x(0);
        sim.apply_x(1);
        sim.apply_cphase(0, 1, HALF_PI);
        // |11> -> i|11>
        assert!(approx_eq(re(&sim, 3), 0.0));
        assert!(approx_eq(im(&sim, 3), 1.0));
    }

    #[test]
    fn test_cswap_gate() {
        let mut sim = CpuBackend::new(3);
        sim.apply_x(2); // qubit 2 = control, qubit 0 set
        sim.apply_x(0);
        // |1 0 1> (ctl=1, b=0, c=1)
        sim.apply_cswap(2, 0, 1);
        // |1 1 0>
        assert!(approx_eq(re(&sim, 6), 1.0)); // index 6 = binary 110
        assert!(approx_eq(re(&sim, 5), 0.0)); // index 5 = binary 101
    }

    #[test]
    fn test_mcx_gate() {
        let mut sim = CpuBackend::new(3);
        sim.apply_x(0);
        sim.apply_x(1);
        // |011> index 3; MCX with mask 0b011 target 2
        sim.apply_mcx(0b011, 2);
        // |111> index 7
        assert!(approx_eq(re(&sim, 7), 1.0));
        assert!(approx_eq(re(&sim, 3), 0.0));
    }

    #[test]
    fn test_reset_qubit() {
        let mut sim = CpuBackend::new(1);
        sim.apply_x(0); // |1>
        let mut rng = Lcg64::new(1);
        sim.reset_qubit(0, &mut rng);
        // Reset always collapses to |0>.
        assert!(approx_eq(re(&sim, 0), 1.0));
        assert!(approx_eq(re(&sim, 1), 0.0));
    }

    #[test]
    fn test_measure_x_plus_state() {
        let mut sim = CpuBackend::new(1);
        sim.apply_h(0); // |+>
        let mut rng = Lcg64::new(1);
        let out = sim.measure_x(0, &mut rng);
        // |+> measured in X basis is always 0.
        assert_eq!(out, 0);
    }

    #[test]
    fn test_measure_y_plus_y_state() {
        let mut sim = CpuBackend::new(1);
        sim.apply_s(0); // |0> -> |0>
        sim.apply_h(0); // |+>
        sim.apply_s(0); // |+Y>
        let mut rng = Lcg64::new(1);
        let out = sim.measure_y(0, &mut rng);
        // |+Y> measured in Y basis is always 0.
        assert_eq!(out, 0);
    }

    #[test]
    fn test_classical_ops_program() {
        let mut classical = [0u8; 8];
        let mut histogram = vec![0u32; 2];
        let mut rng = Lcg64::new(42);

        let mut prog = Program::new();
        prog.num_qubits = 1;
        prog.num_classical = 8;
        prog.emit(IrOp::Set(0, 5));
        prog.emit(IrOp::Set(1, 3));
        prog.emit(IrOp::And(0, 1)); // 5 & 3 = 1
        prog.emit(IrOp::Not(0)); // 1 -> 0
        prog.emit(IrOp::Xor(0, 1)); // 0 ^ 3 = 3
        prog.emit(IrOp::Add(0, 1)); // 3 + 3 = 6
        prog.emit(IrOp::Sub(0, 1)); // 6 - 3 = 3
        prog.emit(IrOp::Set(4, 4));
        prog.emit(IrOp::Or(0, 4)); // 3 | 4 = 7

        // Pure classical: no SHOT means a single execution + one sample.
        run_program(&prog, &mut Results::default(), &mut classical, &mut histogram, &mut rng, NoiseModel::default());

        assert_eq!(classical[0], 7);
        assert_eq!(classical[1], 3);
    }

    // Gate-fusion agreement (M2)

    fn apply_to_state(sim: &mut CpuBackend, prog: &Program) {
        let mut classical = [0u8; 64];
        let mut histogram = vec![0u32; 1 << prog.num_qubits];
        let mut rng = Lcg64::new(1);
        exec_ops(prog, 0, prog.len, sim, &mut Results::default(), &mut classical, &mut histogram, &mut rng, &[None; 3]);
    }

    #[test]
    fn test_fused_matches_unfused_state() {
        let mut prog = Program::new();
        prog.num_qubits = 3;
        prog.num_classical = 3;
        // A gate-dense circuit with rotations, entanglement, and SWAP family.
        prog.emit(IrOp::H(0));
        prog.emit(IrOp::CNOT(0, 1));
        prog.emit(IrOp::RZ(1, 0));
        prog.emit(IrOp::CNOT(0, 1));
        prog.emit(IrOp::H(1));
        prog.emit(IrOp::RY(2, 1));
        prog.emit(IrOp::SWAP(1, 2));
        prog.emit(IrOp::X(0));
        prog.emit(IrOp::CZ(0, 2));
        prog.emit(IrOp::T(2));
        prog.emit(IrOp::CNOT(1, 2));
        prog.emit(IrOp::SX(0));
        // const pool for RZ/RY
        prog.num_consts = 2;
        prog.consts[0] = 0.5;
        prog.consts[1] = 1.3;

        let mut a = CpuBackend::new(prog.num_qubits);
        apply_to_state(&mut a, &prog);

        let fused = crate::fusion::fuse_program(&prog);
        let mut b = CpuBackend::new(prog.num_qubits);
        let mut classical = [0u8; 64];
        let mut histogram = vec![0u32; 1 << prog.num_qubits];
        let mut rng = Lcg64::new(1);
        exec_fused_ops(&prog, &fused, 0, fused.len(), &mut b, &mut Results::default(), &mut classical, &mut histogram, &mut rng, &[None; 3]);

        // Amplitudes must agree to high fidelity despite the reordered math.
        let n = a.state.re.len();
        let mut worst = 0.0f64;
        for i in 0..n {
            let de = (a.state.re[i] - b.state.re[i]).abs()
                + (a.state.im[i] - b.state.im[i]).abs();
            if de > worst {
                worst = de;
            }
        }
        assert!(worst < 1e-9, "worst amplitude diff {worst:.3e}");
    }

    #[test]
    fn test_fused_shot_histogram_agrees() {
        let mut prog = Program::new();
        prog.num_qubits = 2;
        prog.num_classical = 2;
        prog.emit(IrOp::H(0));
        prog.emit(IrOp::CNOT(0, 1));
        prog.emit(IrOp::RZ(1, 0));
        prog.emit(IrOp::H(1));
        prog.emit(IrOp::Shot(core::num::NonZeroU32::new(5000).unwrap(), 0, 0));
        prog.has_explicit_shot = true;
        prog.num_consts = 1;
        prog.consts[0] = 0.7;

        let fused = crate::fusion::fuse_program(&prog);
        assert!(fused.len() <= 2); // one U2 block + the Shot

        let mut h1 = vec![0u32; 4];
        let mut h2 = vec![0u32; 4];
        let mut c1 = [0u8; 64];
        let mut c2 = [0u8; 64];
        let mut r1 = Lcg64::new(9);
        let mut r2 = Lcg64::new(9);

        run_program_shots(&prog, 5000, &mut Results::default(), &mut c1, &mut h1, &mut r1, NoiseModel::default());
        // Fused shot path: run the fused stream in a bare-shot loop.
        let mut backend = CpuBackend::new(prog.num_qubits);
        for _ in 0..5000 {
            backend.reset();
            exec_fused_ops(&prog, &fused, 0, fused.len(), &mut backend, &mut Results::default(), &mut c2, &mut h2, &mut r2, &[None; 3]);
            let s = backend.sample(&mut r2);
            h2[s] += 1;
        }

        // Fused and unfused histograms must agree to sampling tolerance.
        let mut worst = 0i64;
        for i in 0..4 {
            let d = (h1[i] as i64 - h2[i] as i64).abs();
            if d > worst {
                worst = d;
            }
        }
        assert!(worst <= 15, "worst bin diff {worst} (of 5000)");
    }

    // EXPECT / SAVE_* (M5)

    #[test]
    fn test_expect_basis_states() {
        // |0>: Z0=1, X0=0, Y0=0. |1>: Z0=-1.
        let mut sim = CpuBackend::new(1);
        assert!((sim.expect_value(3) - 1.0).abs() < 1e-12); // Z0
        assert!(sim.expect_value(1).abs() < 1e-12); // X0
        assert!(sim.expect_value(2).abs() < 1e-12); // Y0
        sim.apply_x(0);
        assert!((sim.expect_value(3) + 1.0).abs() < 1e-12); // Z0 = -1
    }

    #[test]
    fn test_expect_bell_state() {
        // Bell (|00>+|11>)/sqrt2: <Z0>=0, <Z1>=0, <Z0Z1>=1, <X0X1>=1, <Y0Y1>=-1.
        let mut sim = CpuBackend::new(2);
        sim.apply_h(0);
        sim.apply_cnot(0, 1);
        assert!(sim.expect_value(3).abs() < 1e-12); // Z0
        assert!(sim.expect_value(12).abs() < 1e-12); // Z1
        assert!((sim.expect_value(15) - 1.0).abs() < 1e-12); // Z0Z1
        assert!((sim.expect_value(5) - 1.0).abs() < 1e-12); // X0X1
        assert!((sim.expect_value(10) + 1.0).abs() < 1e-12); // Y0Y1 = -1
    }

    #[test]
    fn test_expect_plus_state() {
        // |+>: <X0>=1, <Z0>=0, <Y0>=0.
        let mut sim = CpuBackend::new(1);
        sim.apply_h(0);
        assert!((sim.expect_value(1) - 1.0).abs() < 1e-12); // X0
        assert!(sim.expect_value(3).abs() < 1e-12); // Z0
        assert!(sim.expect_value(2).abs() < 1e-12); // Y0
    }

    #[test]
    fn test_save_ops() {
        let mut classical = [0u8; 64];
        let mut histogram = vec![0u32; 4];
        let mut rng = Lcg64::new(1);
        let mut results = Results::default();

        let mut prog = Program::new();
        prog.num_qubits = 2;
        prog.num_classical = 2;
        prog.emit(IrOp::H(0));
        prog.emit(IrOp::SaveState);
        prog.emit(IrOp::CNOT(0, 1));
        prog.emit(IrOp::SaveProbs);
        prog.emit(IrOp::SaveAmps);

        run_program(&prog, &mut results, &mut classical, &mut histogram, &mut rng, NoiseModel::default());

        // State saved after H 0: |00>+|10> / sqrt2.
        assert_eq!(results.saved_states.len(), 1);
        let st = &results.saved_states[0];
        assert!((st[0] - 1.0 / sqrt(2.0)).abs() < 1e-12); // |00> re
        assert!((st[2] - 1.0 / sqrt(2.0)).abs() < 1e-12); // |10> re
        // Probabilities saved after Bell: 0.5 at 0 and 3.
        assert_eq!(results.saved_probs.len(), 1);
        let pr = &results.saved_probs[0];
        assert!((pr[0] - 0.5).abs() < 1e-12);
        assert!((pr[3] - 0.5).abs() < 1e-12);
        // Amplitudes saved after Bell.
        assert_eq!(results.saved_amplitudes.len(), 1);
        assert!((results.saved_amplitudes[0][0].1 - 1.0 / sqrt(2.0)).abs() < 1e-12);
    }


    #[test]
    fn test_subroutine_two_calls() {
        // Parse sub5: bell 1 2 then bell 0 1; the second must equal a fresh
        // bell 0 1 on the post-first-call state.
        let src = b"QUBITS 3\nGATE bell 2\nH 0\nCNOT 0 1\nENDGATE\nbell 1 2\nSAVE_STATEVECTOR\nbell 0 1\nSAVE_STATEVECTOR\nPRINT\n";
        let mut lex = crate::lexer::LexResult::new();
        crate::lexer::tokenise(src, &mut lex);
        let mut prog = Program::new();
        let mut diag = crate::diag::Diag::new();
        assert!(crate::parser::parse(&lex, &mut prog, &mut diag), "{}", diag.message());
        let mut results = Results::default();
        let mut classical = [0u8; 64];
        let mut hist = vec![0u32; 8];
        let mut rng = Lcg64::new(1);
        run_program(&prog, &mut results, &mut classical, &mut hist, &mut rng, NoiseModel::default());
        assert_eq!(results.saved_states.len(), 2);
        // state 0 = (|000>+|110>)/sqrt2
        let s0 = &results.saved_states[0];
        assert!((s0[0] - 1.0 / sqrt(2.0)).abs() < 1e-9, "s0[0]={}", s0[0]);
        assert!((s0[12] - 1.0 / sqrt(2.0)).abs() < 1e-9, "s0[6]={}", s0[12]);
        // state 1 = (|000>+|011>+|101>+|110>)/2 = indices 0,3,5,6 each +0.5
        let s1 = &results.saved_states[1];
        assert!(
            (s1[0]-0.5).abs() < 1e-9 && (s1[6]-0.5).abs() < 1e-9
            && (s1[10]-0.5).abs() < 1e-9 && (s1[12]-0.5).abs() < 1e-9,
            "state1 = [0]={} [3]={} [5]={} [6]={}",
            s1[0], s1[6], s1[10], s1[12]
        );
    }


    #[test]
    fn test_fused_fallback_with_calls() {
        // Programs with CALLs must not go through the fused stream (sub
        // bodies are not in the fused array); run_fused_program must fall
        // back to run_program and still sample the explicit SHOT.
        let src = b"QUBITS 2\nGATE h2 2\nH 0\nCNOT 0 1\nENDGATE\nh2 0 1\nh2 0 1\nSHOT 1000\nPRINT\n";
        let mut lex = crate::lexer::LexResult::new();
        crate::lexer::tokenise(src, &mut lex);
        let mut prog = Program::new();
        let mut diag = crate::diag::Diag::new();
        assert!(crate::parser::parse(&lex, &mut prog, &mut diag), "{}", diag.message());
        assert!(program_has_calls(&prog), "program_has_calls must be true");
        let fused = crate::fusion::fuse_program(&prog);
        let mut results = Results::default();
        let mut classical = [0u8; 64];
        let mut hist = vec![0u32; 4];
        let mut rng = Lcg64::new(1);
        run_fused_program(&prog, &fused, &mut results, &mut classical, &mut hist, &mut rng, NoiseModel::default());
        let total: u32 = hist.iter().sum();
        assert_eq!(total, 1000, "fused-with-calls fallback must sample the explicit SHOT (got {total})");
    }


    #[test]
    fn test_fused_cli_scenario() {
        // Exact CLI scenario: fused program with explicit SHOT and calls.
        let src = b"QUBITS 3\nGATE bell 2\nH 0\nCNOT 0 1\nENDGATE\nbell 1 2\nbell 0 1\nEXPECT Z0Z1Z2\nSHOT 5000\nPRINT\n";
        let mut lex = crate::lexer::LexResult::new();
        crate::lexer::tokenise(src, &mut lex);
        let mut prog = Program::new();
        let mut diag = crate::diag::Diag::new();
        assert!(crate::parser::parse(&lex, &mut prog, &mut diag), "{}", diag.message());
        let fused = crate::fusion::fuse_program(&prog);
        let mut results = Results::default();
        let mut classical = [0u8; 64];
        let mut hist = vec![0u32; 8];
        let mut rng = Lcg64::new(3);
        run_fused_program(&prog, &fused, &mut results, &mut classical, &mut hist, &mut rng, NoiseModel::default());
        let total: u32 = hist.iter().sum();
        assert_eq!(total, 5000, "CLI fused scenario must sample the explicit SHOT (got {total})");
    }


    #[test]
    fn test_noise_readout_distribution() {
        // Measuring |0> with readout rate 0.5 must report ~50/50, and the
        // run must be deterministic for a fixed seed.
        let prog = parse_test(b"QUBITS 1\nMEASURE 0\nSHOT 20000\nPRINT\n");
        let noise = NoiseModel { readout: 0.5, ..NoiseModel::default() };
        let mut results = Results::default();
        let mut classical = [0u8; 64];
        let mut hist = vec![0u32; 2];
        let mut rng = Lcg64::new(1);
        run_program(&prog, &mut results, &mut classical, &mut hist, &mut rng, noise);
        let ones = hist[1];
        assert!((0.45 * 20000.0) < ones as f64 && (ones as f64) < (0.55 * 20000.0),
                "readout flip fraction {ones} not ~50%");
    }

    #[test]
    fn test_noise_deterministic() {
        let prog = parse_test(b"QUBITS 2\nH 0\nCNOT 0 1\nRY 1 1.3\nSHOT 8000\nPRINT\n");
        let noise = NoiseModel { depolarizing: 0.2, amp_damping: 0.1, phase_damping: 0.1, ..NoiseModel::default() };
        let run = |seed: u64| {
            let mut results = Results::default();
            let mut classical = [0u8; 64];
            let mut hist = vec![0u32; 4];
            let mut rng = Lcg64::new(seed);
            run_program(&prog, &mut results, &mut classical, &mut hist, &mut rng, noise);
            hist
        };
        assert_eq!(run(7), run(7), "noisy runs must be bit-reproducible per seed");
        let mut r1 = run(1);
        let mut r2 = run(2);
        r1.sort(); r2.sort();
        assert_ne!(r1, r2, "different seeds should give different trajectories");
    }

    #[test]
    fn test_noise_preserves_norm() {
        let prog = parse_test(b"QUBITS 3\nH 0\nCNOT 0 1\nRY 1 2.2\nCNOT 1 2\nH 2\nSAVE_STATEVECTOR\nPRINT\n");
        let noise = NoiseModel { depolarizing: 0.3, amp_damping: 0.2, phase_damping: 0.15, ..NoiseModel::default() };
        let mut results = Results::default();
        let mut classical = [0u8; 64];
        let mut hist = vec![0u32; 8];
        let mut rng = Lcg64::new(5);
        run_program(&prog, &mut results, &mut classical, &mut hist, &mut rng, noise);
        let sv = &results.saved_states[0];
        let mut norm = 0.0;
        for k in 0..8 {
            norm += sv[2 * k] * sv[2 * k] + sv[2 * k + 1] * sv[2 * k + 1];
        }
        assert!((norm - 1.0).abs() < 1e-9, "noisy state not normalized: {norm}");
    }

    fn parse_test(src: &[u8]) -> Program {
        let mut lex = crate::lexer::LexResult::new();
        crate::lexer::tokenise(src, &mut lex);
        let mut prog = Program::new();
        let mut diag = crate::diag::Diag::new();
        assert!(crate::parser::parse(&lex, &mut prog, &mut diag), "{}", diag.message());
        prog
    }


    #[test]
    fn test_estimate_weighted_sum() {
        // Bell state: Z0Z1 = X0X1 = 1, Y0Y1 = -1.
        let prog = parse_test(b"QUBITS 2\nH 0\nCNOT 0 1\nESTIMATE 0.5 Z0Z1 0.3 X0X1 -0.2 Y0Y1\nPRINT\n");
        let mut results = Results::default();
        let mut classical = [0u8; 64];
        let mut hist = vec![0u32; 4];
        let mut rng = Lcg64::new(1);
        run_program(&prog, &mut results, &mut classical, &mut hist, &mut rng, NoiseModel::default());
        assert_eq!(results.estimates.len(), 1);
        let h = results.estimates[0];
        assert!((h - 1.0).abs() < 1e-9, "estimate = {h}, expected 1.0");
    }

    #[test]
    fn test_mitigate_readout() {
        // c_true = M^-1 c_obs for a single qubit: perfect |0> measurements
        // contaminated by readout rate p; inversion recovers the ideal.
        let p = 0.1;
        let mut hist = vec![9000u32, 1000u32]; // observed: 9000|0>, 1000|1>
        mitigate_readout(&mut hist, 1, p);
        // Ideal would be 10000 |0>. The inversion gives ~10000 - small term.
        assert!(hist[1] < 300, "mitigated |1> count {} should be near 0", hist[1]);
        assert!(hist[0] > 9700, "mitigated |0> count {} should be near 10000", hist[0]);

        // Two qubits, independent confusion: prepare |01> (index 2).
        let mut h2 = vec![0u32, 0, 9000, 0, 1000, 0, 0, 0]; // index 2 mostly, index 4 leak
        mitigate_readout(&mut h2, 3, p);
        assert!(h2[2] > 9700, "mitigated |010> count {} should dominate", h2[2]);
    }


    #[test]
    fn test_stabilizer_matches_statevector() {
        // Randomized Clifford circuits WITHOUT measurement: EXPECT values are
        // deterministic (+/-1 or 0) and must match the statevector exactly;
        // shot histograms must be statistically identical.
        let mut gen: u64 = 0x9E3779B97F4A7C15;
        let mut rand = |m: u64| {
            gen ^= gen << 13;
            gen ^= gen >> 7;
            gen ^= gen << 17;
            gen % m
        };
        for trial in 0..60 {
            let n = 2 + (rand(4) as u8); // 2..5 qubits
            let mut src = format!("QUBITS {n}\n");
            let gates = 6 + rand(14);
            for _ in 0..gates {
                match rand(6) {
                    0 => src.push_str(&format!("H {}\n", rand(n as u64))),
                    1 => src.push_str(&format!("S {}\n", rand(n as u64))),
                    2 => src.push_str(&format!("X {}\n", rand(n as u64))),
                    3 => {
                        let c = rand(n as u64);
                        let mut t = rand(n as u64);
                        while t == c {
                            t = rand(n as u64);
                        }
                        src.push_str(&format!("CNOT {c} {t}\n"));
                    }
                    4 => {
                        let a = rand(n as u64);
                        let mut b = rand(n as u64);
                        while b == a {
                            b = rand(n as u64);
                        }
                        src.push_str(&format!("SWAP {a} {b}\n"));
                    }
                    _ => {
                        let a = rand(n as u64);
                        let mut b = rand(n as u64);
                        while b == a {
                            b = rand(n as u64);
                        }
                        src.push_str(&format!("CZ {a} {b}\n"));
                    }
                }
            }
            // random single-qubit Pauli observables
            for _ in 0..2 {
                let q = rand(n as u64);
                let p = ["Z", "X", "Y"][rand(3) as usize];
                src.push_str(&format!("EXPECT {p}{q}\n"));
            }
            src.push_str("SHOT 4000\nPRINT\n");
            let prog = parse_test(src.as_bytes());
            assert!(is_clifford_program(&prog), "trial {trial} should be Clifford");
            let mut sv_hist = vec![0u32; 1 << n];
            let mut st_hist = vec![0u32; 1 << n];
            let mut sv_res = Results::default();
            let mut st_res = Results::default();
            let mut sv_class = [0u8; 64];
            let mut st_class = [0u8; 64];
            let mut r1 = Lcg64::new(trial);
            let mut r2 = Lcg64::new(trial);
            run_program(&prog, &mut sv_res, &mut sv_class, &mut sv_hist, &mut r1, NoiseModel::default());
            run_stabilizer_program(&prog, &mut st_res, &mut st_class, &mut st_hist, &mut r2);
            assert_eq!(sv_res.expectations.len(), st_res.expectations.len(),
                       "trial {trial}: EXPECT length mismatch");
            for (a, b) in sv_res.expectations.iter().zip(st_res.expectations.iter()) {
                assert!((a - b).abs() < 1e-9, "trial {trial}: EXPECT sv={a} st={b}\nprogram:\n{src}");
            }
            let s1: u64 = sv_hist.iter().map(|&x| x as u64).sum();
            let s2: u64 = st_hist.iter().map(|&x| x as u64).sum();
            assert_eq!(s1, s2, "trial {trial}: shot totals differ");
            let tvd: f64 = sv_hist.iter().zip(st_hist.iter()).map(|(a, b)| {
                ((*a as f64 / s1 as f64) - (*b as f64 / s2 as f64)).abs()
            }).sum::<f64>() / 2.0;
            assert!(tvd < 0.05, "trial {trial}: histogram TVD {tvd}\nprogram:\n{src}");
        }
    }

    #[test]
    fn test_stabilizer_deterministic() {
        let src = b"QUBITS 3\nH 0\nCNOT 0 1\nCNOT 1 2\nS 0\nX 2\nMEASURE 0\nSHOT 8000\nPRINT\n";
        let prog = parse_test(src);
        let mut r1 = vec![0u32; 8];
        let mut r2 = vec![0u32; 8];
        let mut a = Results::default();
        let mut b = Results::default();
        let mut ca = [0u8; 64];
        let mut cb = [0u8; 64];
        run_stabilizer_program(&prog, &mut a, &mut ca, &mut r1, &mut Lcg64::new(9));
        run_stabilizer_program(&prog, &mut b, &mut cb, &mut r2, &mut Lcg64::new(9));
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_clifford_detection() {
        let ok = parse_test(b"QUBITS 2\nH 0\nCNOT 0 1\nS 1\nSWAP 0 1\nEXPECT Z0Z1\nPRINT\n");
        assert!(is_clifford_program(&ok));
        let bad = parse_test(b"QUBITS 2\nH 0\nRY 0 1.3\nPRINT\n");
        assert!(!is_clifford_program(&bad));
        let toff = parse_test(b"QUBITS 3\nH 0\nTOFF 0 1 2\nPRINT\n");
        assert!(!is_clifford_program(&toff));
        let save = parse_test(b"QUBITS 2\nH 0\nSAVE_STATEVECTOR\nPRINT\n");
        assert!(!is_clifford_program(&save));
    }


    #[test]
    fn test_stabilizer_measure_basics() {
        // Measurement histograms must be statistically identical to the
        // statevector reference (the RNG is consumed in different orders, so
        // individual outcomes differ but the distribution is the same).
        let cases: &[&[u8]] = &[
            b"QUBITS 2\nH 0\nCNOT 0 1\nMEASURE 0\nSHOT 6000\nPRINT\n",
            b"QUBITS 2\nH 0\nCNOT 0 1\nX 1\nMEASURE 0\nSHOT 6000\nPRINT\n",
            b"QUBITS 2\nH 0\nCNOT 0 1\nS 1\nMEASURE 0\nSHOT 6000\nPRINT\n",
            b"QUBITS 3\nH 0\nCNOT 0 1\nCNOT 1 2\nX 2\nS 0\nMEASURE 0\nSHOT 6000\nPRINT\n",
            b"QUBITS 3\nH 2\nMEASURE 1\nSHOT 6000\nPRINT\n",
        ];
        for (k, src) in cases.iter().enumerate() {
            let prog = parse_test(src);
            let mut sv_h = vec![0u32; 1 << prog.num_qubits];
            let mut st_h = vec![0u32; 1 << prog.num_qubits];
            let mut sv = Results::default();
            let mut st = Results::default();
            let mut sc = [0u8; 64];
            let mut tc = [0u8; 64];
            run_program(&prog, &mut sv, &mut sc, &mut sv_h, &mut Lcg64::new(1), NoiseModel::default());
            run_stabilizer_program(&prog, &mut st, &mut tc, &mut st_h, &mut Lcg64::new(1));
            let s1: u64 = sv_h.iter().map(|&x| x as u64).sum();
            let s2: u64 = st_h.iter().map(|&x| x as u64).sum();
            assert_eq!(s1, s2, "case {k}: shot totals differ");
            let tvd: f64 = sv_h.iter().zip(st_h.iter()).map(|(a, b)| {
                ((*a as f64 / s1 as f64) - (*b as f64 / s2 as f64)).abs()
            }).sum::<f64>() / 2.0;
            assert!(tvd < 0.08, "case {k}: histogram TVD {tvd}");
        }
    }


    #[test]
    fn test_mps_matches_statevector() {
        // Low-entanglement circuits: the MPS must reproduce the statevector's
        // EXPECT values (exact) and shot histograms (statistically identical).
        let src = b"QUBITS 4\nH 0\nCNOT 0 1\nRY 1 1.3\nCNOT 1 2\nH 3\nRX 3 2.1\nCNOT 2 3\nEXPECT Z0Z1\nEXPECT X0X1Z2\nSHOT 8000\nPRINT\n";
        let prog = parse_test(src);
        let mut sv_h = vec![0u32; 16];
        let mut mp_h = vec![0u32; 16];
        let mut sv = Results::default();
        let mut mp = Results::default();
        let mut sc = [0u8; 64];
        let mut mc = [0u8; 64];
        run_program(&prog, &mut sv, &mut sc, &mut sv_h, &mut Lcg64::new(3), NoiseModel::default());
        run_mps_program(&prog, 32, &mut mp, &mut mc, &mut mp_h, &mut Lcg64::new(3));
        assert_eq!(sv.expectations.len(), mp.expectations.len());
        for (a, b) in sv.expectations.iter().zip(mp.expectations.iter()) {
            assert!((a - b).abs() < 1e-9, "EXPECT sv={a} mps={b}");
        }
        let s1: u64 = sv_h.iter().map(|&x| x as u64).sum();
        let s2: u64 = mp_h.iter().map(|&x| x as u64).sum();
        assert_eq!(s1, s2);
        let tvd: f64 = sv_h.iter().zip(mp_h.iter()).map(|(a, b)| {
            ((*a as f64 / s1 as f64) - (*b as f64 / s2 as f64)).abs()
        }).sum::<f64>() / 2.0;
        assert!(tvd < 0.08, "histogram TVD {tvd}");
    }

    #[test]
    fn test_mps_measurement() {
        // GHZ with measurement: MPS must sample the correlated outcomes.
        let src = b"QUBITS 3\nH 0\nCNOT 0 1\nCNOT 1 2\nMEASURE 0\nSHOT 4000\nPRINT\n";
        let prog = parse_test(src);
        let mut sv_h = vec![0u32; 8];
        let mut mp_h = vec![0u32; 8];
        let mut sv = Results::default();
        let mut mp = Results::default();
        let mut sc = [0u8; 64];
        let mut mc = [0u8; 64];
        run_program(&prog, &mut sv, &mut sc, &mut sv_h, &mut Lcg64::new(1), NoiseModel::default());
        run_mps_program(&prog, 16, &mut mp, &mut mc, &mut mp_h, &mut Lcg64::new(1));
        let s1: u64 = sv_h.iter().map(|&x| x as u64).sum();
        let s2: u64 = mp_h.iter().map(|&x| x as u64).sum();
        let tvd: f64 = sv_h.iter().zip(mp_h.iter()).map(|(a, b)| {
            ((*a as f64 / s1 as f64) - (*b as f64 / s2 as f64)).abs()
        }).sum::<f64>() / 2.0;
        assert!(tvd < 0.08, "histogram TVD {tvd}");
    }

    #[test]
    fn test_mps_deterministic() {
        let src = b"QUBITS 3\nH 0\nCNOT 0 1\nRY 1 1.1\nCNOT 1 2\nSHOT 2000\nPRINT\n";
        let prog = parse_test(src);
        let mut h1 = vec![0u32; 8];
        let mut h2 = vec![0u32; 8];
        let mut a = Results::default();
        let mut b = Results::default();
        let mut ca = [0u8; 64];
        let mut cb = [0u8; 64];
        run_mps_program(&prog, 16, &mut a, &mut ca, &mut h1, &mut Lcg64::new(9));
        run_mps_program(&prog, 16, &mut b, &mut cb, &mut h2, &mut Lcg64::new(9));
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_error_model() {
        // Monotone, sqrt-scaling, sensible magnitude.
        let e1 = crate::sim::estimate_f32_amp_error(10);
        let e2 = crate::sim::estimate_f32_amp_error(1000);
        let e3 = crate::sim::estimate_f32_amp_error(100000);
        assert!(e1 < e2 && e2 < e3);
        assert!(e1 > 1e-8 && e1 < 1e-5);
        assert!((e3 / e1 - 100.0).abs() < 1.0); // sqrt(10000) = 100
    }

    #[test]
    fn test_expect_in_program() {
        let mut classical = [0u8; 64];
        let mut histogram = vec![0u32; 4];
        let mut rng = Lcg64::new(1);
        let mut results = Results::default();

        let mut prog = Program::new();
        prog.num_qubits = 2;
        prog.num_classical = 2;
        prog.emit(IrOp::H(0));
        prog.emit(IrOp::CNOT(0, 1));
        prog.emit(IrOp::Expect(15)); // Z0Z1
        prog.emit(IrOp::Expect(5)); // X0X1
        prog.emit(IrOp::Expect(10)); // Y0Y1

        run_program(&prog, &mut results, &mut classical, &mut histogram, &mut rng, NoiseModel::default());
        assert_eq!(results.expectations.len(), 3);
        assert!((results.expectations[0] - 1.0).abs() < 1e-12);
        assert!((results.expectations[1] - 1.0).abs() < 1e-12);
        assert!((results.expectations[2] + 1.0).abs() < 1e-12);
    }
}
