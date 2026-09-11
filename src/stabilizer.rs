// Copyright (c) 2026 Faris Alfarhan
// SPDX-License-Identifier: GPL-3.0-only

//! Stabilizer (CH-formalism) simulation engine.
//!
//! Exact, polynomial simulation of Clifford circuits (Gottesman-Knill). The
//! state is a 2n-row tableau: rows `0..n` are the destabilizers, rows
//! `n..2n` the stabilizers; each row holds an X-part and a Z-part (u64 bitsets,
//! `n <= 28`) and a phase `r in {0,1,2,3}` meaning the Pauli `i^r X^x Z^z`.
//! All randomness comes from the host seeded RNG in a fixed order, so the
//! engine is bit-reproducible and exact for the Clifford subset:
//! `H S X Z CNOT CZ SWAP` plus `MEASURE`/`MEASUREX`/`MEASUREY`, `RESET`,
//! `EXPECT`/`ESTIMATE`, and all classical/control-flow ops.
//!
//! Gate rules are derived by conjugation (see `apply_h` etc.); the phase
//! increments are modular (mod 4), avoiding the sign conventions that trip up
//! naive implementations. Correctness is validated against the statevector
//! backend on randomized Clifford circuits.

use alloc::vec::Vec;

/// Max qubits the tableau supports (packed into u64 bitsets).
pub const STAB_MAX_QUBITS: u8 = 28;

#[derive(Clone, Debug)]
pub struct StabilizerBackend {
    n: usize,
    /// 2n rows; `x[i]`/`z[i]` are `n`-bit masks, `r[i]` in 0..=3.
    x: [u64; 2 * STAB_MAX_QUBITS as usize],
    z: [u64; 2 * STAB_MAX_QUBITS as usize],
    r: [u8; 2 * STAB_MAX_QUBITS as usize],
}

impl StabilizerBackend {
    pub fn new(num_qubits: u8) -> Self {
        assert!(num_qubits <= STAB_MAX_QUBITS);
        let n = num_qubits as usize;
        let mut sb = Self {
            n,
            x: [0; 2 * STAB_MAX_QUBITS as usize],
            z: [0; 2 * STAB_MAX_QUBITS as usize],
            r: [0; 2 * STAB_MAX_QUBITS as usize],
        };
        // |0...0>: destabilizer D_k = X_k, stabilizer S_k = Z_k.
        for k in 0..n {
            sb.x[k] = 1u64 << k;
            sb.z[n + k] = 1u64 << k;
        }
        sb
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.n as u8);
    }

    #[inline]
    fn phase_add(&mut self, i: usize, d: u32) {
        self.r[i] = ((self.r[i] as u32 + d) & 3) as u8;
    }

    /// Number of basis states (2^n) for `sample`.
    pub fn dim(&self) -> usize {
        1usize << self.n
    }

    // Gates (conjugation rules)

    pub fn apply_h(&mut self, q: u8) {
        let bit = 1u64 << q;
        for i in 0..2 * self.n {
            let xq = (self.x[i] >> q) & 1;
            let zq = (self.z[i] >> q) & 1;
            // swap x_q <-> z_q
            self.x[i] = (self.x[i] & !bit) | (zq << q);
            self.z[i] = (self.z[i] & !bit) | (xq << q);
            if xq & zq == 1 {
                self.phase_add(i, 2);
            }
        }
    }

    pub fn apply_s(&mut self, q: u8) {
        for i in 0..2 * self.n {
            let xq = (self.x[i] >> q) & 1;
            self.phase_add(i, xq as u32);
            self.z[i] ^= xq << q;
        }
    }

    pub fn apply_x(&mut self, q: u8) {
        for i in 0..2 * self.n {
            let zq = (self.z[i] >> q) & 1;
            self.phase_add(i, 2 * zq as u32);
        }
    }

    pub fn apply_z(&mut self, q: u8) {
        for i in 0..2 * self.n {
            let xq = (self.x[i] >> q) & 1;
            self.phase_add(i, 2 * xq as u32);
        }
    }

    pub fn apply_cnot(&mut self, c: u8, t: u8) {
        for i in 0..2 * self.n {
            let xc = (self.x[i] >> c) & 1;
            let zt = (self.z[i] >> t) & 1;
            // X_t ^= X_c ; Z_c ^= Z_t ; no phase (derived via conjugation).
            self.x[i] ^= xc << t;
            self.z[i] ^= zt << c;
        }
    }

    pub fn apply_cz(&mut self, a: u8, b: u8) {
        self.apply_h(b);
        self.apply_cnot(a, b);
        self.apply_h(b);
    }

    pub fn apply_swap(&mut self, a: u8, b: u8) {
        self.apply_cnot(a, b);
        self.apply_cnot(b, a);
        self.apply_cnot(a, b);
    }

    /// Row-sum a specific row `(ax, az, ar)` into row `h`.
    fn rowsum_into(&mut self, h: usize, ax: u64, az: u64, ar: u8) {
        let ph = ((self.r[h] as u32 + ar as u32
            + 2 * ((ax & self.z[h]).count_ones() & 1))
            & 3) as u8;
        self.x[h] ^= ax;
        self.z[h] ^= az;
        self.r[h] = ph;
    }

    /// Eigenvalue of a Pauli (given by x/z masks) on the stabilizer state, if
    /// the Pauli lies in the stabilizer group (up to sign).
    ///
    /// Uses the symplectic pairing with the destabilizers: for a Pauli P that
    /// commutes with every stabilizer, the coefficient of stabilizer S_k in
    /// P's decomposition is `a_k = [P, D_k]` (the symplectic inner product
    /// with the paired destabilizer D_k). The phase is accumulated by
    /// multiplying the selected stabilizers in order. This avoids the
    /// pivot-oscillation failure of a naive Gaussian elimination.
    fn group_value(&self, tx: u64, tz: u64, tphase: u32) -> Option<f64> {
        let n = self.n;
        let mut ax = 0u64;
        let mut az = 0u64;
        let mut phase = 0u32;
        for k in 0..n {
            let dx = self.x[k];
            let dz = self.z[k];
            let coef = ((tx & dz).count_ones() ^ (dx & tz).count_ones()) & 1;
            if coef == 1 {
                let (sx, sz, sr) = (self.x[n + k], self.z[n + k], self.r[n + k] as u32);
                phase = (phase + sr + 2 * ((sx & az).count_ones() & 1)) & 3;
                ax ^= sx;
                az ^= sz;
            }
        }
        if ax != tx || az != tz {
            return None; // P is not in the stabilizer group
        }
        // The stabilizer combination equals i^(phase-tphase) P; the eigenvalue
        // is +1 if the phases match, -1 if they differ by 2 (P is in the
        // group up to sign), and the Pauli has eigenvalue +-i otherwise (0).
        let d = (phase.wrapping_sub(tphase)) & 3;
        match d {
            0 => Some(1.0),
            2 => Some(-1.0),
            _ => None,
        }
    }

    /// Probability that a computational-basis measurement of qubit `q`
    /// returns 1 (0, 1/2, or 1 for a stabilizer state).
    pub fn probability(&self, q: u8) -> f64 {
        // Outcome determined iff no stabilizer has an X-part on q.
        let bit = 1u64 << q;
        for i in self.n..2 * self.n {
            if self.x[i] & bit != 0 {
                return 0.5;
            }
        }
        // Determined: Z_q eigenvalue +/-1.
        match self.group_value(0, bit, 0) {
            Some(v) => {
                if v > 0.0 {
                    0.0
                } else {
                    1.0
                }
            }
            None => 0.5, // defensive; should not happen
        }
    }

    /// Measure qubit `q` in the Z basis (randomness from the seeded RNG).
    pub fn measure(&mut self, q: u8, rng: &mut super::rng::Lcg64) -> u8 {
        let bit = 1u64 << q;
        // Find the first stabilizer with an X-part on q.
        let mut i: Option<usize> = None;
        for k in self.n..2 * self.n {
            if self.x[k] & bit != 0 {
                i = Some(k);
                break;
            }
        }
        match i {
            None => {
                // Determined outcome.
                match self.group_value(0, bit, 0) {
                    Some(v) => {
                        if v > 0.0 {
                            0
                        } else {
                            1
                        }
                    }
                    None => 0,
                }
            }
            Some(i) => {
                // Random outcome (p = 1/2). Projection update from first
                // principles:
                //   1. capture the pivot stabilizer row i (the one with an
                //      X-part on q);
                //   2. row-sum it into every other row with an X-part on q
                //      (so only row i has an X-part on q);
                //   3. replace stabilizer row i with +/- Z_q (phase encodes
                //      the outcome);
                //   4. the pivot row becomes the new destabilizer at i-n.
                let outcome = if rng.next_f64() < 0.5 { 1 } else { 0 };
                let (px, pz, pr) = (self.x[i], self.z[i], self.r[i]);
                for j in 0..2 * self.n {
                    if j != i && (self.x[j] & bit) != 0 {
                        self.rowsum_into(j, px, pz, pr);
                    }
                }
                self.x[i] = 0;
                self.z[i] = bit;
                self.r[i] = 2 * outcome;
                self.x[i - self.n] = px;
                self.z[i - self.n] = pz;
                self.r[i - self.n] = pr;
                outcome
            }
        }
    }

    /// Measure in the X basis: conjugate by H, measure Z.
    pub fn measure_x(&mut self, q: u8, rng: &mut super::rng::Lcg64) -> u8 {
        self.apply_h(q);
        let o = self.measure(q, rng);
        self.apply_h(q);
        o
    }

    /// Measure in the Y basis: conjugate by S^dagger H (Y = -i S H Z H S...).
    pub fn measure_y(&mut self, q: u8, rng: &mut super::rng::Lcg64) -> u8 {
        // Y = Z X up to phase; measuring Y: apply H then S to map Y -> Z.
        // U Y U^dag = Z with U = H S (verify: (HS) Y (HS)^dag).
        // For a stabilizer engine we only need the outcome statistics: the
        // conjugation by a Clifford maps the measurement basis; the state is
        // tracked consistently. Apply H then S^dag then measure Z then undo.
        self.apply_h(q);
        self.apply_s(q);
        self.apply_s(q);
        self.apply_s(q); // S^dag = S^3
        let o = self.measure(q, rng);
        self.apply_s(q); // S
        self.apply_h(q);
        o
    }

    /// Collapse after a determined measurement (used by RESET): measure then
    /// restore the qubit to |0> if the outcome was 1.
    pub fn measure_and_reset(&mut self, q: u8, rng: &mut super::rng::Lcg64) -> u8 {
        let o = self.measure(q, rng);
        if o == 1 {
            self.apply_x(q);
        }
        o
    }

    /// Sample a computational-basis state from the stabilizer state by
    /// measuring all qubits (the standard stabilizer sampling algorithm).
    pub fn sample(&mut self, rng: &mut super::rng::Lcg64) -> usize {
        let mut out = 0usize;
        for q in 0..self.n as u8 {
            if self.measure(q, rng) == 1 {
                out |= 1usize << q;
            }
        }
        out
    }

    /// `<P>` for a Pauli code (2 bits per qubit: 00=I,01=X,10=Y,11=Z).
    pub fn expect_value(&self, pauli: u64) -> f64 {
        let mut tx = 0u64;
        let mut tz = 0u64;
        let mut ty = 0u64;
        for q in 0..32u64 {
            let p = (pauli >> (2 * q)) & 3;
            match p {
                1 => tx |= 1u64 << q,
                2 => ty |= 1u64 << q,
                3 => tz |= 1u64 << q,
                _ => {}
            }
        }
        // Y = X Z up to i; for the group membership only the x/z support
        // matters (Y has both an X and a Z component).
        tx ^= ty;
        tz ^= ty;
        let tphase = (ty.count_ones() as u32) & 3; // Y = i X Z
        // Anticommutation with any stabilizer -> expectation 0.
        for i in self.n..2 * self.n {
            let anti = ((tx & self.z[i]).count_ones() ^ (self.x[i] & tz).count_ones()) & 1;
            if anti == 1 {
                return 0.0;
            }
        }
        self.group_value(tx, tz, tphase).unwrap_or(0.0)
    }

    /// Collapse a measured qubit to a given outcome (used by the trait
    /// interface; for the stabilizer engine this is the measured state).
    pub fn collapse(&mut self, _q: u8, _outcome: u8) {}

    /// Total probability (always 1 for a stabilizer state).
    pub fn norm(&self) -> f64 {
        1.0
    }

    /// Debug: dump the tableau rows (x, z, phase).
    pub fn dump_tableau(&self) -> Vec<(u64, u64, u8)> {
        (0..2 * self.n).map(|i| (self.x[i], self.z[i], self.r[i])).collect()
    }
}
impl super::sim::SimBackend for StabilizerBackend {
    fn reset(&mut self) {
        StabilizerBackend::reset(self);
    }
    fn apply_h(&mut self, q: u8) {
        StabilizerBackend::apply_h(self, q);
    }
    fn apply_x(&mut self, q: u8) {
        StabilizerBackend::apply_x(self, q);
    }
    fn apply_cnot(&mut self, c: u8, t: u8) {
        StabilizerBackend::apply_cnot(self, c, t);
    }
    fn apply_toff(&mut self, _c1: u8, _c2: u8, _t: u8) {
        panic!("non-Clifford gate on the stabilizer engine");
    }
    fn apply_rz(&mut self, _q: u8, _theta: f64) {
        panic!("non-Clifford gate on the stabilizer engine");
    }
    fn apply_rx(&mut self, _q: u8, _theta: f64) {
        panic!("non-Clifford gate on the stabilizer engine");
    }
    fn apply_ry(&mut self, _q: u8, _theta: f64) {
        panic!("non-Clifford gate on the stabilizer engine");
    }
    fn apply_phase(&mut self, _q: u8, _theta: f64) {
        panic!("non-Clifford gate on the stabilizer engine");
    }
    fn apply_s(&mut self, q: u8) {
        StabilizerBackend::apply_s(self, q);
    }
    fn apply_t(&mut self, _q: u8) {
        panic!("non-Clifford gate on the stabilizer engine");
    }
    fn apply_sx(&mut self, _q: u8) {
        panic!("non-Clifford gate on the stabilizer engine");
    }
    fn apply_swap(&mut self, a: u8, b: u8) {
        StabilizerBackend::apply_swap(self, a, b);
    }
    fn apply_iswap(&mut self, a: u8, b: u8) {
        // ISWAP = (S_a ⊗ S_b) × CZ × SWAP, all Clifford; applied rightmost-first.
        StabilizerBackend::apply_swap(self, a, b);
        StabilizerBackend::apply_cz(self, a, b);
        StabilizerBackend::apply_s(self, a);
        StabilizerBackend::apply_s(self, b);
    }
    fn apply_cz(&mut self, a: u8, b: u8) {
        StabilizerBackend::apply_cz(self, a, b);
    }
    fn apply_cphase(&mut self, _a: u8, _b: u8, _theta: f64) {
        panic!("non-Clifford gate on the stabilizer engine");
    }
    fn apply_cswap(&mut self, _c: u8, _b: u8, _t: u8) {
        panic!("non-Clifford gate on the stabilizer engine");
    }
    fn apply_mcx(&mut self, _mask: u32, _t: u8) {
        panic!("non-Clifford gate on the stabilizer engine");
    }
    fn reset_qubit(&mut self, q: u8, rng: &mut super::rng::Lcg64) {
        let _ = self.measure_and_reset(q, rng);
    }
    fn measure(&mut self, q: u8, rng: &mut super::rng::Lcg64) -> u8 {
        StabilizerBackend::measure(self, q, rng)
    }
    fn measure_x(&mut self, q: u8, rng: &mut super::rng::Lcg64) -> u8 {
        StabilizerBackend::measure_x(self, q, rng)
    }
    fn measure_y(&mut self, q: u8, rng: &mut super::rng::Lcg64) -> u8 {
        StabilizerBackend::measure_y(self, q, rng)
    }
    fn sample(&mut self, rng: &mut super::rng::Lcg64) -> usize {
        StabilizerBackend::sample(self, rng)
    }
    fn expect_value(&self, pauli: u64) -> f64 {
        StabilizerBackend::expect_value(self, pauli)
    }
    fn save_state(&mut self, results: &mut super::sim::Results) {
        // SAVE_* are excluded from the Clifford path (they need the
        // exponential statevector); defensive panic if reached.
        let _ = results;
        panic!("SAVE_* requires the statevector backend");
    }
    fn save_amplitudes(&mut self, results: &mut super::sim::Results) {
        let _ = results;
        panic!("SAVE_* requires the statevector backend");
    }
    fn save_probabilities(&mut self, results: &mut super::sim::Results) {
        let _ = results;
        panic!("SAVE_* requires the statevector backend");
    }
}
