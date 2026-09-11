// Copyright (c) 2026 Faris Alfarhan
// SPDX-License-Identifier: GPL-3.0-only

//! Matrix-product-state (MPS) simulation engine.
//!
//! An MPS represents the state as a chain of complex tensors `A[k]` of shape
//! `[D_k, 2, D_{k+1}]` (left bond, physical, right bond), with
//! `D_0 = D_n = 1`:
//!
//! ```text
//! |ψ> = sum_{i} A[0][i0] A[1][i1] ... A[n-1][i_{n-1}] |i0 ... i_{n-1}>
//! ```
//!
//! Gates: single-qubit gates are a local tensor contraction; two-qubit gates
//! on adjacent sites are applied by contracting the pair, applying the gate,
//! and re-splitting with a truncated SVD (Jacobi eigendecomposition of the
//! Gram matrix). Non-adjacent gates are routed with SWAPs.
//! Each truncation discards the smallest singular values; the accumulated
//! squared-norm error is tracked and reported (`--mps D`), so the engine is
//! exact when the entanglement stays within the bond dimension and
//! approximate (with a reported error) when it does not.
//!
//! Sampling: the per-site marginal is computed by contracting the site with
//! a right environment (all sites to the right, built once per sample in
//! O(n × D^3)) and a left density matrix that accumulates the collapse of all
//! sites sampled so far. Total cost is O(n × D^3), exact for any MPS (no
//! canonical-form assumption), and consumes the seeded host RNG in a fixed
//! site order.
//!
//! Determinism: all sampling uses the seeded host RNG in a fixed order.

use alloc::vec;
use alloc::vec::Vec;

/// A complex number stored as (re, im).
#[derive(Clone, Copy, Debug)]
pub(crate) struct C {
    re: f64,
    im: f64,
}

impl C {
    #[inline]
    pub(crate) const fn new(re: f64, im: f64) -> Self {
        C { re, im }
    }
    #[inline]
    pub(crate) const fn zero() -> Self {
        C { re: 0.0, im: 0.0 }
    }
    #[inline]
    pub(crate) const fn one() -> Self {
        C { re: 1.0, im: 0.0 }
    }
    #[inline]
    fn conj(&self) -> Self {
        C::new(self.re, -self.im)
    }
    #[inline]
    fn add(&self, o: &C) -> C {
        C::new(self.re + o.re, self.im + o.im)
    }
    #[inline]
    fn mul(&self, o: &C) -> C {
        C::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
    #[inline]
    fn scale(&self, s: f64) -> C {
        C::new(self.re * s, self.im * s)
    }
    #[inline]
    fn norm2(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

pub struct MpsBackend {
    n: usize,
    dmax: usize,
    /// Tensors flattened `[l, phys, r]`.
    a: Vec<Vec<C>>,
    dims: Vec<usize>, // dims[k] = left bond dim of site k; dims[n] = 1
    /// Physical-to-site permutation (site index -> physical qubit).
    perm: Vec<usize>,
    /// Accumulated squared-norm lost to SVD truncation.
    pub trunc_err: f64,
}

impl MpsBackend {
    pub fn new(num_qubits: u8, dmax: usize) -> Self {
        let n = num_qubits as usize;
        let mut a = Vec::with_capacity(n);
        let dims = vec![1usize; n + 1];
        for _ in 0..n {
            // |0>: A[k][0, 0, 0] = 1, A[k][0, 1, 0] = 0  (shape [1, 2, 1]).
            a.push(vec![C::one(), C::zero()]);
        }
        let perm: Vec<usize> = (0..n).collect();
        MpsBackend {
            n,
            dmax: dmax.max(1),
            a,
            dims,
            perm,
            trunc_err: 0.0,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.n as u8, self.dmax);
    }

    /// Collapse position `k` to the given outcome (used by sampling).
    pub fn collapse_position(&mut self, k: usize, outcome: usize) {
        let d_l = self.dims[k];
        let d_r = self.dims[k + 1];
        let mut keep = vec![C::zero(); d_l * d_r];
        for l in 0..d_l {
            for r in 0..d_r {
                keep[l * d_r + r] = self.a[k][(l * 2 + outcome) * d_r + r];
            }
        }
        if k + 1 < self.n {
            let d_nr = self.dims[k + 2];
            let mut a_next = vec![C::zero(); d_l * 2 * d_nr];
            for l in 0..d_l {
                for j in 0..2 {
                    for m in 0..d_nr {
                        let mut acc = C::zero();
                        for r in 0..d_r {
                            acc = acc.add(&keep[l * d_r + r].mul(&self.a[k + 1][(r * 2 + j) * d_nr + m]));
                        }
                        a_next[(l * 2 + j) * d_nr + m] = acc;
                    }
                }
            }
            self.a[k + 1] = a_next;
            self.dims[k + 1] = d_l;
        }
        let rb = if k + 1 < self.n { d_l } else { 1 };
        let mut t = vec![C::zero(); 1 * 2 * rb];
        for b in 0..rb {
            t[outcome * rb + b] = C::one();
        }
        self.a[k] = t;
        self.dims[k] = 1;
    }

    /// The site index currently holding physical qubit `p`.
    fn site_of(&self, p: usize) -> usize {
        self.perm.iter().position(|&x| x == p).unwrap()
    }

    /// Apply a 2×2 complex unitary to physical site `p`.
    fn apply_single_u(&mut self, p: usize, u: &[C; 4]) {
        let k = self.site_of(p);
        let d_l = self.dims[k];
        let d_r = self.dims[k + 1];
        let mut t = vec![C::zero(); d_l * 2 * d_r];
        for l in 0..d_l {
            for i in 0..2 {
                for r in 0..d_r {
                    let mut acc = C::zero();
                    for j in 0..2 {
                        acc = acc.add(&u[i * 2 + j].mul(&self.a[k][(l * 2 + j) * d_r + r]));
                    }
                    t[(l * 2 + i) * d_r + r] = acc;
                }
            }
        }
        self.a[k] = t;
    }

    /// Apply a 4×4 complex unitary to physical sites `pa` and `pb`, routing
    /// non-adjacent pairs to adjacent sites with SWAPs (tracking the
    /// permutation), then routing back.
    fn apply_two_u(&mut self, pa: usize, pb: usize, u: &[C; 16]) {
        let (ka, kb) = (self.site_of(pa), self.site_of(pb));
        if ka.abs_diff(kb) == 1 {
            self.apply_two_adjacent(ka.min(kb), u);
            return;
        }
        let lo = ka.min(kb);
        let hi = ka.max(kb);
        for s in (lo + 1..hi).rev() {
            self.apply_two_adjacent(s, &SWAP4);
            self.perm.swap(s, s + 1);
        }
        self.apply_two_adjacent(lo, u);
        for s in lo + 1..hi {
            self.apply_two_adjacent(s, &SWAP4);
            self.perm.swap(s, s + 1);
        }
    }

    /// Apply a two-qubit gate to adjacent sites `k, k+1`: contract, apply the
    /// gate, and re-split with a truncated SVD.
    fn apply_two_adjacent(&mut self, k: usize, u: &[C; 16]) {
        let d_l = self.dims[k];
        let d_m = self.dims[k + 1];
        let d_r = self.dims[k + 2];
        let mut c = vec![C::zero(); d_l * 2 * 2 * d_r];
        for l in 0..d_l {
            for i in 0..2 {
                for m in 0..d_m {
                    for j in 0..2 {
                        for r in 0..d_r {
                            let cidx = ((l * 2 + i) * 2 + j) * d_r + r;
                            c[cidx] = c[cidx].add(&self.a[k][(l * 2 + i) * d_m + m].mul(&self.a[k + 1][(m * 2 + j) * d_r + r]));
                        }
                    }
                }
            }
        }
        let mut g = vec![C::zero(); d_l * 2 * 2 * d_r];
        for l in 0..d_l {
            for i in 0..2 {
                for j in 0..2 {
                    for ip in 0..2 {
                        for jp in 0..2 {
                            let ug = u[(i * 2 + j) * 4 + (ip * 2 + jp)];
                            if ug.re == 0.0 && ug.im == 0.0 {
                                continue;
                            }
                            for r in 0..d_r {
                                let src = ((l * 2 + ip) * 2 + jp) * d_r + r;
                                let dst = ((l * 2 + i) * 2 + j) * d_r + r;
                                g[dst] = g[dst].add(&ug.mul(&c[src]));
                            }
                        }
                    }
                }
            }
        }
        let m = d_l * 2;
        let nn = 2 * d_r;
        let (x, sv, yh, err) = svd_truncate(&g, m, nn, self.dmax);
        self.trunc_err += err;
        let d = sv.len();
        let mut a_k = vec![C::zero(); d_l * 2 * d];
        for l in 0..d_l {
            for i in 0..2 {
                for b in 0..d {
                    a_k[(l * 2 + i) * d + b] = x[(l * 2 + i) * d + b];
                }
            }
        }
        let mut a_k1 = vec![C::zero(); d * 2 * d_r];
        for b in 0..d {
            for j in 0..2 {
                for r in 0..d_r {
                    a_k1[(b * 2 + j) * d_r + r] = yh[b * (2 * d_r) + j * d_r + r].scale(sv[b]);
                }
            }
        }
        self.a[k] = a_k;
        self.a[k + 1] = a_k1;
        self.dims[k + 1] = d;
    }

    /// Sampling: exact marginal sampling in O(n × D^3) total. A right
    /// environment is built once (contraction of all sites to the right of
    /// each site) and the collapse of previously-sampled sites is carried in
    /// a left density matrix. Works for any MPS (no canonical-form
    /// assumption) and consumes the host RNG once per site in site order,
    /// matching the RNG semantics of the sandwich path.
    pub fn sample_outcome(&mut self, rng: &mut super::rng::Lcg64) -> usize {
        let env = self.right_envs();
        let mut out = 0usize;
        let mut left: Vec<C> = vec![C::one()]; // dims[0] = 1
        for k in 0..self.n {
            let d_l = self.dims[k];
            let d_r = self.dims[k + 1];
            let mut pr = [0.0f64; 2];
            for outcome in 0..2 {
                // P = tr(left . A_k[outcome] . R_k . A_k[outcome]^dag) via
                // M = B^dag . left . B, P = sum Re(M . conj(R)).
                let b = self.slice_b(k, d_l, d_r, outcome);
                let m = left_b_dag(&left, &b, d_l, d_r);
                let mut p = 0.0;
                for r in 0..d_r {
                    for rp in 0..d_r {
                        let e = env[k][r * d_r + rp];
                        let mv = m[r * d_r + rp];
                        p += mv.re * e.re + mv.im * e.im;
                    }
                }
                pr[outcome] = p;
            }
            let total = pr[0] + pr[1];
            let outcome: usize = if total > 0.0 && rng.next_f64() * total < pr[0] { 0 } else { 1 };
            // Fold the collapsed site into the left density matrix.
            let b = self.slice_b(k, d_l, d_r, outcome);
            left = left_b_dag(&left, &b, d_l, d_r);
            out |= outcome << self.perm[k];
        }
        out
    }

    /// The matrix `B[a, r] = A[k][(a, outcome), r]` (a `d_l × d_r` slice).
    fn slice_b(&self, k: usize, d_l: usize, d_r: usize, outcome: usize) -> Vec<C> {
        let mut b = vec![C::zero(); d_l * d_r];
        for a in 0..d_l {
            for r in 0..d_r {
                b[a * d_r + r] = self.a[k][(a * 2 + outcome) * d_r + r];
            }
        }
        b
    }

    /// Right environments `R[k]`: the `[dims[k+1] × dims[k+1]]` contraction
    /// of all sites to the right of `k` (with their mutual bonds). `R[n-1]`
    /// is the identity on the trivial last bond. Built once, O(n × D^3).
    fn right_envs(&self) -> Vec<Vec<C>> {
        let mut env: Vec<Vec<C>> = Vec::with_capacity(self.n);
        for _ in 0..self.n {
            env.push(Vec::new());
        }
        env[self.n - 1] = vec![C::one()];
        for k in (0..self.n - 1).rev() {
            let s = k + 1;
            let d_l = self.dims[s];
            let d_r = self.dims[s + 1];
            let mut acc = vec![C::zero(); d_l * d_l];
            for i in 0..2 {
                // B[a, c] = A[s][(a, i), c]; acc += B . R[k+1] . B^dag.
                let b = self.slice_b(s, d_l, d_r, i);
                let tmp = mmul(&b, d_l, d_r, &env[k + 1], d_r);
                for a in 0..d_l {
                    for ap in 0..d_l {
                        let mut v = C::zero();
                        for c in 0..d_r {
                            v = v.add(&tmp[a * d_r + c].mul(&b[ap * d_r + c].conj()));
                        }
                        acc[a * d_l + ap] = acc[a * d_l + ap].add(&v);
                    }
                }
            }
            env[k] = acc;
        }
        env
    }

    /// Measure one physical qubit: route it to the front, sample/collapse it.
    pub fn measure_physical(&mut self, p: usize, rng: &mut super::rng::Lcg64) -> u8 {
        let k = self.site_of(p);
        for s in (0..k).rev() {
            self.apply_two_adjacent(s, &SWAP4);
            self.perm.swap(s, s + 1);
        }
        let pr = self.site_marginals(0);
        let total = pr[0] + pr[1];
        let outcome: u8 = if total > 0.0 && rng.next_f64() * total < pr[0] { 0 } else { 1 };
        self.collapse_position(0, outcome as usize);
        outcome
    }

    /// Compute <P> for a Pauli code by the bra-ket sandwich contraction
    /// (the correct observable for a left-canonical MPS).
    pub fn expect_value(&self, pauli: u64) -> f64 {
        // Left environment E[a, a'] (bra bond, ket bond), starting at [1,1].
        let mut e = vec![C::zero(); 1 * 1];
        e[0] = C::one();
        for k in 0..self.n {
            let p = (pauli >> (2 * self.perm[k])) & 3;
            let op: [C; 4] = match p {
                1 => [C::zero(), C::one(), C::one(), C::zero()],
                2 => [C::zero(), C::new(0.0, -1.0), C::new(0.0, 1.0), C::zero()],
                3 => [C::one(), C::zero(), C::zero(), C::new(-1.0, 0.0)],
                _ => [C::one(), C::zero(), C::zero(), C::one()],
            };
            let d_l = self.dims[k];
            let d_r = self.dims[k + 1];
            let mut ne = vec![C::zero(); d_r * d_r];
            for a in 0..d_l {
                for ap in 0..d_l {
                    if e[a * d_l + ap].norm2() == 0.0 {
                        continue;
                    }
                    for i in 0..2 {
                        for j in 0..2 {
                            let o = op[i * 2 + j];
                            if o.norm2() == 0.0 {
                                continue;
                            }
                            for c in 0..d_r {
                                for cp in 0..d_r {
                                    let ak_ai = self.a[k][(a * 2 + i) * d_r + c];
                                    let ak_apj = self.a[k][(ap * 2 + j) * d_r + cp];
                                    ne[c * d_r + cp] = ne[c * d_r + cp].add(
                                        &ak_ai.conj().mul(&o).mul(&ak_apj).mul(&e[a * d_l + ap]),
                                    );
                                }
                            }
                        }
                    }
                }
            }
            e = ne;
        }
        e[0].re
    }

    /// The marginal probabilities P(q_k = 0), P(q_k = 1) by a full
    /// bra-ket sandwich contraction with the projector |i><i| on site k.
    /// Exact for any MPS (no canonical-form assumption); O(n D^3).
    pub fn site_marginals(&self, k: usize) -> [f64; 2] {
        let mut out = [0.0f64; 2];
        for outcome in 0..2 {
            let mut e = vec![C::zero(); 1];
            e[0] = C::one();
            for j in 0..self.n {
                let op: [C; 4] = if j == k {
                    match outcome {
                        0 => [C::one(), C::zero(), C::zero(), C::zero()],
                        _ => [C::zero(), C::zero(), C::zero(), C::one()],
                    }
                } else {
                    [C::one(), C::zero(), C::zero(), C::one()]
                };
                let d_l = self.dims[j];
                let d_r = self.dims[j + 1];
                let mut ne = vec![C::zero(); d_r * d_r];
                for a in 0..d_l {
                    for ap in 0..d_l {
                        if e[a * d_l + ap].norm2() == 0.0 {
                            continue;
                        }
                        for i in 0..2 {
                            for ip in 0..2 {
                                let o = op[i * 2 + ip];
                                if o.norm2() == 0.0 {
                                    continue;
                                }
                                for r in 0..d_r {
                                    for rp in 0..d_r {
                                        ne[r * d_r + rp] = ne[r * d_r + rp].add(
                                            &self.a[j][(a * 2 + i) * d_r + r]
                                                .conj()
                                                .mul(&o)
                                                .mul(&self.a[j][(ap * 2 + ip) * d_r + rp])
                                                .mul(&e[a * d_l + ap]),
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                e = ne;
            }
            out[outcome] = e[0].re;
        }
        out
    }

    /// Norm of the state (1.0 for a left-canonical MPS, up to truncation).
    pub fn norm(&self) -> f64 {
        self.expect_value(0)
    }
}

/// Complex matrix multiply: `a` is `ma × na`, `b` is `na × nb`; result
/// `ma × nb` flattened row-major.
fn mmul(a: &[C], ma: usize, na: usize, b: &[C], nb: usize) -> Vec<C> {
    let mut out = vec![C::zero(); ma * nb];
    for i in 0..ma {
        for k in 0..na {
            let av = a[i * na + k];
            if av.re == 0.0 && av.im == 0.0 {
                continue;
            }
            for j in 0..nb {
                out[i * nb + j] = out[i * nb + j].add(&av.mul(&b[k * nb + j]));
            }
        }
    }
    out
}

/// `B^dag · left · B` for `left` (d_l × d_l) and `B` (d_l × d_r), returning
/// a `d_r × d_r` matrix. Used to fold a collapsed site into the left density
/// matrix and to form the marginal sandwich.
fn left_b_dag(left: &[C], b: &[C], d_l: usize, d_r: usize) -> Vec<C> {
    let tmp = mmul(left, d_l, d_l, b, d_r);
    let mut out = vec![C::zero(); d_r * d_r];
    for r in 0..d_r {
        for rp in 0..d_r {
            let mut v = C::zero();
            for a in 0..d_l {
                v = v.add(&b[a * d_r + r].conj().mul(&tmp[a * d_r + rp]));
            }
            out[r * d_r + rp] = v;
        }
    }
    out
}

/// Truncated SVD of an m x n complex matrix `g` (flattened), keeping the
/// largest `dmax` singular values. Returns (U, singular_values, V^dag, error)
/// with U of shape [m x d] and V^dag of shape [d x n], where
/// `g = U S V^dag`. Uses the Gram eigendecomposition: V from the Hermitian
/// `G = g^dag g` (Jacobi rotations), then U = g V S^-1.
pub(crate) fn svd_truncate(g: &[C], m: usize, n: usize, dmax: usize) -> (Vec<C>, Vec<f64>, Vec<C>, f64) {
    // Gram matrix G = g^dag g (n x n), Hermitian.
    let mut gr = vec![0.0f64; n * n];
    let mut gi = vec![0.0f64; n * n];
    for a in 0..n {
        for b in 0..n {
            let mut acc = C::zero();
            for i in 0..m {
                acc = acc.add(&g[i * n + a].conj().mul(&g[i * n + b]));
            }
            gr[a * n + b] = acc.re;
            gi[a * n + b] = acc.im;
        }
    }
    // Eigendecomposition of the Hermitian matrix G -> V (columns are
    // eigenvectors), eigenvalues in `ev`. Uses exact 2×2 block
    // diagonalization: for each off-diagonal pair, the 2×2 Hermitian block is
    // diagonalized in closed form (correct by construction) and the unitary
    // is applied to the full matrix. Converges quadratically.
    let mut vr = gr.clone();
    let mut vi = gi.clone();
    let mut eig: Vec<C> = (0..n * n).map(|i| if i % (n + 1) == 0 { C::one() } else { C::zero() }).collect();
    let mut ev = vec![0.0f64; n];
    for _ in 0..100 {
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += vr[p * n + q] * vr[p * n + q] + vi[p * n + q] * vi[p * n + q];
            }
        }
        if off < 1e-44 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let app = vr[p * n + p];
                let aqq = vr[q * n + q];
                let apq = C::new(vr[p * n + q], vi[p * n + q]);
                let mag2 = apq.norm2();
                if mag2 < 1e-300 {
                    continue;
                }
                let half = (aqq - app) / 2.0;
                let delta = super::math::sqrt(half * half + mag2);
                // Put the larger eigenvalue at position p.
                let lam_p = (app + aqq) / 2.0 + delta;
                let lam_q = (app + aqq) / 2.0 - delta;
                // Eigenvector for λ_p: v = [apq, λ_p - app].
                let mut v1 = C::new(apq.re, apq.im);
                let v2 = C::new(lam_p - app, 0.0);
                let mut n1 = v1.norm2() + v2.norm2();
                if n1 < 1e-300 {
                    continue;
                }
                n1 = 1.0 / super::math::sqrt(n1);
                v1 = v1.scale(n1);
                let v2 = v2.scale(n1);
                // Orthogonal partner: v_lo = [conj(v2), -conj(v1)].
                let w1 = v2.conj();
                let w2 = v1.conj().scale(-1.0);
                // U2 = [v, w] as columns: [[v1, w1], [v2, w2]].
                let (u11, u21) = (v1, v2); // column 0
                let (u12, u22) = (w1, w2); // column 1
                // A' = U2^dag A U2.
                for k in 0..n {
                    if k == p || k == q {
                        continue;
                    }
                    let a_kp = C::new(vr[k * n + p], vi[k * n + p]);
                    let a_kq = C::new(vr[k * n + q], vi[k * n + q]);
                    // A'[k,p] = conj(u11)*A[k,p] + conj(u12)*A[k,q]
                    let n_kp = a_kp.mul(&u11.conj()).add(&a_kq.mul(&u12.conj()));
                    // A'[k,q] = conj(u21)*A[k,p] + conj(u22)*A[k,q]
                    let n_kq = a_kp.mul(&u21.conj()).add(&a_kq.mul(&u22.conj()));
                    vr[k * n + p] = n_kp.re;
                    vi[k * n + p] = n_kp.im;
                    vr[k * n + q] = n_kq.re;
                    vi[k * n + q] = n_kq.im;
                    vr[p * n + k] = n_kp.re;
                    vi[p * n + k] = -n_kp.im;
                    vr[q * n + k] = n_kq.re;
                    vi[q * n + k] = -n_kq.im;
                }
                // Diagonal block: U2^dag H2 U2 = diag(λ_p, λ_q).
                vr[p * n + p] = lam_p;
                vi[p * n + p] = 0.0;
                vr[q * n + q] = lam_q;
                vi[q * n + q] = 0.0;
                vr[p * n + q] = 0.0;
                vi[p * n + q] = 0.0;
                vr[q * n + p] = 0.0;
                vi[q * n + p] = 0.0;
                // V <- V U2.
                for k in 0..n {
                    let v_kp = eig[k * n + p];
                    let v_kq = eig[k * n + q];
                    eig[k * n + p] = v_kp.mul(&u11).add(&v_kq.mul(&u12));
                    eig[k * n + q] = v_kp.mul(&u21).add(&v_kq.mul(&u22));
                }
            }
        }
    }
    for i in 0..n {
        ev[i] = vr[i * n + i];
    }
    for i in 0..n {
        ev[i] = vr[i * n + i];
        if ev[i] < 0.0 && ev[i] > -1e-12 {
            ev[i] = 0.0;
        }
    }
    // Singular values = sqrt(eigenvalues), descending order; keep the top
    // `dmax` that are above the numerical-noise floor (a zero singular value
    // would pair with a garbage U column, breaking left-orthonormality).
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| ev[b].partial_cmp(&ev[a]).unwrap());
    let top = ev[idx[0]].max(0.0);
    let rank = idx.iter().take_while(|&&ii| ev[ii].max(0.0) > 1e-30 * top.max(1e-300)).count().max(1);
    let d = dmax.min(n).min(m).min(rank);
    let mut sv = vec![0.0f64; d];
    for (k, &ii) in idx.iter().take(d).enumerate() {
        sv[k] = super::math::sqrt(ev[ii].max(0.0));
    }
    // Truncation error: sum of discarded squared singular values / total.
    let total_sq: f64 = ev.iter().sum();
    let kept_sq: f64 = sv.iter().map(|s| s * s).sum();
    let err = if total_sq > 0.0 {
        (total_sq - kept_sq).max(0.0) / total_sq
    } else {
        0.0
    };
    // U = g V S^-1: [m x n] . [n x d] . diag(1/sv)
    let mut u = vec![C::zero(); m * d];
    for i in 0..m {
        for k in 0..d {
            let ii = idx[k];
            let mut acc = C::zero();
            for j in 0..n {
                acc = acc.add(&g[i * n + j].mul(&eig[j * n + ii]));
            }
            u[i * d + k] = acc.scale(1.0 / sv[k].max(1e-300));
        }
    }
    // V^dag [d x n]
    let mut vd = vec![C::zero(); d * n];
    for k in 0..d {
        let ii = idx[k];
        for j in 0..n {
            vd[k * n + j] = eig[j * n + ii].conj();
        }
    }
    (u, sv, vd, err)
}

/// The SWAP gate as a 4×4 matrix (row-major, |ij> -> |ji>).
const SWAP4: [C; 16] = {
    let z = C::new(0.0, 0.0);
    let o = C::new(1.0, 0.0);
    [
        o, z, z, z,
        z, z, o, z,
        z, o, z, z,
        z, z, z, o,
    ]
};
impl super::sim::SimBackend for MpsBackend {
    fn reset(&mut self) {
        self.reset();
    }
    fn apply_h(&mut self, q: u8) {
        let s = 1.0 / super::math::sqrt(2.0);
        let u = [C::new(s, 0.0), C::new(s, 0.0), C::new(s, 0.0), C::new(-s, 0.0)];
        self.apply_single_u(q as usize, &u);
    }
    fn apply_x(&mut self, q: u8) {
        let u = [C::zero(), C::one(), C::one(), C::zero()];
        self.apply_single_u(q as usize, &u);
    }
    fn apply_cnot(&mut self, c: u8, t: u8) {
        let z = C::zero();
        let o = C::one();
        // |00>->|00>, |01>->|01>, |10>->|11>, |11>->|10>
        let u = [
            o, z, z, z,
            z, o, z, z,
            z, z, z, o,
            z, z, o, z,
        ];
        self.apply_two_u(c as usize, t as usize, &u);
    }
    fn apply_toff(&mut self, _c1: u8, _c2: u8, _t: u8) {
        panic!("3-qubit gate on the MPS engine");
    }
    fn apply_rz(&mut self, q: u8, theta: f64) {
        let (s, c) = super::math::sin_cos(theta / 2.0);
        let u = [
            C::new(c, -s), C::zero(),
            C::zero(), C::new(c, s),
        ];
        self.apply_single_u(q as usize, &u);
    }
    fn apply_rx(&mut self, q: u8, theta: f64) {
        let (s, c) = super::math::sin_cos(theta / 2.0);
        let u = [
            C::new(c, 0.0), C::new(0.0, -s),
            C::new(0.0, -s), C::new(c, 0.0),
        ];
        self.apply_single_u(q as usize, &u);
    }
    fn apply_ry(&mut self, q: u8, theta: f64) {
        let (s, c) = super::math::sin_cos(theta / 2.0);
        let u = [
            C::new(c, 0.0), C::new(-s, 0.0),
            C::new(s, 0.0), C::new(c, 0.0),
        ];
        self.apply_single_u(q as usize, &u);
    }
    fn apply_phase(&mut self, q: u8, theta: f64) {
        let (s, c) = super::math::sin_cos(theta);
        let u = [C::one(), C::zero(), C::zero(), C::new(c, s)];
        self.apply_single_u(q as usize, &u);
    }
    fn apply_s(&mut self, q: u8) {
        let u = [C::one(), C::zero(), C::zero(), C::new(0.0, 1.0)];
        self.apply_single_u(q as usize, &u);
    }
    fn apply_t(&mut self, q: u8) {
        let s = super::math::sin_cos(0.7853981633974483096156608458198757210492923498437764);
        let u = [C::one(), C::zero(), C::zero(), C::new(s.1, s.0)];
        self.apply_single_u(q as usize, &u);
    }
    fn apply_sx(&mut self, q: u8) {
        // sqrt(X) = ((1+i)|0><0| + (1-i)|0><1| + (1-i)|1><0| + (1+i)|1><1|)/2
        let u = [
            C::new(0.5, 0.5), C::new(0.5, -0.5),
            C::new(0.5, -0.5), C::new(0.5, 0.5),
        ];
        self.apply_single_u(q as usize, &u);
    }
    fn apply_swap(&mut self, a: u8, b: u8) {
        self.apply_two_u(a as usize, b as usize, &SWAP4);
    }
    fn apply_iswap(&mut self, a: u8, b: u8) {
        let z = C::zero();
        let o = C::one();
        let i = C::new(0.0, 1.0);
        let u = [
            o, z, z, z,
            z, z, i, z,
            z, i, z, z,
            z, z, z, o,
        ];
        self.apply_two_u(a as usize, b as usize, &u);
    }
    fn apply_cz(&mut self, a: u8, b: u8) {
        let z = C::zero();
        let o = C::one();
        let m = C::new(-1.0, 0.0);
        let u = [
            o, z, z, z,
            z, o, z, z,
            z, z, o, z,
            z, z, z, m,
        ];
        self.apply_two_u(a as usize, b as usize, &u);
    }
    fn apply_cphase(&mut self, a: u8, b: u8, theta: f64) {
        let z = C::zero();
        let o = C::one();
        let (s, c) = super::math::sin_cos(theta);
        let u = [
            o, z, z, z,
            z, o, z, z,
            z, z, o, z,
            z, z, z, C::new(c, s),
        ];
        self.apply_two_u(a as usize, b as usize, &u);
    }
    fn apply_cswap(&mut self, _c: u8, _b: u8, _t: u8) {
        panic!("3-qubit gate on the MPS engine");
    }
    fn apply_mcx(&mut self, _mask: u32, _t: u8) {
        panic!("multi-control gate on the MPS engine");
    }
    fn reset_qubit(&mut self, q: u8, rng: &mut super::rng::Lcg64) {
        let o = self.measure_physical(q as usize, rng);
        if o == 1 {
            self.apply_x(q);
        }
    }
    fn measure(&mut self, q: u8, rng: &mut super::rng::Lcg64) -> u8 {
        self.measure_physical(q as usize, rng)
    }
    fn measure_x(&mut self, q: u8, rng: &mut super::rng::Lcg64) -> u8 {
        self.apply_h(q);
        let o = self.measure_physical(q as usize, rng);
        self.apply_h(q);
        o
    }
    fn measure_y(&mut self, q: u8, rng: &mut super::rng::Lcg64) -> u8 {
        self.apply_h(q);
        self.apply_s(q);
        self.apply_s(q);
        self.apply_s(q);
        let o = self.measure_physical(q as usize, rng);
        self.apply_s(q);
        self.apply_h(q);
        o
    }
    fn sample(&mut self, rng: &mut super::rng::Lcg64) -> usize {
        self.sample_outcome(rng)
    }
    fn expect_value(&self, pauli: u64) -> f64 {
        self.expect_value(pauli)
    }
    fn save_state(&mut self, _results: &mut super::sim::Results) {
        panic!("SAVE_* requires the statevector backend");
    }
    fn save_amplitudes(&mut self, _results: &mut super::sim::Results) {
        panic!("SAVE_* requires the statevector backend");
    }
    fn save_probabilities(&mut self, _results: &mut super::sim::Results) {
        panic!("SAVE_* requires the statevector backend");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h2() -> [C; 4] {
        let s = 1.0 / crate::math::sqrt(2.0);
        [C::new(s, 0.0), C::new(s, 0.0), C::new(s, 0.0), C::new(-s, 0.0)]
    }

    fn cnot4() -> [C; 16] {
        let mut u = [C::zero(); 16];
        u[0 * 4 + 0] = C::one(); // |00> -> |00>
        u[1 * 4 + 1] = C::one(); // |01> -> |01>
        u[3 * 4 + 2] = C::one(); // |10> -> |11>
        u[2 * 4 + 3] = C::one(); // |11> -> |10>
        u
    }

    fn build_4q(d: usize) -> MpsBackend {
        let mut m = MpsBackend::new(4, d);
        m.apply_single_u(0, &h2());
        m.apply_two_u(0, 1, &cnot4());
        m.apply_two_u(1, 2, &cnot4());
        m.apply_single_u(3, &h2());
        m.apply_two_u(2, 3, &cnot4());
        m
    }

    #[test]
    fn test_walk_marginals_match_sandwich() {
        // The left-walk marginals (right environment + folded left density
        // matrix) must match the bra-ket sandwich marginals of the reference
        // MPS collapsed step-by-step in the same order.
        for d in [1usize, 2, 4, 8] {
            let walk = build_4q(d);
            let mut reference = build_4q(d);
            let env = walk.right_envs();
            let mut left: Vec<C> = vec![C::one()];
            for k in 0..walk.n {
                let d_l = walk.dims[k];
                let d_r = walk.dims[k + 1];
                let mut pr = [0.0f64; 2];
                for outcome in 0..2 {
                    let b = walk.slice_b(k, d_l, d_r, outcome);
                    let mm = left_b_dag(&left, &b, d_l, d_r);
                    let mut p = 0.0;
                    for r in 0..d_r {
                        for rp in 0..d_r {
                            let e = env[k][r * d_r + rp];
                            let mv = mm[r * d_r + rp];
                            p += mv.re * e.re + mv.im * e.im;
                        }
                    }
                    pr[outcome] = p;
                }
                let sw = reference.site_marginals(k);
                for outcome in 0..2 {
                    let tol = 1e-9;
                    assert!(
                        (pr[outcome] - sw[outcome]).abs() < tol,
                        "d={d} site {k}: walk {} vs sandwich {}",
                        pr[outcome],
                        sw[outcome]
                    );
                }
                // Fold outcome 0 into the walk's left matrix and the reference
                // tensors identically.
                let b = walk.slice_b(k, d_l, d_r, 0);
                left = left_b_dag(&left, &b, d_l, d_r);
                reference.collapse_position(k, 0);
            }
        }
    }
}
