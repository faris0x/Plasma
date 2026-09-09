// Copyright (c) 2026 Faris Alfarhan
// SPDX-License-Identifier: GPL-3.0-only

//! Deterministic, heap-free float math for the `no_std` core.
//!
//! `core::f64` provides no transcendental methods on this toolchain, so the
//! gates that need `sin`/`cos`/`sqrt` use faithful f64 ports of the
//! deterministic algorithms from NumCore (`numcore/src/math`):
//!
//!   - `sin_cos`: 22-iteration CORDIC rotation mode plus a first-order Taylor
//!     correction on the residual angle, with quadrant folding. Worst-case
//!     error ~2.3e-13. Pure shift/add -> bit-deterministic on every platform.
//!   - `sqrt`: CLZ + 32-entry LUT initial guess for 1/√x, 3 Newton-Raphson
//!     iterations on the reciprocal, then one final Newton step refining √x.
//!
//! Both are self-contained copies (the algorithm, not a dependency): Plasma
//! and planckOS stay independent, and the result is reproducible and
//! cross-platform deterministic, matching the project's determinism brand.

/// CORDIC rotation-mode gain K = ∏ cos(atan(2^-i)) ≈ 0.60725293501.
const CORDIC_GAIN: f64 = 0.6072529350088812561694;

/// CORDIC atan table: entry i = atan(2^-i) radians. 22 iterations.
const CORDIC_ATAN: [f64; 22] = [
    0.7853981633974483,
    0.4636476090008061,
    0.2449786631268641,
    0.1243549945467614,
    0.0624188099959573,
    0.0312398334302683,
    0.0156237286204768,
    0.0078123410601011,
    0.0039062301319669,
    0.0019531225164788,
    0.0009765621895593,
    0.0004882812111949,
    0.0002441406201494,
    0.0001220703118937,
    0.0000610351561742,
    0.0000305175781155,
    0.0000152587890613,
    0.0000076293945311,
    0.0000038146972656,
    0.0000019073486328,
    0.0000009536743164,
    0.0000004768371582,
];

/// π (CORDIC convergence / folding bound).
const PI: f64 = 3.141592653589793;
/// π/2 (quadrant-folding threshold).
const PI_OVER_2: f64 = 1.5707963267948966;

/// √2 for the LUT exponent scaling.
const SQRT2: f64 = 1.4142135623730951;
/// 1/√2 for the LUT exponent scaling.
const INV_SQRT2: f64 = 0.7071067811865475;

/// LUT of 1/√m for m sampled at {1 + k/64 : k = 0..31}; error < 2%.
/// Derived from NumCore's Q31.32 RSQRT_INIT_TABLE (values / 2^32).
const RSQRT_INIT_TABLE: [f64; 32] = [
    0.9924120378, 0.9774365818, 0.9631394502, 0.9494631699,
    0.9363571075, 0.9237735835, 0.9116694064, 0.9000051841,
    0.8887447298, 0.8778556340, 0.8673082046, 0.8570755048,
    0.8471332207, 0.8374596014, 0.8280352937, 0.8188431865,
    0.8098693356, 0.8010961781, 0.7925132204, 0.7841054802,
    0.7758665398, 0.7677876962, 0.7598618536, 0.7520809333,
    0.7444399848, 0.7369321888, 0.7295537757, 0.7222968445,
    0.7151589041, 0.7081337792, 0.7012135898, 0.6943985634,
];

/// Reduce a radian angle to the principal range [-π, π].
fn reduce_angle(x: f64) -> f64 {
    let two_pi = 2.0 * PI;
    // `x % two_pi` (core f64 `%` is available) then one conditional shift.
    let mut a = x % two_pi;
    if a > PI {
        a -= two_pi;
    } else if a < -PI {
        a += two_pi;
    }
    a
}

/// Raw CORDIC rotation mode for an angle in (-π/2, π/2).
///
/// 22 iterations (i = 0..21) plus a first-order Taylor correction on the
/// residual `z`: cos(θ0+δ) ≈ cos(θ0) - δ·sin(θ0), sin(θ0+δ) ≈ sin(θ0) + δ·cos(θ0).
/// Returns `(sin, cos)`.
fn cordic_raw(angle: f64) -> (f64, f64) {
    if angle == 0.0 {
        return (0.0, 1.0);
    }
    let mut x = CORDIC_GAIN; // K·cos(θ0)
    let mut y = 0.0; // K·sin(θ0)
    let mut z = angle;

    for i in 0..22 {
        let xp = x;
        let yp = y;
        if z >= 0.0 {
            x = xp - yp * (1.0 / (1u64 << i) as f64);
            y = yp + xp * (1.0 / (1u64 << i) as f64);
            z -= CORDIC_ATAN[i];
        } else {
            x = xp + yp * (1.0 / (1u64 << i) as f64);
            y = yp - xp * (1.0 / (1u64 << i) as f64);
            z += CORDIC_ATAN[i];
        }
    }

    // First-order Taylor correction for the residual angle δ = z.
    // cos(θ0+δ) ≈ cos(θ0) - δ·sin(θ0),  sin(θ0+δ) ≈ sin(θ0) + δ·cos(θ0).
    // With (x, y) = (K·cos θ0, K·sin θ0): x_out = x - y·δ, y_out = y + x·δ.
    let delta = z;
    let ty = y * delta; // y·δ
    let tx = x * delta; // x·δ
    let x_out = x - ty;
    let y_out = y + tx;

    (y_out, x_out)
}

/// sin/cos of a radian angle in f64, via CORDIC with quadrant folding.
/// Worst-case error ~2.3e-13. Returns `(sin, cos)`.
pub fn sin_cos(angle: f64) -> (f64, f64) {
    let a = reduce_angle(angle);
    let mut a = a;
    let mut negate_sin = false;
    let mut negate_cos = false;

    // Fold quadrant II: (π/2, π) → (0, π/2), cos negated.
    if a > PI_OVER_2 {
        a = PI - a;
        negate_cos = true;
    } else if a < -PI_OVER_2 {
        // Fold quadrant III: (-π, -π/2) → (-π/2, 0), sin negated.
        a = -a;
        negate_sin = true;
        if a > PI_OVER_2 {
            a = PI - a;
            negate_cos = true;
        }
    }

    let (sin_v, cos_v) = cordic_raw(a);
    let sin_out = if negate_sin { -sin_v } else { sin_v };
    let cos_out = if negate_cos { -cos_v } else { cos_v };
    (sin_out, cos_out)
}

/// 2^k as an f64, built via the exponent bits (exact for |k| ≤ 1023).
fn pow2i(k: i64) -> f64 {
    let exp_field = (k + 1023) as u64;
    f64::from_bits(exp_field << 52)
}

/// 32-entry LUT initial guess for 1/√x. Returns an approximation accurate to
/// <2% (sufficient as a Newton seed). `x > 0`.
fn rsqrt_initial(x: f64) -> f64 {
    // Decompose x = m · 2^e with m ∈ [1, 2) via the exponent bits.
    let bits = x.to_bits();
    let exponent = ((bits >> 52) & 0x7FF) as i64;
    // Unbiased exponent; with m ∈ [1,2) the top implicit bit is set.
    let e = exponent - 1023;

    // Mantissa fraction (52 bits), m = 1.f.
    let frac = bits & ((1u64 << 52) - 1);
    // Index into the 32-entry LUT using the top 5 fraction bits.
    let idx = ((frac >> (52 - 5)) as usize) & 31;
    let lut_val = RSQRT_INIT_TABLE[idx];

    // 1/√(m·2^e) = (1/√m) · 2^(-e/2); correct for the odd/even exponent.
    // Exponent scaling uses pow2i, the exponent can be ~±1000 for extreme
    // inputs, far beyond u64 shift range.
    if e >= 0 {
        if e & 1 == 0 {
            lut_val * pow2i(-(e / 2))
        } else {
            lut_val * INV_SQRT2 * pow2i(-((e - 1) / 2))
        }
    } else {
        let neg_e = -e;
        if neg_e & 1 == 0 {
            lut_val * pow2i(neg_e / 2)
        } else {
            lut_val * SQRT2 * pow2i((neg_e - 1) / 2)
        }
    }
}

/// Deterministic sqrt for `x >= 0`, via CLZ/LUT initial guess + 3 Newton
/// iterations on 1/√x, then one final Newton step refining √x directly.
/// Pure arithmetic, so bit-identical on every platform.
pub fn sqrt(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    // y = 1/√x: 3 Newton iterations on y ← y·(3 - x·y^2)/2.
    let mut y = rsqrt_initial(x);
    for _ in 0..3 {
        let y_sq = y * y;
        let x_y_sq = x * y_sq;
        y = y * (3.0 - x_y_sq) * 0.5;
    }
    // s = x·y ≈ √x; one final Newton refinement s ← (s + x/s)/2.
    let mut s = x * y;
    s = (s + x / s) * 0.5;
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_sin_cos_known_values() {
        // π/2, π, 0
        assert!(close(sin_cos(0.0).0, 0.0, 1e-12));
        assert!(close(sin_cos(0.0).1, 1.0, 1e-12));
        let (s, c) = sin_cos(PI_OVER_2);
        assert!(close(s, 1.0, 1e-12));
        assert!(close(c, 0.0, 1e-12));
        let (s, c) = sin_cos(PI);
        assert!(close(s, 0.0, 1e-12));
        assert!(close(c, -1.0, 1e-12));
    }

    #[test]
    fn test_sin_cos_identity() {
        // sin^2 + cos^2 ≈ 1 across a sweep (error budget 2.3e-13 headline).
        let mut worst = 0.0f64;
        let mut i = -1000;
        while i <= 1000 {
            let a = (i as f64) * 0.003;
            let (s, c) = sin_cos(a);
            let e = (s * s + c * c - 1.0).abs();
            if e > worst {
                worst = e;
            }
            i += 1;
        }
        assert!(worst < 1e-9, "worst sin^2+cos^2 error {worst:.3e}");
    }

    #[test]
    fn test_sqrt_perfect_squares() {
        assert!(close(sqrt(4.0), 2.0, 1e-12));
        assert!(close(sqrt(9.0), 3.0, 1e-12));
        assert!(close(sqrt(1.0), 1.0, 1e-12));
        assert_eq!(sqrt(0.0), 0.0);
    }

    #[test]
    fn test_sqrt_accuracy() {
        // Relative error < 1e-9 across a wide range.
        let mut worst = 0.0f64;
        let mut i = 1;
        while i < 100000 {
            let x = (i as f64) * 0.037 + 0.001;
            let r = sqrt(x);
            let rel = ((r * r - x) / x).abs();
            if rel > worst {
                worst = rel;
            }
            i += 1;
        }
        assert!(worst < 1e-9, "worst sqrt relative error {worst:.3e}");
    }

    #[test]
    fn test_sqrt_large_and_tiny() {
        // Exponent scaling across a huge dynamic range.
        assert!(close(sqrt(1e300), 1e150, 1e140));
        assert!(close(sqrt(1e-300), 1e-150, 1e-160));
    }
}
