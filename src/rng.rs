// Copyright (c) 2026 Faris Alfarhan
// SPDX-License-Identifier: GPL-3.0-only

/// LCG64 — 64-bit linear congruential generator (MMIX variant).
///
/// Period: 2^64. Statistical quality is sufficient for quantum
/// measurement sampling where we only need ~10^4 samples per simulation.
#[derive(Clone, Copy, Debug)]
pub struct Lcg64(pub u64);

impl Lcg64 {
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Return a uniform f64 in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_bounds() {
        let mut rng = Lcg64::new(42);
        for _ in 0..100_000 {
            let v = rng.next_f64();
            assert!(v >= 0.0 && v < 1.0);
        }
    }

    #[test]
    fn test_rng_reasonable_mean() {
        let mut rng = Lcg64::new(12345);
        let mut sum = 0.0f64;
        let n = 100_000;
        for _ in 0..n {
            sum += rng.next_f64();
        }
        let mean = sum / n as f64;
        // Mean should be ~0.5 within statistical error
        assert!((mean - 0.5).abs() < 0.01);
    }
}
