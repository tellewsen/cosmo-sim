//! Shared numeric types and grid index helpers.
//!
//! # Indexing convention
//!
//! All 3-D grids use **C-order (row-major)** layout:
//!
//!   flat_index(i, j, k, n) = i * n * n  +  j * n  +  k
//!
//! where `i` is the slowest-varying (x) index and `k` the fastest (z).
//! This matches `numpy.reshape(n, n, n)` and is consistent throughout
//! the codebase (CIC, Poisson, scalar-field multigrid, FFT).
//!
//! The same layout is enforced via the `idx3` function below; all inner
//! loops should iterate `for i { for j { for k { ... } } }`.

/// Floating-point precision.  Change to f32 for speed experiments.
pub type Real = f64;

/// 3-component vector in R³.
pub type Vec3 = [Real; 3];

/// AMR refinement level (0 = domain grid).
pub type Level = u8;

// ── Grid indexing ─────────────────────────────────────────────────────────────

/// Flat 3-D **C-order** index (no bounds check, no wrapping).
///
/// Layout:  `idx3(i, j, k, n) = i*n² + j*n + k`.
#[inline(always)]
pub const fn idx3(i: usize, j: usize, k: usize, n: usize) -> usize {
    i * n * n + j * n + k
}

/// Flat 3-D **C-order** index with **periodic wrapping** for signed offsets.
///
/// Use this in stencils where `i` may be negative or ≥ n.
#[inline(always)]
pub fn idx3p(i: isize, j: isize, k: isize, n: usize) -> usize {
    let n = n as isize;
    let wi = i.rem_euclid(n) as usize;
    let wj = j.rem_euclid(n) as usize;
    let wk = k.rem_euclid(n) as usize;
    wi * (n as usize) * (n as usize) + wj * (n as usize) + wk
}

/// Decompose a flat C-order index back into (i, j, k).
#[inline(always)]
pub fn from_flat(c: usize, n: usize) -> (usize, usize, usize) {
    let i = c / (n * n);
    let j = (c / n) % n;
    let k = c % n;
    (i, j, k)
}

// ── Physical constants ────────────────────────────────────────────────────────

/// Newton's constant in units of  (km/s)² (Mpc/h) / M_sun_h
/// where M_sun_h = M_sun / h  and  Mpc/h = Mpc / h.
///
/// G = 4.301×10⁻³ (km/s)² pc / M_sun
///   = 4.301×10⁻³ × 10⁻⁶ × h (km/s)² Mpc_h / M_sun_h
pub const G_COSMO: Real = 4.301e-3 * 1e-6; // (km/s)² (Mpc/h) / (M_sun/h)

/// Speed of light [km/s].
pub const C_LIGHT: Real = 299_792.458;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idx3_basic() {
        let n = 4_usize;
        assert_eq!(idx3(0, 0, 0, n), 0);
        assert_eq!(idx3(0, 0, 1, n), 1);
        assert_eq!(idx3(0, 1, 0, n), 4);
        assert_eq!(idx3(1, 0, 0, n), 16);
        assert_eq!(idx3(n - 1, n - 1, n - 1, n), n * n * n - 1);
    }

    #[test]
    fn idx3p_wraps() {
        let n = 4_usize;
        // Periodic: idx3p(-1, 0, 0, 4) = idx3(3, 0, 0, 4) = 48
        assert_eq!(idx3p(-1, 0, 0, n), idx3(3, 0, 0, n));
        assert_eq!(idx3p(4, 0, 0, n), idx3(0, 0, 0, n));
        assert_eq!(idx3p(0, 0, -1, n), idx3(0, 0, 3, n));
    }

    #[test]
    fn round_trip_flat() {
        let n = 5_usize;
        for c in 0..n * n * n {
            let (i, j, k) = from_flat(c, n);
            assert_eq!(idx3(i, j, k, n), c);
        }
    }
}
