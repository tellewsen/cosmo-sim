//! Zel'dovich approximation (1LPT) initial condition generator.
//!
//! Generates cosmological N-body initial conditions by:
//!
//! 1. Placing N_p³ particles on a regular Lagrangian grid **q**.
//! 2. Generating a Gaussian random overdensity field δ(**k**) with variance P(k)/V.
//! 3. Computing the displacement field **Ψ**(**k**) = −i**k**/k² · δ(**k**).
//! 4. Displacing:  **x** = **q** + D(a) **Ψ**(**q**).
//! 5. Setting velocity:  **ṽ** = a² f(a) H(a) D(a) **Ψ**(**q**).
//!
//! # Power spectrum
//!
//! Uses the BBKS transfer function with the Sugiyama (1995) shape parameter:
//!   P(k) = A_s · k^{n_s} · T²(k)
//! where A_s is chosen to normalise σ₈ to the target value.
//!
//! # Velocity convention
//!
//! Matches the simulation's supercomoving DKD leapfrog:
//!   **ṽ** ≡ a² d**x**/dt  →  drift: Δ**x** = **ṽ** Δt / a²
//!
//! # References
//!
//! - Bardeen, Bond, Kaiser & Szalay (1986) *ApJ* 304, 15 [BBKS transfer function]
//! - Sugiyama (1995) *ApJS* 100, 281 [shape parameter correction]
//! - Zel'dovich (1970) *A&A* 5, 84

use crate::utils::{cosmology::Cosmology, types::Real};

// ─── Pseudo-random number generator (xorshift64 + Box-Muller) ─────────────────

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self { Rng(seed | 1) }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in [0, 1).
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Pair of independent standard-normal variates (Box-Muller).
    fn gauss_pair(&mut self) -> (f64, f64) {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        let r  = (-2.0 * u1.ln()).sqrt();
        let (s, c) = (std::f64::consts::TAU * u2).sin_cos();
        (r * c, r * s)
    }
}

// ─── Transfer function and power spectrum ─────────────────────────────────────

/// BBKS transfer function (Bardeen et al. 1986, Eq. G3).
///
/// `k` in h/Mpc, `gamma` is the Sugiyama shape parameter in h/Mpc.
fn bbks_transfer(k: f64, gamma: f64) -> f64 {
    let q = k / gamma;
    if q < 1e-6 { return 1.0; }
    let denom = 1.0
        + 3.89  * q
        + (16.1 * q).powi(2)
        + (5.46 * q).powi(3)
        + (6.71 * q).powi(4);
    (1.0 + 2.34 * q).ln() / (2.34 * q * denom.powf(0.25))
}

/// Unnormalized matter power spectrum P₀(k) ∝ k^{n_s} T²(k)  [(Mpc/h)³ up to A_s].
fn power_unnorm(k: f64, n_s: f64, gamma: f64) -> f64 {
    if k < 1e-10 { return 0.0; }
    let t = bbks_transfer(k, gamma);
    k.powf(n_s) * t * t
}

/// Top-hat window function W(x) = 3[sin x − x cos x] / x³  (Taylor-expanded near 0).
fn tophat(x: f64) -> f64 {
    if x < 1e-3 { return 1.0 - x * x / 10.0 + x * x * x * x / 280.0; }
    3.0 * (x.sin() - x * x.cos()) / (x * x * x)
}

/// Compute the overall amplitude A_s such that σ₈(A_s · P₀) = σ₈_target.
///
/// σ₈² = 1/(2π²) ∫ P(k) k² W(kR)² dk  with R = 8 Mpc/h.
fn amplitude(sigma8_target: f64, n_s: f64, gamma: f64) -> f64 {
    use std::f64::consts::PI;
    let r    = 8.0_f64;          // 8 Mpc/h
    let n    = 10_000_usize;
    let lnk0 = (1e-4_f64).ln();
    let lnk1 = (1e2_f64 ).ln();
    let dlnk = (lnk1 - lnk0) / n as f64;
    let mut sig8sq = 0.0_f64;
    for i in 0..n {
        let k  = (lnk0 + (i as f64 + 0.5) * dlnk).exp();
        let w  = tophat(k * r);
        // integrand = P(k) k³ W² / (2π²)  ×  dlnk  (change of variable k → ln k)
        sig8sq += power_unnorm(k, n_s, gamma) * k * k * k * w * w * dlnk;
    }
    sig8sq /= 2.0 * PI * PI;
    sigma8_target * sigma8_target / sig8sq
}

// ─── 3-D FFT helper ───────────────────────────────────────────────────────────

#[cfg(feature = "fft_pm")]
fn fft3d(data: &mut [rustfft::num_complex::Complex<f64>], n: usize, inverse: bool) {
    use rustfft::{FftPlanner, num_complex::Complex};
    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(n)
    } else {
        planner.plan_fft_forward(n)
    };

    // k-axis (fastest, contiguous in memory)
    for i in 0..n {
        for j in 0..n {
            fft.process(&mut data[i * n * n + j * n .. i * n * n + j * n + n]);
        }
    }

    // j-axis
    {
        let mut col = vec![Complex::new(0.0, 0.0); n];
        for i in 0..n {
            for k in 0..n {
                for j in 0..n { col[j] = data[i * n * n + j * n + k]; }
                fft.process(&mut col);
                for j in 0..n { data[i * n * n + j * n + k] = col[j]; }
            }
        }
    }

    // i-axis (slowest)
    {
        let mut col = vec![Complex::new(0.0, 0.0); n];
        for j in 0..n {
            for k in 0..n {
                for i in 0..n { col[i] = data[i * n * n + j * n + k]; }
                fft.process(&mut col);
                for i in 0..n { data[i * n * n + j * n + k] = col[i]; }
            }
        }
    }
}

// ─── Public interface ─────────────────────────────────────────────────────────

/// Generate Zel'dovich approximation initial conditions for `n_p`³ particles.
///
/// Returns `(positions, velocities)` where:
/// - positions are comoving Mpc/h, periodic in `[0, box_size)`.
/// - velocities are in supercomoving units **ṽ** = a²·km·s⁻¹·h⁻¹.
///
/// When the `fft_pm` feature is disabled, falls back to a uniform grid with
/// zero velocities (useful for unit tests without the FFT dependency).
pub fn generate_zeldovich(
    n_p:      usize,
    box_size: Real,
    a_init:   Real,
    cosmo:    &Cosmology,
    seed:     u64,
) -> (Vec<[Real; 3]>, Vec<[Real; 3]>) {
    #[cfg(feature = "fft_pm")]
    {
        _gen_fft(n_p, box_size, a_init, cosmo, seed)
    }
    #[cfg(not(feature = "fft_pm"))]
    {
        log::warn!("fft_pm feature disabled — IC generator uses uniform grid with zero velocities");
        let n3  = n_p * n_p * n_p;
        let dx  = box_size / n_p as Real;
        let mut pos = vec![[0.0_f64; 3]; n3];
        let vel = vec![[0.0_f64; 3]; n3];
        let mut idx = 0;
        for i in 0..n_p { for j in 0..n_p { for k in 0..n_p {
            pos[idx] = [
                (i as Real + 0.5) * dx,
                (j as Real + 0.5) * dx,
                (k as Real + 0.5) * dx,
            ];
            idx += 1;
        }}}
        (pos, vel)
    }
}

// ─── FFT-based implementation ─────────────────────────────────────────────────

#[cfg(feature = "fft_pm")]
fn _gen_fft(
    n_p:  usize,
    l:    Real,
    a:    Real,
    cosmo: &Cosmology,
    seed: u64,
) -> (Vec<[Real; 3]>, Vec<[Real; 3]>) {
    use rustfft::num_complex::Complex;
    use std::f64::consts::TAU;

    let n3    = n_p * n_p * n_p;
    let vol   = l * l * l;
    let k_f   = TAU / l;                // fundamental wavenumber [h/Mpc]

    // Sugiyama (1995) shape parameter Γ
    let gamma = cosmo.omega_m * cosmo.h
        * (-cosmo.omega_b * (1.0 + (2.0 * cosmo.h).sqrt() / cosmo.omega_m)).exp();

    let a_s   = amplitude(cosmo.sigma8, cosmo.n_s, gamma);   // σ₈ normalisation
    let d_a   = cosmo.growth_factor(a);                        // D(a_init)
    let f_a   = cosmo.growth_rate(a);                          // f = d ln D / d ln a
    let h_a   = cosmo.hubble(a);                               // H(a) [km/s/Mpc]

    // Velocity prefactor: ṽ = vel_fac · Ψ   [a²·km·s⁻¹·h⁻¹]
    let vel_fac = a * a * f_a * h_a * d_a;

    // ── Step 1: Gaussian white noise on the Lagrangian grid ──────────────
    let mut rng  = Rng::new(seed);
    let mut buf: Vec<Complex<f64>> = (0..n3)
        .map(|_| { let (g, _) = rng.gauss_pair(); Complex::new(g, 0.0) })
        .collect();

    // ── Step 2: Forward 3-D FFT → buf[k] is Hermitian-symmetric ─────────
    fft3d(&mut buf, n_p, false);
    // After FFT of N³ unit-variance real values: E|buf[k]|² = N³ for k ≠ 0.

    // ── Step 3: Build three displacement-field components in Fourier space
    //
    //   δ̂[k]    = buf[k] × sqrt(P(k) / V)      E|δ̂|² = N³ P(k)/V ✓
    //   Ψ̂_d[k]  = −i m_d · δ̂[k] / (k_f · m²)  where m_d ∈ {mx,my,mz}
    let mut psi: [Vec<Complex<f64>>; 3] = [
        vec![Complex::new(0.0, 0.0); n3],
        vec![Complex::new(0.0, 0.0); n3],
        vec![Complex::new(0.0, 0.0); n3],
    ];

    for mi in 0..n_p {
        let mx = mode_freq(mi, n_p);
        for mj in 0..n_p {
            let my = mode_freq(mj, n_p);
            for mk in 0..n_p {
                let mz = mode_freq(mk, n_p);
                let c  = mi * n_p * n_p + mj * n_p + mk;

                // Zero mode → no bulk displacement
                if mi == 0 && mj == 0 && mk == 0 { continue; }

                let m2  = mx * mx + my * my + mz * mz;
                let k_m = k_f * m2.sqrt();
                let pk  = a_s * power_unnorm(k_m, cosmo.n_s, gamma);

                // δ̂[k] = buf[k] × sqrt(P(k)/V)
                let delta = buf[c] * (pk / vol).sqrt();

                // −i × delta × (m_d / (k_f m²)):
                //   −i (re + i·im) × s  =  (im·s) + i·(−re·s)
                let c1 = 1.0 / (k_f * m2);
                for (dim, md) in [(0usize, mx), (1, my), (2, mz)] {
                    let s = md * c1;
                    psi[dim][c] = Complex::new(delta.im * s, -delta.re * s);
                }
            }
        }
    }

    // ── Step 4: Inverse FFT each displacement component ──────────────────
    let inv_n3 = 1.0 / n3 as f64;
    for dim in 0..3 {
        fft3d(&mut psi[dim], n_p, true);
        for x in psi[dim].iter_mut() { *x *= inv_n3; }
    }
    // psi[dim][j].re  is now the Lagrangian displacement Ψ_dim at grid point j.

    // ── Step 5: Place particles on regular grid and apply displacement ────
    let dx_q = l / n_p as Real;
    let mut pos = vec![[0.0_f64; 3]; n3];
    let mut vel = vec![[0.0_f64; 3]; n3];

    for i in 0..n_p {
        for j in 0..n_p {
            for k in 0..n_p {
                let c = i * n_p * n_p + j * n_p + k;
                let q = [
                    (i as f64 + 0.5) * dx_q,
                    (j as f64 + 0.5) * dx_q,
                    (k as f64 + 0.5) * dx_q,
                ];
                for dim in 0..3 {
                    let psi_val = psi[dim][c].re;
                    pos[c][dim] = (q[dim] + d_a * psi_val).rem_euclid(l);
                    vel[c][dim] = vel_fac * psi_val;
                }
            }
        }
    }

    log::info!(
        "Zel'dovich IC: n_p={}³={}, L={:.1} Mpc/h, a={:.4}, D(a)={:.4e}, f(a)={:.4}, H={:.2} km/s/Mpc",
        n_p, n3, l, a, d_a, f_a, h_a,
    );

    (pos, vel)
}

/// Map FFT array index → signed wavenumber integer.
///
/// For n even: 0..n/2 are positive, n/2+1..n-1 are negative.
/// For n odd:  0..(n-1)/2 positive, (n+1)/2..n-1 negative.
#[inline]
fn mode_freq(i: usize, n: usize) -> f64 {
    if i <= n / 2 { i as f64 } else { i as f64 - n as f64 }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn planck() -> Cosmology {
        Cosmology::default()
    }

    // ── Transfer function ─────────────────────────────────────────────────

    #[test]
    fn bbks_k0_is_one() {
        let c = planck();
        let gamma = c.omega_m * c.h
            * (-c.omega_b * (1.0 + (2.0 * c.h).sqrt() / c.omega_m)).exp();
        assert!((bbks_transfer(1e-8, gamma) - 1.0).abs() < 1e-4,
                "T(k→0) should be 1");
    }

    #[test]
    fn bbks_decays_at_large_k() {
        let c = planck();
        let gamma = c.omega_m * c.h
            * (-c.omega_b * (1.0 + (2.0 * c.h).sqrt() / c.omega_m)).exp();
        let t_small = bbks_transfer(0.01, gamma);
        let t_large = bbks_transfer(10.0, gamma);
        assert!(t_large < t_small, "T(k) should decrease with k");
        assert!(t_large < 0.1, "T(10 h/Mpc) << 1");
    }

    // ── σ₈ normalisation ─────────────────────────────────────────────────

    #[test]
    fn sigma8_normalisation_roundtrip() {
        use std::f64::consts::PI;
        let c     = planck();
        let gamma = c.omega_m * c.h
            * (-c.omega_b * (1.0 + (2.0 * c.h).sqrt() / c.omega_m)).exp();
        let a_s   = amplitude(c.sigma8, c.n_s, gamma);

        // Recompute σ₈ with the returned A_s
        let r    = 8.0_f64;
        let n    = 10_000_usize;
        let lnk0 = (1e-4_f64).ln();
        let lnk1 = (1e2_f64).ln();
        let dlnk = (lnk1 - lnk0) / n as f64;
        let mut sig8sq = 0.0_f64;
        for i in 0..n {
            let k = (lnk0 + (i as f64 + 0.5) * dlnk).exp();
            let w = tophat(k * r);
            sig8sq += a_s * power_unnorm(k, c.n_s, gamma) * k * k * k * w * w * dlnk;
        }
        sig8sq /= 2.0 * PI * PI;
        let sig8 = sig8sq.sqrt();
        let err  = (sig8 - c.sigma8).abs() / c.sigma8;
        assert!(err < 1e-4, "σ₈ roundtrip error = {err:.3e}  (got {sig8:.4}, want {})", c.sigma8);
    }

    // ── IC generation ─────────────────────────────────────────────────────

    #[cfg(feature = "fft_pm")]
    #[test]
    fn ic_particle_count() {
        let c       = planck();
        let n_p     = 8_usize;
        let (pos, vel) = generate_zeldovich(n_p, 64.0, 0.02, &c, 42);
        assert_eq!(pos.len(), n_p * n_p * n_p);
        assert_eq!(vel.len(), n_p * n_p * n_p);
    }

    #[cfg(feature = "fft_pm")]
    #[test]
    fn ic_positions_in_box() {
        let c   = planck();
        let n_p = 8_usize;
        let l   = 64.0_f64;
        let (pos, _) = generate_zeldovich(n_p, l, 0.02, &c, 42);
        for (i, p) in pos.iter().enumerate() {
            for d in 0..3 {
                assert!(p[d] >= 0.0 && p[d] < l,
                    "particle {i} dim {d}: x = {} not in [0, {l})", p[d]);
            }
        }
    }

    #[cfg(feature = "fft_pm")]
    #[test]
    fn ic_centre_of_mass_near_box_centre() {
        // Centre-of-mass should be close to L/2 (periodic box average)
        let c   = planck();
        let n_p = 16_usize;
        let l   = 64.0_f64;
        let (pos, _) = generate_zeldovich(n_p, l, 0.02, &c, 42);
        let n3f = pos.len() as f64;
        let com: [f64; 3] = pos.iter().fold([0.0; 3], |mut acc, p| {
            for d in 0..3 { acc[d] += p[d]; }
            acc
        }).map(|s| s / n3f);
        for d in 0..3 {
            let err = (com[d] - l * 0.5).abs() / l;
            assert!(err < 0.1, "com[{d}] = {:.2} off-centre by {:.1}%", com[d], err * 100.0);
        }
    }

    #[cfg(feature = "fft_pm")]
    #[test]
    fn ic_rms_velocity_reasonable() {
        // At a_init = 0.02 (z≈49), rms peculiar velocity should be small (~0.1-1 km/s)
        let c   = planck();
        let n_p = 16_usize;
        let l   = 64.0_f64;
        let a   = 0.02_f64;
        let (_, vel) = generate_zeldovich(n_p, l, a, &c, 42);
        let n3f  = vel.len() as f64;
        let sum_v2: f64 = vel.iter().map(|v| v[0]*v[0]+v[1]*v[1]+v[2]*v[2]).sum();
        // ṽ = a² v_pec,  v_rms = sqrt(sum ṽ² / N) / a
        let v_rms = (sum_v2 / n3f).sqrt() / (a * a);
        println!("IC rms peculiar velocity at a={a}: {v_rms:.4} km/s/h");
        // Very rough sanity: should be << 10000 and > 0
        assert!(v_rms > 0.0 && v_rms < 1e4,
                "unreasonable v_rms = {v_rms:.3e}");
    }

    #[cfg(feature = "fft_pm")]
    #[test]
    fn ic_different_seeds_give_different_positions() {
        let c = planck();
        let n_p = 4_usize;
        let (p1, _) = generate_zeldovich(n_p, 64.0, 0.02, &c, 1);
        let (p2, _) = generate_zeldovich(n_p, 64.0, 0.02, &c, 2);
        let diff: f64 = p1.iter().zip(p2.iter())
            .map(|(a, b)| (0..3).map(|d| (a[d]-b[d]).abs()).sum::<f64>())
            .sum();
        assert!(diff > 0.0, "different seeds gave identical positions");
    }

    #[cfg(feature = "fft_pm")]
    #[test]
    fn ic_same_seed_is_deterministic() {
        let c = planck();
        let n_p = 4_usize;
        let (p1, v1) = generate_zeldovich(n_p, 64.0, 0.02, &c, 99);
        let (p2, v2) = generate_zeldovich(n_p, 64.0, 0.02, &c, 99);
        for i in 0..p1.len() {
            for d in 0..3 {
                assert_eq!(p1[i][d], p2[i][d], "pos not deterministic at ({i},{d})");
                assert_eq!(v1[i][d], v2[i][d], "vel not deterministic at ({i},{d})");
            }
        }
    }
}
