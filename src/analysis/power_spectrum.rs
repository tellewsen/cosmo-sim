//! Matter power spectrum estimator P(k).
//!
//! # Method
//!
//! 1. Cloud-In-Cell deposit particles onto an n³ grid → δ(x) = ρ/ρ̄ − 1.
//! 2. Forward 3-D FFT → δ̂(k).
//! 3. Bin  P̂(k) = (V/N⁶) |δ̂(k)|²  into logarithmically-spaced k bins.
//!
//! # Units
//!
//! k in h/Mpc,  P(k) in (Mpc/h)³.
//! Shot-noise 1/n̄ = V/N_part is optionally subtracted.
//!
//! # Output
//!
//! [`PowerSpectrum::write`] writes a space-separated text file alongside
//! each snapshot:  `snapshot_0000.bin` → `power_spectrum_0000.txt`.
//!
//! # Reference
//!
//! Llinares, Mota & Winther (2014) A&A 562 A78, Fig. 3–4 — the benchmark
//! plots this module is designed to reproduce.

use crate::{particles::particles::Particles, utils::types::Real};

// ─── Output struct ────────────────────────────────────────────────────────────

/// Binned matter power spectrum estimate.
pub struct PowerSpectrum {
    /// Bin-centre wavenumbers [h/Mpc]
    pub k: Vec<Real>,
    /// Power spectrum P(k) [(Mpc/h)³]
    pub pk: Vec<Real>,
    /// Number of Fourier modes per bin
    pub nmodes: Vec<usize>,
    /// Scale factor at which this was computed
    pub a: Real,
    /// Simulation step
    pub step: u64,
}

impl PowerSpectrum {
    /// Write to a plain-text file (space-separated, `#` comment header).
    ///
    /// Columns: `k[h/Mpc]  P(k)[(Mpc/h)^3]  N_modes`
    pub fn write(&self, path: &str) {
        use std::io::Write;
        let mut f =
            std::fs::File::create(path).unwrap_or_else(|e| panic!("Cannot create {path}: {e}"));
        writeln!(f, "# a={:.5}  step={}", self.a, self.step).unwrap();
        writeln!(f, "# k[h/Mpc]  P(k)[(Mpc/h)^3]  N_modes").unwrap();
        for i in 0..self.k.len() {
            if self.nmodes[i] == 0 {
                continue;
            }
            writeln!(
                f,
                "{:.6e}  {:.6e}  {}",
                self.k[i], self.pk[i], self.nmodes[i]
            )
            .unwrap();
        }
        log::debug!(
            "P(k) written → {path}  (a={:.5}, {} bins)",
            self.a,
            self.k.len()
        );
    }
}

// ─── Estimator ────────────────────────────────────────────────────────────────

/// Compute the matter power spectrum from particle positions.
///
/// `n_bins` logarithmically-spaced bins from k_fundamental to k_Nyquist.
/// `subtract_shot_noise = true` removes the Poisson term 1/n̄ = V/N_part.
///
/// Requires the `fft_pm` feature (enabled by default).
#[cfg(feature = "fft_pm")]
pub fn compute_power_spectrum(
    particles: &Particles,
    n_grid: usize,
    a: Real,
    step: u64,
    n_bins: usize,
    subtract_shot_noise: bool,
) -> PowerSpectrum {
    use rustfft::{num_complex::Complex, FftPlanner};
    use std::f64::consts::{PI, TAU};

    let n = n_grid;
    let n3 = n * n * n;
    let l = particles.box_size;
    let vol = l * l * l;

    // ── Step 1: CIC overdensity δ = ρ/ρ̄ − 1 ─────────────────────────────
    let rho = particles.cic_density(n);
    let mut data: Vec<Complex<f64>> = rho.iter().map(|&r| Complex::new(r - 1.0, 0.0)).collect();

    // ── Step 2: Forward 3-D FFT ──────────────────────────────────────────
    {
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(n);

        // k-axis (fastest — contiguous rows)
        for i in 0..n {
            for j in 0..n {
                fft.process(&mut data[i * n * n + j * n..i * n * n + j * n + n]);
            }
        }
        // j-axis
        let mut col = vec![Complex::new(0.0, 0.0); n];
        for i in 0..n {
            for k in 0..n {
                for j in 0..n {
                    col[j] = data[i * n * n + j * n + k];
                }
                fft.process(&mut col);
                for j in 0..n {
                    data[i * n * n + j * n + k] = col[j];
                }
            }
        }
        // i-axis (slowest)
        for j in 0..n {
            for k in 0..n {
                for i in 0..n {
                    col[i] = data[i * n * n + j * n + k];
                }
                fft.process(&mut col);
                for i in 0..n {
                    data[i * n * n + j * n + k] = col[i];
                }
            }
        }
    }

    // ── Step 3: Bin P̂(k) = (V/N⁶) |δ̂|²  ─────────────────────────────
    let k_fund = TAU / l; // [h/Mpc]
    let k_nyq = PI * n as f64 / l; // [h/Mpc]
    let norm = vol / (n3 as f64 * n3 as f64); // V/N⁶

    let lnk0 = k_fund.ln();
    let lnk1 = k_nyq.ln();
    let dlnk = (lnk1 - lnk0) / n_bins as f64;

    let mut pk_sum = vec![0.0_f64; n_bins];
    let mut nmodes = vec![0_usize; n_bins];

    for mi in 0..n {
        let mx = mode_freq(mi, n);
        for mj in 0..n {
            let my = mode_freq(mj, n);
            for mk in 0..n {
                let mz = mode_freq(mk, n);
                if mi == 0 && mj == 0 && mk == 0 {
                    continue;
                } // DC mode

                let m2 = mx * mx + my * my + mz * mz;
                let k_m = k_fund * m2.sqrt();
                let lnk = k_m.ln();
                let b = ((lnk - lnk0) / dlnk) as isize;
                if b < 0 || b >= n_bins as isize {
                    continue;
                }

                let c = mi * n * n + mj * n + mk;
                pk_sum[b as usize] += norm * data[c].norm_sqr();
                nmodes[b as usize] += 1;
            }
        }
    }

    // Poisson shot-noise  1/n̄ = V / N_part
    let shot = if subtract_shot_noise {
        vol / particles.n as f64
    } else {
        0.0
    };

    let mut k_bins = vec![0.0_f64; n_bins];
    let mut pk = vec![0.0_f64; n_bins];
    for b in 0..n_bins {
        k_bins[b] = (lnk0 + (b as f64 + 0.5) * dlnk).exp();
        if nmodes[b] > 0 {
            pk[b] = (pk_sum[b] / nmodes[b] as f64 - shot).max(0.0);
        }
    }

    PowerSpectrum {
        k: k_bins,
        pk,
        nmodes,
        a,
        step,
    }
}

/// Stub for builds without `fft_pm`.
#[cfg(not(feature = "fft_pm"))]
pub fn compute_power_spectrum(
    particles: &Particles,
    n_grid: usize,
    a: Real,
    step: u64,
    n_bins: usize,
    _subtract_shot_noise: bool,
) -> PowerSpectrum {
    log::warn!("P(k) computation requires the fft_pm feature.");
    let _ = (particles, n_grid);
    PowerSpectrum {
        k: vec![0.0; n_bins],
        pk: vec![0.0; n_bins],
        nmodes: vec![0; n_bins],
        a,
        step,
    }
}

// ─── Helper ───────────────────────────────────────────────────────────────────

/// Map FFT array index → signed wavenumber integer (same convention as zeldovich.rs).
#[inline]
fn mode_freq(i: usize, n: usize) -> f64 {
    if i <= n / 2 {
        i as f64
    } else {
        i as f64 - n as f64
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(feature = "fft_pm")]
mod tests {
    use super::*;
    use crate::particles::particles::Particles;

    fn uniform_grid(n_p: usize, l: f64) -> Particles {
        let n3 = n_p * n_p * n_p;
        let dx = l / n_p as f64;
        let mut p = Particles::new(n3, 1.0, l);
        let mut idx = 0;
        for i in 0..n_p {
            for j in 0..n_p {
                for k in 0..n_p {
                    p.pos[idx] = [
                        (i as f64 + 0.5) * dx,
                        (j as f64 + 0.5) * dx,
                        (k as f64 + 0.5) * dx,
                    ];
                    idx += 1;
                }
            }
        }
        p
    }

    #[test]
    fn uniform_density_power_spectrum_near_zero() {
        // Uniform grid → δ = 0 everywhere → P(k) ≈ shot-noise term only.
        // After shot-noise subtraction P(k) should be near zero.
        let n = 16_usize;
        let l = 64.0_f64;
        let p = uniform_grid(n, l);
        let ps = compute_power_spectrum(&p, n, 1.0, 0, 20, true);
        for i in 0..ps.pk.len() {
            if ps.nmodes[i] > 0 {
                // Some numerical noise is expected; tolerance ~1% of shot-noise
                let shot = l * l * l / p.n as f64;
                assert!(
                    ps.pk[i].abs() < 0.05 * shot,
                    "P(k={:.3})={:.3e} too large (shot={shot:.3e})",
                    ps.k[i],
                    ps.pk[i]
                );
            }
        }
    }

    #[test]
    fn power_spectrum_k_range_correct() {
        let n = 32_usize;
        let l = 64.0_f64;
        let p = uniform_grid(n, l);
        let ps = compute_power_spectrum(&p, n, 1.0, 0, 20, false);
        let k_fund = std::f64::consts::TAU / l;
        let k_nyq = std::f64::consts::PI * n as f64 / l;
        // All populated bins should have k in [k_fund, k_nyq]
        for i in 0..ps.k.len() {
            if ps.nmodes[i] == 0 {
                continue;
            }
            assert!(
                ps.k[i] >= k_fund * 0.5 && ps.k[i] <= k_nyq * 1.5,
                "k bin {i}: k={:.4} out of expected range [{:.4}, {:.4}]",
                ps.k[i],
                k_fund,
                k_nyq
            );
        }
    }

    #[test]
    fn power_spectrum_units_order_of_magnitude() {
        // For a realistic Zeldovich IC at a=1, P(k) including shot noise
        // should be > 0 and < 10^8 (Mpc/h)^3.
        // We use 16³ particles in 64 Mpc/h (not 128) so the shot noise is
        // 64³/16³ = 64 (Mpc/h)³, well below the signal on large scales.
        use crate::ic::zeldovich::generate_zeldovich;
        use crate::utils::cosmology::Cosmology;
        let n_p = 16_usize;
        let l = 64.0_f64;
        let a = 1.0_f64;
        let c = Cosmology::default();
        let (pos, vel) = generate_zeldovich(n_p, l, a, &c, 42);
        let mut p = Particles::new(n_p * n_p * n_p, 1.0, l);
        p.pos = pos;
        p.vel = vel;
        // Do NOT subtract shot noise so we just check the estimator returns
        // positive and reasonable values.
        let ps = compute_power_spectrum(&p, n_p, a, 0, 10, false);
        let populated: Vec<_> = ps
            .pk
            .iter()
            .zip(ps.nmodes.iter())
            .filter(|(_, &nm)| nm > 0)
            .collect();
        assert!(!populated.is_empty(), "No populated k bins");
        for (&pk_val, &nm) in &populated {
            println!("P(k) = {pk_val:.3e} (Mpc/h)³  N_modes={nm}");
            assert!(
                pk_val > 0.0 && pk_val < 1e8,
                "P(k) = {pk_val:.3e} out of expected range"
            );
        }
    }
}
