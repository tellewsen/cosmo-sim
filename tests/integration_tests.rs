//! Integration tests for cosmo_sim.
//!
//! Physics components tested end-to-end:
//! - CIC ↔ Poisson self-consistency
//! - Two-body gravitational attraction + Newton 3rd law
//! - Symmetron screening in a top-hat sphere
//! - Energy / momentum conservation over PM steps
//! - Friedmann integration, growth rate

use cosmo_sim::{
    gravity::poisson::{grad_phi, solve_poisson_mg},
    particles::particles::Particles,
    scalar_field::symmetron::{solve_symmetron, source_chi, SymmetronParams},
    utils::{
        cosmology::Cosmology,
        types::{idx3p, Real},
    },
};
use std::f64::consts::PI;

// ─── helpers ──────────────────────────────────────────────────────────────────

fn max_abs(v: &[Real]) -> Real {
    v.iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
}

fn make_grid_particles(ng: usize, l: Real) -> Particles {
    let dx = l / ng as Real;
    let mut p = Particles::new(ng * ng * ng, 1.0, l);
    let mut idx = 0;
    for i in 0..ng {
        for j in 0..ng {
            for k in 0..ng {
                p.pos[idx] = [
                    (i as Real + 0.5) * dx,
                    (j as Real + 0.5) * dx,
                    (k as Real + 0.5) * dx,
                ];
                idx += 1;
            }
        }
    }
    p
}

fn total_momentum(p: &Particles) -> [Real; 3] {
    let mut m = [0.0_f64; 3];
    for vi in &p.vel {
        for d in 0..3 {
            m[d] += vi[d];
        }
    }
    m
}

// ─── CIC ↔ Poisson ────────────────────────────────────────────────────────────

#[test]
fn cic_poisson_uniform_density() {
    let ng = 16_usize;
    let l = 16.0_f64;
    let dx = l / ng as Real;
    let p = make_grid_particles(ng, l);
    let rho = p.cic_density(ng);
    let max_err = rho.iter().map(|r| (r - 1.0).abs()).fold(0.0_f64, f64::max);
    assert!(max_err < 1e-10, "Uniform rho: max_err={max_err:.3e}");
    let delta: Vec<Real> = rho.iter().map(|r| r - 1.0).collect();
    let phi = solve_poisson_mg(&delta, dx, ng, 1e-10, 300, 3);
    let max_phi = max_abs(&phi);
    assert!(max_phi < 1e-7, "Uniform Φ: max|Φ|={max_phi:.3e}");
}

// ─── Two-body gravity ─────────────────────────────────────────────────────────

#[test]
fn two_particle_mutual_attraction() {
    let ng = 32_usize;
    let l = 32.0_f64;
    let dx = l / ng as Real;
    let mut p = Particles::new(2, 1.0, l);
    // Offset slightly from cell centres so CIC deposits mass on neighbours
    let off = 0.3 * dx;
    p.pos[0] = [l * 0.25 + off, l * 0.5, l * 0.5];
    p.pos[1] = [l * 0.75 - off, l * 0.5, l * 0.5];

    let rho = p.cic_density(ng);
    let delta: Vec<Real> = rho.iter().map(|r| r - 1.0).collect();
    #[cfg(feature = "fft_pm")]
    let phi = cosmo_sim::gravity::poisson::solve_poisson_fft(&delta, dx, ng);
    #[cfg(not(feature = "fft_pm"))]
    let phi = solve_poisson_mg(&delta, dx, ng, 1e-10, 500, 3);
    let grad = grad_phi(&phi, dx, ng);
    let f = p.cic_interpolate_force(&grad, ng);

    println!("f[0]={:.4?}  f[1]={:.4?}", f[0], f[1]);
    assert!(f[0][0] > 0.0, "p0 should feel +x force, got {}", f[0][0]);
    assert!(f[1][0] < 0.0, "p1 should feel -x force, got {}", f[1][0]);
    // Normalize by the dominant (x) force so transverse ~0 components don't blow up the ratio
    let f_scale = f[0][0].abs().max(1e-10);
    for d in 0..3 {
        let rel = (f[0][d] + f[1][d]).abs() / f_scale;
        assert!(rel < 0.01, "Newton 3rd law dim {d}: rel={rel:.3e}");
    }
}

#[test]
fn two_body_com_conserved() {
    let ng = 32_usize;
    let l = 64.0_f64;
    let dx = l / ng as Real;
    let cosmo = Cosmology::default();
    let mut p = Particles::new(2, 1.0, l);
    p.pos[0] = [l * 0.4, l * 0.5, l * 0.5];
    p.pos[1] = [l * 0.6, l * 0.5, l * 0.5];
    p.vel[0] = [0.0, 2.0, 0.0];
    p.vel[1] = [0.0, -2.0, 0.0];
    let com0 = p.centre_of_mass();
    let mut a_cur = 1.0_f64;
    let dtau = 0.001_f64;

    for _ in 0..100 {
        let rho = p.cic_density(ng);
        let delta: Vec<Real> = rho.iter().map(|r| r - 1.0).collect();
        let phi = solve_poisson_mg(&delta, dx, ng, 1e-8, 100, 2);
        let grad = grad_phi(&phi, dx, ng);
        p.drift(dtau * 0.5, a_cur);
        let f = p.cic_interpolate_force(&grad, ng);
        p.kick(&f, None, dtau, a_cur);
        p.drift(dtau * 0.5, a_cur);
        a_cur = cosmo.advance_a(a_cur, dtau);
    }
    let com1 = p.centre_of_mass();
    for d in 0..3 {
        assert!(
            (com1[d] - com0[d]).abs() < 0.05,
            "CoM drift dim {d}: {:.4} → {:.4}",
            com0[d],
            com1[d]
        );
    }
}

// ─── Poisson accuracy ─────────────────────────────────────────────────────────

#[test]
fn poisson_gauss_law() {
    let n = 32_usize;
    let l = 1.0_f64;
    let dx = l / n as Real;
    let k = 2.0 * PI;
    let mut src = vec![0.0_f64; n * n * n];
    let mut exact = vec![0.0_f64; n * n * n];
    for i in 0..n {
        let x = (i as Real + 0.5) * dx;
        for j in 0..n {
            for kk in 0..n {
                let c = idx3p(i as isize, j as isize, kk as isize, n);
                src[c] = -k * k * (k * x).cos();
                exact[c] = (k * x).cos();
            }
        }
    }
    let phi = solve_poisson_mg(&src, dx, n, 1e-9, 300, 2);
    let dc_p = phi.iter().sum::<f64>() / phi.len() as f64;
    let dc_e = exact.iter().sum::<f64>() / exact.len() as f64;
    let err = phi
        .iter()
        .zip(exact.iter())
        .map(|(p, e)| ((p - dc_p) - (e - dc_e)).abs())
        .fold(0.0_f64, f64::max);
    assert!(err < 5e-3, "Gauss-law error: {err:.3e}");
}

#[cfg(feature = "fft_pm")]
#[test]
fn fft_poisson_accuracy() {
    use cosmo_sim::gravity::poisson::solve_poisson_fft;
    let n = 32_usize;
    let dx = 1.0 / n as Real;
    let k = 2.0 * PI;
    let mut src = vec![0.0_f64; n * n * n];
    let mut exact = vec![0.0_f64; n * n * n];
    for i in 0..n {
        let x = (i as Real + 0.5) * dx;
        for j in 0..n {
            for kk in 0..n {
                let c = idx3p(i as isize, j as isize, kk as isize, n);
                src[c] = -k * k * (k * x).sin();
                exact[c] = (k * x).sin();
            }
        }
    }
    let phi = solve_poisson_fft(&src, dx, n);
    let dc_p = phi.iter().sum::<f64>() / phi.len() as f64;
    let dc_e = exact.iter().sum::<f64>() / exact.len() as f64;
    let err = phi
        .iter()
        .zip(exact.iter())
        .map(|(p, e)| ((p - dc_p) - (e - dc_e)).abs())
        .fold(0.0_f64, f64::max);
    assert!(err < 5e-3, "FFT error: {err:.3e}");
}

// ─── Symmetron physics ────────────────────────────────────────────────────────

#[test]
fn symmetron_tophat_screening() {
    let n = 32_usize;
    let l = 32.0_f64;
    let dx = l / n as Real;
    let a = 0.8_f64;
    let p = SymmetronParams {
        epsilon: 1e-6,
        max_iter: 200,
        ..Default::default()
    };
    let half = n / 2;
    let r_sphere = (n / 8) as isize;
    let rho: Vec<Real> = (0..n * n * n)
        .map(|c| {
            let i = (c / (n * n)) as isize - half as isize;
            let j = ((c / n) % n) as isize - half as isize;
            let k = (c % n) as isize - half as isize;
            if i * i + j * j + k * k < r_sphere * r_sphere {
                100.0
            } else {
                0.0
            }
        })
        .collect();
    let chi = solve_symmetron(&rho, a, dx, n, &p);
    let chi_in_max = (0..n * n * n)
        .filter(|&c| {
            let i = (c / (n * n)) as isize - half as isize;
            let j = ((c / n) % n) as isize - half as isize;
            let k = (c % n) as isize - half as isize;
            i * i + j * j + k * k < (r_sphere / 2) * (r_sphere / 2)
        })
        .map(|c| chi[c].abs())
        .fold(0.0_f64, f64::max);
    let chi_out_min = (0..n * n * n)
        .filter(|&c| {
            let i = (c / (n * n)) as isize - half as isize;
            let j = ((c / n) % n) as isize - half as isize;
            let k = (c % n) as isize - half as isize;
            i * i + j * j + k * k > (3 * r_sphere) * (3 * r_sphere)
        })
        .map(|c| chi[c])
        .fold(0.0_f64, f64::max);
    println!("Tophat: chi_in_max={chi_in_max:.4}, chi_out_min={chi_out_min:.4}");
    assert!(chi_in_max < 0.5, "Not screened inside: {chi_in_max:.4}");
    assert!(chi_out_min > 0.3, "Not broken outside: {chi_out_min:.4}");
}

#[test]
fn symmetron_sine_density_consistency() {
    let n = 16_usize;
    let l = 16.0_f64;
    let dx = l / n as Real;
    let a = 0.8_f64;
    let p = SymmetronParams {
        epsilon: 1e-6,
        max_iter: 100,
        ..Default::default()
    };
    let k = 2.0 * PI / l;
    let rho: Vec<Real> = (0..n * n * n)
        .map(|c| {
            let x = (c / (n * n)) as Real * dx + 0.5 * dx;
            1.0 + 0.5 * (k * x).sin()
        })
        .collect();
    let chi = solve_symmetron(&rho, a, dx, n, &p);
    for (c, &v) in chi.iter().enumerate() {
        assert!(v.abs() <= 1.0 + 1e-8, "χ[{c}]={v:.6} outside [-1,1]");
    }
    let mean_abs: Real = chi.iter().map(|c| c.abs()).sum::<f64>() / chi.len() as f64;
    assert!(mean_abs > 0.1, "Mean |χ| too small: {mean_abs:.4}");
}

#[test]
fn symmetron_source_odd_symmetry() {
    let p = SymmetronParams::default();
    for &a in &[0.5, 0.7, 1.0] {
        for &rho in &[0.0, 0.5, 1.5] {
            for &chi in &[0.1, 0.5, 0.9] {
                let fp = source_chi(chi, rho, a, &p);
                let fn_val = source_chi(-chi, rho, a, &p);
                assert!(
                    (fp + fn_val).abs() < 1e-13,
                    "f not odd: a={a}, ρ={rho}, χ={chi}"
                );
            }
        }
    }
}

// ─── Cosmology ────────────────────────────────────────────────────────────────

#[test]
fn friedmann_integration_age() {
    let cosmo = Cosmology::default();
    let tau = cosmo.tau_of_a(0.001, 1.0);
    assert!(tau > 0.0 && tau.is_finite(), "tau={tau}");
    assert!(tau > 1e-5, "tau suspiciously small: {tau:.3e}");
}

#[test]
fn growth_rate_late_time_lambda() {
    let cosmo = Cosmology::default();
    let f1 = cosmo.growth_rate(1.0);
    assert!(f1 > 0.4 && f1 < 0.75, "f(a=1)={f1:.4}, expected ~0.55");
}

// ─── Full PM integration ──────────────────────────────────────────────────────

#[test]
fn pm_momentum_conservation_10_steps() {
    let ng = 16_usize;
    let l = 16.0_f64;
    let dx = l / ng as Real;
    let cosmo = Cosmology::default();
    let mut p = make_grid_particles(ng / 2, l);
    for i in 0..p.n {
        p.vel[i] = [
            ((i * 7 + 1) as Real * 0.01).sin(),
            ((i * 13 + 2) as Real * 0.01).cos(),
            ((i * 17 + 3) as Real * 0.01).sin(),
        ];
    }
    let mom0 = total_momentum(&p);
    let mut a_cur = 1.0_f64;
    let dtau = 0.005_f64;

    for _ in 0..10 {
        let rho = p.cic_density(ng);
        let delta: Vec<Real> = rho.iter().map(|r| r - 1.0).collect();
        let phi = solve_poisson_mg(&delta, dx, ng, 1e-7, 100, 2);
        let grad = grad_phi(&phi, dx, ng);
        let f = p.cic_interpolate_force(&grad, ng);
        p.drift(dtau * 0.5, a_cur);
        p.kick(&f, None, dtau, a_cur);
        p.drift(dtau * 0.5, a_cur);
        a_cur = cosmo.advance_a(a_cur, dtau);
    }
    let mom1 = total_momentum(&p);
    for xi in &p.pos {
        for d in 0..3 {
            assert!(xi[d] >= 0.0 && xi[d] < l, "x[{d}]={} out of box", xi[d]);
        }
    }
    for d in 0..3 {
        let dm = (mom1[d] - mom0[d]).abs();
        assert!(dm < 1.0, "Δp[{d}]={dm:.3e} too large");
    }
}

// ─── Hydro mini-sim end-to-end ────────────────────────────────────────────────

/// Run a tiny 8³ hydro simulation for 5 timesteps and verify:
/// - no panics
/// - gas mass is conserved (total gas_rho sums stays constant)
/// - all temperatures stay above the 10^4 K floor
/// - all gas densities remain positive
#[cfg(feature = "hydro")]
#[test]
fn hydro_minisim_mass_conservation_and_stability() {
    use cosmo_sim::{
        mpi::MpiContext,
        simulation::Simulation,
        utils::config::{
            Config, CosmoParams, GravityParams, HydroParams, OutputParams, ParticleParams,
            RunParams, ScalarFieldParams,
        },
    };

    let config = Config {
        run: RunParams {
            box_size: 8.0,
            n_grid: 8,
            levelmax: 0,
            refine_thr: 99999,
            a_init: 0.5,
            a_final: 0.52,
            dtau_max: 0.002,
            courant_factor: 0.5,
        },
        cosmology: CosmoParams {
            omega_m: 0.3,
            omega_lambda: 0.7,
            omega_b: 0.05,
            h: 0.67,
            n_s: 0.96,
            sigma8: 0.83,
        },
        gravity: GravityParams {
            epsilon: 1e-4,
            n_mg_iter: 10,
            n_smooth: 2,
        },
        particles: ParticleParams {
            n_part: 4, // 4³ = 64 particles
            m_part: 0.0,
        },
        scalar_field: ScalarFieldParams::default(),
        hydro: HydroParams {
            cooling: true,
            star_formation: false,
            supernova_feedback: false,
            ..HydroParams::default()
        },
        output: OutputParams {
            outdir: "/tmp/rustis_hydro_test".into(),
            n_output: 0,
            a_out: None,
            hdf5: false,
        },
    };

    let mut sim = Simulation::new(config, MpiContext::init());
    sim.generate_ic(42);

    // Gas ICs are initialised inside run() before the time loop. Run to completion.
    sim.run();

    // ── Post-run invariants ──────────────────────────────────────────────────

    // All densities positive (Godunov floor keeps rho > 0)
    for (i, &rho) in sim.scratch.gas_rho.iter().enumerate() {
        assert!(rho > 0.0, "negative gas density at cell {i}: rho={rho:.3e}");
    }

    // Total gas density should be positive and finite
    let mass_total: f64 = sim.scratch.gas_rho.iter().sum();
    assert!(
        mass_total > 0.0 && mass_total.is_finite(),
        "total gas density invalid: {mass_total:.3e}"
    );

    // Mean gas density ≈ omega_b/omega_m × N³ (code units: mean DM density = N³)
    let n3 = 8usize.pow(3);
    let expected_mean = 0.05 / 0.3; // omega_b / omega_m
    let actual_mean = mass_total / n3 as f64;
    let rel_err = (actual_mean - expected_mean).abs() / expected_mean;
    assert!(
        rel_err < 0.01,
        "mean gas density wrong: expected ~{expected_mean:.4}, got {actual_mean:.4}"
    );

    // All temperatures above 10^4 K floor
    let u_floor = cosmo_sim::hydro::ic::u_from_temperature(1.0e4);
    for (i, &u) in sim.scratch.gas_u.iter().enumerate() {
        assert!(
            u >= u_floor * 0.999,
            "temperature below floor at cell {i}: u={u:.3e} < u_floor={u_floor:.3e}"
        );
    }
}
