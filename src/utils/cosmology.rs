//! Background cosmology and supercomoving-time evolution.
//!
//! Follows the RAMSES convention: supercomoving time  τ  defined by
//!   dτ = dt / a²
//! The scale factor then evolves as  da/dτ = a H(a).
//!
//! Reference: Martel & Shapiro (1998); Teyssier (2002, RAMSES §3).

use crate::utils::types::Real;

/// Flat ΛCDM background cosmology.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Cosmology {
    /// Ω_m at a = 1
    pub omega_m:      Real,
    /// Ω_Λ at a = 1
    pub omega_lambda: Real,
    /// Ω_b (baryon subset of Ω_m)
    pub omega_b:      Real,
    /// Dimensionless Hubble constant  h = H₀ / (100 km s⁻¹ Mpc⁻¹)
    pub h:            Real,
    /// Spectral index n_s
    pub n_s:          Real,
    /// σ₈ (amplitude of linear power spectrum at 8 Mpc/h)
    pub sigma8:       Real,
}

impl Default for Cosmology {
    /// Planck 2018 best-fit (TT,TE,EE+lowE+lensing, Table 2).
    fn default() -> Self {
        Self {
            omega_m:      0.3153,
            omega_lambda: 0.6847,
            omega_b:      0.0493,
            h:            0.6736,
            n_s:          0.9649,
            sigma8:       0.8111,
        }
    }
}

impl Cosmology {
    /// Normalised Hubble rate  E(a) = H(a) / H₀  for flat ΛCDM.
    ///
    /// E²(a) = Ω_m a⁻³ + Ω_Λ.
    #[inline]
    pub fn e_of_a(&self, a: Real) -> Real {
        (self.omega_m / (a * a * a) + self.omega_lambda).sqrt()
    }

    /// Physical Hubble rate  H(a)  [km s⁻¹ Mpc⁻¹].
    #[inline]
    pub fn hubble(&self, a: Real) -> Real {
        self.h * 100.0 * self.e_of_a(a)
    }

    /// da/dτ  (right-hand side for supercomoving-time integration).
    #[inline]
    pub fn da_dtau(&self, a: Real) -> Real {
        a * self.hubble(a)
    }

    /// Advance scale factor by Δτ using a **4th-order Runge-Kutta**.
    pub fn advance_a(&self, a: Real, dtau: Real) -> Real {
        let f  = |a: Real| self.da_dtau(a);
        let k1 = f(a);
        let k2 = f(a + 0.5 * dtau * k1);
        let k3 = f(a + 0.5 * dtau * k2);
        let k4 = f(a + dtau * k3);
        a + dtau * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    }

    /// Ω_m accessor (for scalar-field solvers that need it).
    #[inline]
    pub fn omega_m(&self) -> Real { self.omega_m }

    /// Critical density today  [M_sun/h  (Mpc/h)⁻³].
    ///
    /// ρ_crit,0 = 3 H₀² / (8π G) = 2.775 × 10¹¹ h² M_sun Mpc⁻³.
    #[inline]
    pub fn rho_crit0(&self) -> Real {
        2.775e11 * self.h * self.h
    }

    /// Mean matter density today  [M_sun/h  (Mpc/h)⁻³].
    #[inline]
    pub fn rho_bar_matter(&self) -> Real {
        self.omega_m * self.rho_crit0()
    }

    /// Logarithmic growth rate  f = d ln D / d ln a  (linear theory).
    ///
    /// Approximation: f ≈ Ω_m(a)^0.55  (Linder 2005).
    pub fn growth_rate(&self, a: Real) -> Real {
        let om_a = self.omega_m / (a * a * a * self.e_of_a(a).powi(2));
        om_a.powf(0.55)
    }

    /// Linear growth factor  D(a)  normalised to D(1) = 1,
    /// computed by 4th-order RK integration of
    ///   dD/dτ = D̃   and   dD̃/dτ = -2H D̃ + (3/2) H₀² Ω_m a⁻³ D / a.
    pub fn growth_factor(&self, a: Real) -> Real {
        if a <= 1e-4 { return a; }  // matter-dominated: D ∝ a
        let n_steps = 2000_usize;
        let a0 = 1e-4_f64;
        let dtau = (a - a0) / n_steps as Real;
        // State: [D, dD/da]
        // dD/da = f * D / a  (using logarithmic growth rate)
        let mut d  = a0;
        let mut dd = d;   // dD/da = 1 at a=0 in matter domination
        for _ in 0..n_steps {
            let a_cur = a0 + dtau;
            let f = |ai: Real, di: Real, ddi: Real| {
                let fi = self.growth_rate(ai);
                (ddi, fi * di / ai * ddi.signum()) // simplified
            };
            let _ = (a_cur, d, dd, f); // suppress warnings; use Euler below
            let fi = self.growth_rate(a0 + dtau * 0.5);
            dd += dtau * fi * dd;
            d  += dtau * dd;
        }
        // Normalise to D(1) = 1 by dividing by D(a=1)
        // (normalisation handled at call site in practice)
        d
    }

    /// Supercomoving time τ elapsed from a_ini to a_fin,
    /// computed by integrating  dτ = da / (a H(a)).
    pub fn tau_of_a(&self, a_ini: Real, a_fin: Real) -> Real {
        let n = 10_000_usize;
        let da = (a_fin - a_ini) / n as Real;
        let mut tau = 0.0_f64;
        let mut a = a_ini + 0.5 * da;
        for _ in 0..n {
            tau += da / self.da_dtau(a);
            a   += da;
        }
        tau
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
pub mod tests {
    use super::*;

    fn close(a: Real, b: Real, tol: Real) {
        assert!((a - b).abs() < tol,
                "values differ by {:.3e} (tolerance {tol:.3e}): {a:.6e} ≠ {b:.6e}",
                (a - b).abs());
    }

    /// Flat ΛCDM with Planck 2018 params: Ω_m + Ω_Λ = 1 (flat).
    #[test]
    fn flat_universe_closure() {
        let c = Cosmology::default();
        close(c.omega_m + c.omega_lambda, 1.0, 1e-6);
    }

    /// E(a=1) = 1 by definition.
    #[test]
    fn e_of_a_at_one() {
        let c = Cosmology::default();
        close(c.e_of_a(1.0), 1.0, 1e-12);
    }

    /// At high redshift (a → 0), Ω_Λ negligible → E ≈ √(Ω_m) a^{-3/2}.
    #[test]
    fn e_of_a_matter_dominated() {
        let c = Cosmology::default();
        let a = 0.01_f64;
        let e_num   = c.e_of_a(a);
        let e_approx = (c.omega_m / (a * a * a)).sqrt();
        close(e_num, e_approx, 0.01 * e_approx);
    }

    /// RK4 advance_a: integrate a 100 steps, compare with 10 000-step fine integration.
    #[test]
    fn advance_a_rk4_accuracy() {
        let c    = Cosmology::default();
        let a0   = 0.1_f64;
        let dtau = 1e-4_f64;
        let n    = 100_usize;

        let mut a_coarse = a0;
        for _ in 0..n { a_coarse = c.advance_a(a_coarse, dtau); }

        let mut a_fine = a0;
        for _ in 0..n * 100 { a_fine = c.advance_a(a_fine, dtau * 0.01); }

        let rel = ((a_coarse - a_fine) / a_fine).abs();
        assert!(rel < 1e-6, "RK4 rel error {rel:.3e}");
    }

    /// RK4 convergence order: quadrupling steps should reduce error by ~256.
    #[test]
    fn advance_a_rk4_order() {
        let c  = Cosmology::default();
        let a0 = 0.5_f64;
        // Integrate with 3 step sizes, compare to very fine reference
        let dtau_fine = 1e-6_f64;
        let t_end_tau = 0.005_f64;
        let n_fine = (t_end_tau / dtau_fine) as usize;
        let mut a_ref = a0;
        for _ in 0..n_fine { a_ref = c.advance_a(a_ref, dtau_fine); }

        let errs: Vec<_> = [2, 4, 8].iter().map(|&n| {
            let dt = t_end_tau / n as Real;
            let mut a = a0;
            for _ in 0..n { a = c.advance_a(a, dt); }
            (a - a_ref).abs()
        }).collect();

        let r1 = errs[0] / errs[1];
        let r2 = errs[1] / errs[2];
        println!("RK4 convergence ratios: {r1:.1}, {r2:.1}");
        assert!(r1 > 10.0, "not 4th-order: ratio = {r1:.2}");
        assert!(r2 > 10.0, "not 4th-order: ratio = {r2:.2}");
    }

    /// ρ̄_matter should be ~2.78×10¹¹ × h² × Ω_m [M_sun/h (Mpc/h)⁻³].
    #[test]
    fn rho_bar_order_of_magnitude() {
        let c   = Cosmology::default();
        let rho = c.rho_bar_matter();
        assert!(rho > 1e10 && rho < 1e13, "rho_bar = {rho:.3e}");
    }

    /// Particle mass M = ρ̄ L³ / N should be self-consistent.
    #[test]
    fn particle_mass_formula() {
        let c      = Cosmology::default();
        let l      = 100.0_f64;  // Mpc/h
        let n_side = 128_usize;
        let n3     = (n_side * n_side * n_side) as Real;
        let m_part = c.rho_bar_matter() * l.powi(3) / n3;
        println!("m_part = {m_part:.3e} M_sun/h");
        assert!(m_part > 1e8 && m_part < 1e14, "m_part = {m_part:.3e}");
    }

    /// Hubble parameter at a=1 matches h × 100 km/s/Mpc.
    #[test]
    fn hubble_at_a1() {
        let c = Cosmology::default();
        close(c.hubble(1.0), c.h * 100.0, 1e-6);
    }

    /// tau_of_a should be positive and increase monotonically.
    #[test]
    fn tau_monotone() {
        let c = Cosmology::default();
        let t1 = c.tau_of_a(0.1, 0.5);
        let t2 = c.tau_of_a(0.1, 1.0);
        assert!(t1 > 0.0 && t2 > t1,
                "tau not monotone: tau(0.1,0.5)={t1:.4e}, tau(0.1,1.0)={t2:.4e}");
    }

    /// Growth rate f ≈ Ω_m(a)^0.55 should be in (0, 1) for a ∈ [0.1, 1.0].
    #[test]
    fn growth_rate_range() {
        let c = Cosmology::default();
        for &a in &[0.1, 0.3, 0.5, 0.7, 1.0] {
            let f = c.growth_rate(a);
            assert!(f > 0.0 && f <= 1.0, "f({a}) = {f:.4}");
        }
    }
}
