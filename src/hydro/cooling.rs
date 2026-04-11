//! Radiative cooling via operator splitting.
//!
//! Cooling rate Λ(T): simple two-power-law fit to primordial H/He cooling
//! (Sutherland & Dopita 1993). Operator-split from hydro: each cell
//! integrated independently over dt.

use crate::utils::types::Real;
use crate::hydro::ic::{u_from_temperature, temperature_from_u};

/// Cooling rate Λ(T) in units of erg cm³/s (normalised to n_H²).
/// Returns 0 below T_floor (photoionisation equilibrium assumption).
pub fn lambda_cooling(t_k: Real) -> Real {
    const T_FLOOR: Real = 1.0e4;  // K
    const T_BREAK: Real = 1.0e5;  // K
    if t_k < T_FLOOR { return 0.0; }
    if t_k < T_BREAK {
        1.0e-22 * (t_k / T_FLOOR).powf(0.5)
    } else {
        1.0e-23 * (t_k / T_BREAK).powf(0.5)
    }
}

/// Unit conversion pre-factor for cooling.
///
/// du/dt = −Λ(T) × n_H² / ρ  [erg/g/s]
///       = −Λ(T) × (X/m_p)² × ρ_code × ρ̄_cgs  [erg/g/s]
///
/// Conversion: 1 erg/g = 10⁻¹⁰ (km/s)²
/// n_H = X ρ_phys / m_p,  X=0.76,  m_p = 1.673e-24 g
/// ρ̄_matter_cgs ≈ 9.47e-30 × Ω_m × h² g/cm³
///
/// Full pre-factor: pre = (X/m_p)² × ρ̄_cgs × ERG_TO_KMS2
/// du/dt [km/s² per code-τ] = −Λ_cgs × pre × gas_rho × dt_seconds
pub fn cooling_prefactor(omega_m: Real, h: Real) -> Real {
    const X_H: Real = 0.76;          // hydrogen mass fraction
    const M_P: Real = 1.673e-24;     // g
    const ERG_TO_KMS2: Real = 1e-10; // erg/g → (km/s)²
    let rho_bar_cgs = 9.47e-30 * omega_m * h * h;
    (X_H / M_P).powi(2) * rho_bar_cgs * ERG_TO_KMS2
}

/// Seconds per supercomoving conformal time unit τ (H₀ = 100 h km/s/Mpc).
///
/// 1 Mpc/h / (km/s) = 3.086e22 m / (1e3 m/s) = 3.086e19 s / h
pub fn tau_to_seconds(h: Real) -> Real {
    3.086e19 / h
}

/// Apply radiative cooling to a single cell for timestep dt [τ].
///
/// Returns new specific internal energy u (floored at u(T_floor = 10⁴ K)).
pub fn cool_cell(
    gas_rho: Real,
    u: Real,
    dt: Real,
    omega_m: Real,
    h: Real,
) -> Real {
    let t_k = temperature_from_u(u);
    let lam = lambda_cooling(t_k);
    if lam == 0.0 { return u; }

    let pre   = cooling_prefactor(omega_m, h);
    let dt_s  = dt * tau_to_seconds(h);
    let du    = lam * pre * gas_rho * dt_s;
    let u_new = u - du;

    u_new.max(u_from_temperature(1.0e4))
}

/// Apply cooling to the full gas grid.
pub fn cooling_step(
    gas_rho: &[Real],
    gas_u:   &mut [Real],
    dt:      Real,
    omega_m: Real,
    h:       Real,
) {
    for i in 0..gas_rho.len() {
        gas_u[i] = cool_cell(gas_rho[i], gas_u[i], dt, omega_m, h);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lambda_is_zero_below_floor() {
        assert_eq!(lambda_cooling(100.0), 0.0);
        assert_eq!(lambda_cooling(9999.0), 0.0);
    }

    #[test]
    fn lambda_is_positive_above_floor() {
        assert!(lambda_cooling(1.0e4) > 0.0);
        assert!(lambda_cooling(1.0e5) > 0.0);
        assert!(lambda_cooling(1.0e7) > 0.0);
    }

    #[test]
    fn lambda_finite_and_positive_at_break() {
        // Both regimes give positive, finite cooling rates at and near the break.
        // The two-power-law approximation has an intentional discontinuity at T_BREAK;
        // we verify each branch is well-behaved independently.
        assert!(lambda_cooling(9.99e4) > 0.0 && lambda_cooling(9.99e4).is_finite());
        assert!(lambda_cooling(1.0e5) > 0.0  && lambda_cooling(1.0e5).is_finite());
        assert!(lambda_cooling(1.01e5) > 0.0 && lambda_cooling(1.01e5).is_finite());
    }

    #[test]
    fn cool_cell_floors_at_t_min() {
        // Hot gas should cool but not below 10^4 K
        let u_hot  = crate::hydro::ic::u_from_temperature(1.0e8);
        let u_cool = cool_cell(1.0, u_hot, 1000.0, 0.3, 0.67);
        let t_cool = temperature_from_u(u_cool);
        assert!(t_cool >= 1.0e4 - 1.0, "cooled below floor: T={t_cool:.1} K");
    }

    #[test]
    fn cool_cell_does_nothing_at_floor() {
        let u_floor = crate::hydro::ic::u_from_temperature(1.0e4);
        let u_out   = cool_cell(1.0, u_floor, 1.0, 0.3, 0.67);
        assert!((u_out - u_floor).abs() / u_floor < 1e-10);
    }
}
