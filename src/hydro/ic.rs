//! Gas initial conditions: initialise gas from the DM density and velocity fields.
//!
//! Gas density: ρ_gas = (Ω_b / Ω_m) × ρ_DM  (same spatial distribution as DM)
//! Gas velocity: zero peculiar velocity at start
//! Gas temperature: T_init = 100 K (cold start; quickly thermalised by shocks)

use crate::utils::types::Real;

/// Specific internal energy corresponding to temperature T [K].
///
/// u = k_B T / (μ m_p)  with μ=0.59 (ionised primordial gas)
/// In code units (km/s)²: k_B/m_p = 0.008314 (km/s)²/K
/// So u = 0.008314 / 0.59 × T  (km/s)²
pub fn u_from_temperature(t_kelvin: Real) -> Real {
    const KB_OVER_MU_MP: Real = 0.008314 / 0.59; // (km/s)²/K
    KB_OVER_MU_MP * t_kelvin
}

/// Temperature from specific internal energy [K].
pub fn temperature_from_u(u: Real) -> Real {
    const KB_OVER_MU_MP: Real = 0.008314 / 0.59;
    u / KB_OVER_MU_MP
}

/// Initialise gas state from the DM density field.
///
/// `rho_dm` — DM density field (scratch.rho, length n³)
/// `omega_b`, `omega_m` — baryon and matter density fractions
/// `t_init_k` — initial gas temperature [K] (use 100 K for cold start)
pub fn init_gas_from_dm(
    rho_dm: &[Real],
    omega_b: Real,
    omega_m: Real,
    t_init_k: Real,
    gas_rho: &mut [Real],
    gas_vel: &mut [[Real; 3]],
    gas_u: &mut [Real],
) {
    let frac = omega_b / omega_m;
    let u0 = u_from_temperature(t_init_k);
    for i in 0..rho_dm.len() {
        gas_rho[i] = (rho_dm[i] * frac).max(1e-30);
        gas_vel[i] = [0.0, 0.0, 0.0];
        gas_u[i] = u0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u_temperature_round_trip() {
        for t in [100.0, 1e4, 1e6, 1e8] {
            let u = u_from_temperature(t);
            let t2 = temperature_from_u(u);
            assert!((t2 - t).abs() / t < 1e-12, "T={t}: round-trip gives {t2}");
        }
    }

    #[test]
    fn init_sets_baryon_fraction() {
        let n = 4;
        let rho_dm = vec![1.0f64; n * n * n];
        let mut rho = vec![0.0f64; n * n * n];
        let mut vel = vec![[0.0f64; 3]; n * n * n];
        let mut u = vec![0.0f64; n * n * n];
        init_gas_from_dm(&rho_dm, 0.05, 0.30, 100.0, &mut rho, &mut vel, &mut u);
        let frac = 0.05 / 0.30;
        for &r in &rho {
            assert!((r - frac).abs() < 1e-12);
        }
    }
}
