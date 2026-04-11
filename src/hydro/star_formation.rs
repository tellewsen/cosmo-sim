//! Stochastic star formation following Schmidt (1959) + Katz (1992).
//!
//! Cells denser than ρ_SF and cooler than T_SF spawn star particles with
//! probability p = 1 − exp(−ε_SF × Δt / t_ff).

use crate::utils::types::Real;
use crate::hydro::ic::temperature_from_u;
use crate::simulation::StarParticle;

/// Free-fall time t_ff = √(3π / (32 G ρ_phys)) [seconds].
///
/// In code units: ρ_phys = gas_rho × ρ̄_matter_cgs.
/// G_cgs = 6.674e-8 cm³/(g s²).
fn free_fall_time_s(gas_rho: Real, omega_m: Real, h: Real) -> Real {
    let rho_bar_cgs = 9.47e-30 * omega_m * h * h;
    let rho_cgs = gas_rho * rho_bar_cgs;
    let g_cgs = 6.674e-8_f64;
    (3.0 * std::f64::consts::PI / (32.0 * g_cgs * rho_cgs)).sqrt()
}

/// Convert number density threshold from H/cm³ to code units (ρ_gas / ρ̄_matter).
pub fn rho_sf_code(rho_sf_h_cm3: Real, omega_m: Real, h: Real) -> Real {
    let rho_bar_cgs = 9.47e-30 * omega_m * h * h;
    let m_p_cgs     = 1.673e-24_f64;
    let rho_sf_cgs  = rho_sf_h_cm3 * m_p_cgs;
    rho_sf_cgs / rho_bar_cgs
}

/// Attempt to form stars in every eligible cell.
///
/// Returns the list of newly created star particles.
/// Eligible cells: gas_rho > rho_sf_code AND T < t_sf_thresh.
/// Probability: p = 1 − exp(−eps_sf × Δt_s / t_ff_s).
///
/// Uses a deterministic pseudo-random check seeded by cell index + step
/// to avoid introducing an RNG state.
pub fn form_stars(
    gas_rho: &mut [Real],
    gas_vel: &[[Real; 3]],
    gas_u:   &[Real],
    cell_positions: &[[Real; 3]],
    m_cell: Real,
    rho_sf: Real,
    t_sf:   Real,
    eps_sf: Real,
    dt:     Real,
    omega_m: Real,
    h: Real,
    step_id: u64,
) -> Vec<StarParticle> {
    let tau_to_s = 3.086e19 / h;
    let dt_s = dt * tau_to_s;
    let mut new_stars = Vec::new();

    for i in 0..gas_rho.len() {
        if gas_rho[i] < rho_sf { continue; }
        let t_k = temperature_from_u(gas_u[i]);
        if t_k > t_sf { continue; }

        let t_ff = free_fall_time_s(gas_rho[i], omega_m, h);
        let prob = 1.0 - (-eps_sf * dt_s / t_ff).exp();

        // Deterministic pseudo-random: hash cell index and step
        let hash = ((i as u64).wrapping_mul(2654435761) ^ step_id.wrapping_mul(2246822519)) as f64
                   / u64::MAX as f64;
        if hash < prob {
            let star_mass = gas_rho[i] * m_cell * 0.5;
            gas_rho[i] *= 0.5;
            new_stars.push(StarParticle {
                pos:   cell_positions[i],
                vel:   gas_vel[i],
                mass:  star_mass,
                age_a: 0.0, // filled in by caller with current scale factor
            });
        }
    }
    new_stars
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rho_sf_code_is_large_for_0p1_h_cm3() {
        // 0.1 H/cm³ in code units should be >> mean density (~0.167)
        let rho = rho_sf_code(0.1, 0.3, 0.67);
        assert!(rho > 1.0, "rho_sf_code={rho:.1} — should be overdense");
    }

    #[test]
    fn no_stars_below_threshold() {
        let n = 4;
        let mut rho = vec![0.01f64; n]; // well below SF threshold
        let vel = vec![[0.0f64; 3]; n];
        let u: Vec<f64> = vec![crate::hydro::ic::u_from_temperature(100.0); n];
        let pos = vec![[0.0f64; 3]; n];
        let rho_sf = rho_sf_code(0.1, 0.3, 0.67);
        let stars = form_stars(&mut rho, &vel, &u, &pos, 1e8, rho_sf, 2e4, 0.01, 0.01, 0.3, 0.67, 0);
        assert!(stars.is_empty(), "formed stars below threshold");
    }
}
