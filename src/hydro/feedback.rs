//! Supernova thermal feedback.
//!
//! After delay t_delay, each star particle deposits thermal energy
//! E_SN = eps_sn × M_star × c² into the surrounding 27 cells
//! with equal weighting.

use crate::simulation::StarParticle;
use crate::utils::types::Real;

/// Speed of light [km/s].
const C_KMS: Real = 299_792.458;

/// Deposit SN energy from mature star particles into the gas internal energy grid.
///
/// A star particle is "mature" when (a_current - star.age_a) × t_hubble_myr > t_delay_myr.
/// Mature particles deposit thermal energy into the surrounding 27 cells and are consumed.
///
/// Energy is distributed equally among the 27 cells and deposited as specific internal
/// energy Δu = E_SN / (27 × M_gas_cell), where M_gas_cell = gas_rho × ρ̄_matter × dx³.
/// This correctly heats less-dense cells by more than denser cells for the same SN energy.
///
/// `t_hubble_myr` ≈ 9778 / h  Myr (Hubble time at z=0; = 9.78/h Gyr).
/// The delay uses Δa rather than proper time for simplicity.
#[allow(clippy::too_many_arguments)]
pub fn supernova_feedback(
    star_particles: &mut Vec<StarParticle>,
    gas_rho: &[Real],
    gas_u: &mut [Real],
    a_current: Real,
    h: Real,
    omega_m: Real,
    eps_sn: Real,
    t_delay_myr: Real,
    n: usize,
    dx: Real,
    box_size: Real,
) {
    // Δa threshold corresponding to t_delay_myr
    // H₀ = 100h km/s/Mpc → t_H0 = 9778/h Myr (= 9.78/h Gyr at z=0)
    let t_hubble_myr = 9_778.0 / h;
    let da_threshold = t_delay_myr / t_hubble_myr;

    // Physical mean matter density [M_sun/h / (Mpc/h)³]
    let rho_bar = 2.775e11 * omega_m * h * h;
    // Cell volume [Mpc/h]³
    let vol_cell = dx * dx * dx;

    let mut remaining = Vec::new();
    let n2 = n * n;

    for star in star_particles.drain(..) {
        if a_current - star.age_a < da_threshold {
            remaining.push(star);
            continue;
        }

        // Total SN energy [M_sun/h × (km/s)²]
        let e_sn = eps_sn * star.mass * C_KMS * C_KMS;

        // Grid cell containing this star
        let ci = ((star.pos[0] / box_size * n as Real) as usize).min(n - 1);
        let cj = ((star.pos[1] / box_size * n as Real) as usize).min(n - 1);
        let ck = ((star.pos[2] / box_size * n as Real) as usize).min(n - 1);

        // Deposit into 3×3×3 neighbourhood (periodic BCs)
        // Δu_cell = E_SN / (27 × M_gas_cell)  [(km/s)²]
        // M_gas_cell = gas_rho × rho_bar × vol_cell  [M_sun/h]
        for di in 0..3_usize {
            for dj in 0..3_usize {
                for dk in 0..3_usize {
                    let ii = (ci + di + n - 1) % n;
                    let jj = (cj + dj + n - 1) % n;
                    let kk = (ck + dk + n - 1) % n;
                    let idx = ii * n2 + jj * n + kk;
                    let m_gas_cell = gas_rho[idx] * rho_bar * vol_cell;
                    gas_u[idx] += e_sn / (27.0 * m_gas_cell.max(1e-30));
                }
            }
        }
        // Star is consumed — not pushed back to remaining
    }

    *star_particles = remaining;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gas_rho(n: usize) -> Vec<f64> {
        vec![0.167f64; n * n * n] // ≈ omega_b/omega_m ~ uniform
    }

    #[test]
    fn immature_star_not_consumed() {
        let n = 4;
        let mut stars = vec![StarParticle {
            pos: [0.5, 0.5, 0.5],
            vel: [0.0; 3],
            mass: 1e8,
            age_a: 0.9999,
        }];
        let gas_rho = make_gas_rho(n);
        let mut gas_u = vec![1.5f64; n * n * n];
        // a_current = 1.0, Δa = 0.0001 → ~1.5 Myr elapsed << 100 Myr delay → not mature
        supernova_feedback(
            &mut stars, &gas_rho, &mut gas_u, 1.0, 0.67, 0.3, 1e-5, 100.0, n, 1.0, 4.0,
        );
        assert_eq!(stars.len(), 1, "star was consumed before delay");
    }

    #[test]
    fn mature_star_deposits_energy() {
        let n = 8;
        let mut stars = vec![StarParticle {
            pos: [4.0, 4.0, 4.0],
            vel: [0.0; 3],
            mass: 1e8,
            age_a: 0.0,
        }];
        let gas_rho = make_gas_rho(n);
        let u0 = 1.5_f64;
        let mut gas_u = vec![u0; n * n * n];
        // a_current = 1.0, age = 0 → definitely mature (1.0 Myr << Hubble time)
        supernova_feedback(
            &mut stars, &gas_rho, &mut gas_u, 1.0, 0.67, 0.3, 1e-5, 1.0, n, 1.0, 8.0,
        );
        assert!(stars.is_empty(), "mature star not consumed");
        let max_u = gas_u.iter().cloned().fold(u0, f64::max);
        assert!(max_u > u0, "no energy deposited");
    }

    #[test]
    fn denser_cell_receives_less_specific_energy() {
        // A cell with twice the gas density should receive half the specific energy.
        let n = 8;
        let pos_centre = [4.0f64, 4.0, 4.0];
        let run_star = |rho_val: f64| -> f64 {
            let mut stars = vec![StarParticle {
                pos: pos_centre,
                vel: [0.0; 3],
                mass: 1e8,
                age_a: 0.0,
            }];
            let mut gas_rho = vec![rho_val; n * n * n];
            let u0 = 1.5_f64;
            let mut gas_u = vec![u0; n * n * n];
            supernova_feedback(
                &mut stars,
                &mut gas_rho,
                &mut gas_u,
                1.0,
                0.67,
                0.3,
                1e-5,
                1.0,
                n,
                1.0,
                8.0,
            );
            // Return the delta-u at the central cell
            let ci = (pos_centre[0] / 8.0 * n as f64) as usize;
            let cj = (pos_centre[1] / 8.0 * n as f64) as usize;
            let ck = (pos_centre[2] / 8.0 * n as f64) as usize;
            gas_u[ci * n * n + cj * n + ck] - u0
        };

        let du_low = run_star(0.1);
        let du_high = run_star(0.2);
        // Doubling rho should halve Δu
        let ratio = du_low / du_high;
        assert!(
            (ratio - 2.0).abs() < 0.01,
            "expected Δu ratio ≈ 2 for 2× density, got {ratio:.4}"
        );
    }
}
