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
/// `t_hubble_myr` ≈ 9778 / h  Myr (Hubble time at z=0; = 9.78/h Gyr).
/// The delay uses Δa rather than proper time for simplicity.
pub fn supernova_feedback(
    star_particles: &mut Vec<StarParticle>,
    gas_u: &mut [Real],
    a_current: Real,
    h: Real,
    eps_sn: Real,
    t_delay_myr: Real,
    n: usize,
    _dx: Real,
    box_size: Real,
) {
    // Δa threshold corresponding to t_delay_myr
    // H₀ = 100h km/s/Mpc → t_H0 = 9778/h Myr (= 9.78/h Gyr at z=0)
    let t_hubble_myr = 9_778.0 / h;
    let da_threshold = t_delay_myr / t_hubble_myr;

    let mut remaining = Vec::new();
    let n2 = n * n;

    for star in star_particles.drain(..) {
        if a_current - star.age_a < da_threshold {
            remaining.push(star);
            continue;
        }

        // Specific SN energy: E_SN / M_star = eps_sn × c²  [(km/s)²]
        // Deposit equally across 27 surrounding cells.
        let e_sn = eps_sn * star.mass * C_KMS * C_KMS;
        let weight = 1.0 / 27.0;

        // Grid cell containing this star
        let ci = ((star.pos[0] / box_size * n as Real) as usize).min(n - 1);
        let cj = ((star.pos[1] / box_size * n as Real) as usize).min(n - 1);
        let ck = ((star.pos[2] / box_size * n as Real) as usize).min(n - 1);

        // Deposit into 3×3×3 neighbourhood (periodic BCs)
        for di in 0..3_usize {
            for dj in 0..3_usize {
                for dk in 0..3_usize {
                    let ii = (ci + di + n - 1) % n;
                    let jj = (cj + dj + n - 1) % n;
                    let kk = (ck + dk + n - 1) % n;
                    let idx = ii * n2 + jj * n + kk;
                    // Δu_specific = E_SN × weight / M_star  (M_cell ≈ M_star for SF cells)
                    gas_u[idx] += e_sn * weight / star.mass.max(1e-30);
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

    #[test]
    fn immature_star_not_consumed() {
        let mut stars = vec![StarParticle {
            pos: [0.5, 0.5, 0.5],
            vel: [0.0; 3],
            mass: 1e8,
            age_a: 0.9999,
        }];
        let mut gas_u = vec![1.5f64; 4 * 4 * 4];
        // a_current = 1.0, Δa = 0.0001 → ~1.5 Myr elapsed << 100 Myr delay → not mature
        supernova_feedback(&mut stars, &mut gas_u, 1.0, 0.67, 1e-5, 100.0, 4, 1.0, 4.0);
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
        let u0 = 1.5_f64;
        let mut gas_u = vec![u0; n * n * n];
        // a_current = 1.0, age = 0 → definitely mature (1.0 Myr << Hubble time)
        supernova_feedback(&mut stars, &mut gas_u, 1.0, 0.67, 1e-5, 1.0, n, 1.0, 8.0);
        assert!(stars.is_empty(), "mature star not consumed");
        let max_u = gas_u.iter().cloned().fold(u0, f64::max);
        assert!(max_u > u0, "no energy deposited");
    }
}
