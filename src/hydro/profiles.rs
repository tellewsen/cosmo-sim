//! Friends-of-Friends halo finder and stacked radial profile output.
//!
//! FoF linking length: b = 0.2 × mean inter-particle spacing.
//! Profile bins: 20 log-spaced bins from 0.1 to 3.0 × R_200.
//! Output: `halo_profiles_{idx:04}.txt` with columns:
//!   r/R200  rho_gas/rho_mean  T_K  rho_dm/rho_mean  N_cells

use crate::utils::types::Real;

// ── FoF ──────────────────────────────────────────────────────────────────────

/// Find FoF groups in the particle set.
///
/// Returns a `Vec<Vec<usize>>` — each inner vec is the particle indices of one group.
/// `b` is the linking length [Mpc/h].
/// Uses union-find for efficiency (O(N α(N)) ≈ O(N)).
pub fn friends_of_friends(pos: &[[Real; 3]], b: Real) -> Vec<Vec<usize>> {
    let n = pos.len();
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut Vec<usize>, mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    let b2 = b * b;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = pos[i][0] - pos[j][0];
            let dy = pos[i][1] - pos[j][1];
            let dz = pos[i][2] - pos[j][2];
            if dx * dx + dy * dy + dz * dz < b2 {
                let ri = find(&mut parent, i);
                let rj = find(&mut parent, j);
                if ri != rj {
                    parent[ri] = rj;
                }
            }
        }
    }

    // Collect groups
    let mut groups: std::collections::HashMap<usize, Vec<usize>> = Default::default();
    for i in 0..n {
        groups.entry(find(&mut parent, i)).or_default().push(i);
    }
    groups.into_values().collect()
}

/// Centre-of-mass position of a particle group.
pub fn group_centre(pos: &[[Real; 3]], members: &[usize]) -> [Real; 3] {
    let mut c = [0.0f64; 3];
    for &i in members {
        c[0] += pos[i][0];
        c[1] += pos[i][1];
        c[2] += pos[i][2];
    }
    let n = members.len() as Real;
    [c[0] / n, c[1] / n, c[2] / n]
}

/// R_200: radius enclosing mean density = 200 × ρ_crit.
///
/// ρ_crit ≈ 1.246e11 M_sun/(Mpc/h)³ for h=0.67 (hardcoded; pass as param in future).
pub fn r200(n_members: usize, m_part: Real, _omega_m: Real) -> Real {
    let rho_crit = 1.246e11_f64; // M_sun/(Mpc/h)³ for h=0.67
    let m_halo = n_members as Real * m_part;
    (3.0 * m_halo / (4.0 * std::f64::consts::PI * 200.0 * rho_crit)).powf(1.0 / 3.0)
}

// ── Stacked profiles ─────────────────────────────────────────────────────────

const N_BINS: usize = 20;
const R_MIN_FRAC: Real = 0.1; // in units of R_200
const R_MAX_FRAC: Real = 3.0;

/// Compute stacked radial profiles for halos in a given mass range.
///
/// Returns (r_mid/R200, rho_gas, T_K, rho_dm, counts, n_halos).
#[allow(clippy::too_many_arguments)]
pub fn stacked_profiles(
    dm_pos:   &[[Real; 3]],
    gas_rho:  &[Real],
    gas_u:    &[Real],
    n_grid:   usize,
    box_size: Real,
    m_part:   Real,
    omega_m:  Real,
    m_min:    Real,
    m_max:    Real,
    groups:   &[Vec<usize>],
) -> ([Real; N_BINS], [Real; N_BINS], [Real; N_BINS], [Real; N_BINS], [usize; N_BINS], usize) {
    let dx = box_size / n_grid as Real;
    debug_assert_eq!(gas_rho.len(), n_grid * n_grid * n_grid, "gas_rho length mismatch");
    debug_assert_eq!(gas_u.len(),   n_grid * n_grid * n_grid, "gas_u length mismatch");

    // Temperature from u: T = u / (k_B / (μ m_p))
    let u_to_t = 1.0 / (0.008314 / 0.59);

    // Log-spaced bin edges
    let log_rmin = R_MIN_FRAC.ln();
    let log_rmax = R_MAX_FRAC.ln();
    let dlog = (log_rmax - log_rmin) / N_BINS as Real;

    let mut rho_gas_sum = [0.0f64; N_BINS];
    let mut t_sum       = [0.0f64; N_BINS];
    let mut rho_dm_sum  = [0.0f64; N_BINS];
    let mut counts      = [0usize; N_BINS];
    let mut n_halos     = 0usize;

    for group in groups {
        let m_halo = group.len() as Real * m_part;
        if m_halo < m_min || m_halo > m_max {
            continue;
        }
        n_halos += 1;

        let centre = group_centre(dm_pos, group);
        let r200_h = r200(group.len(), m_part, omega_m);

        // DM: count particles in each radial bin
        let mut dm_bin_counts = [0usize; N_BINS];
        for &i in group {
            let dr = [
                dm_pos[i][0] - centre[0],
                dm_pos[i][1] - centre[1],
                dm_pos[i][2] - centre[2],
            ];
            let r = (dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]).sqrt() / r200_h;
            if r < R_MIN_FRAC || r >= R_MAX_FRAC {
                continue;
            }
            let bin = ((r.ln() - log_rmin) / dlog) as usize;
            if bin < N_BINS {
                dm_bin_counts[bin] += 1;
            }
        }

        // Gas: loop over a cube of cells around the halo centre
        let r_max_phys = R_MAX_FRAC * r200_h;
        let cells_per_side = (2.0 * r_max_phys / dx).ceil() as usize + 2;
        let ci0 = ((centre[0] / dx) as isize).saturating_sub(cells_per_side as isize / 2);
        let cj0 = ((centre[1] / dx) as isize).saturating_sub(cells_per_side as isize / 2);
        let ck0 = ((centre[2] / dx) as isize).saturating_sub(cells_per_side as isize / 2);

        for di in 0..cells_per_side {
            for dj in 0..cells_per_side {
                for dk in 0..cells_per_side {
                    let ci = ((ci0 + di as isize).rem_euclid(n_grid as isize)) as usize;
                    let cj = ((cj0 + dj as isize).rem_euclid(n_grid as isize)) as usize;
                    let ck = ((ck0 + dk as isize).rem_euclid(n_grid as isize)) as usize;
                    let cell_pos = [
                        (ci as Real + 0.5) * dx,
                        (cj as Real + 0.5) * dx,
                        (ck as Real + 0.5) * dx,
                    ];
                    let dr = [
                        cell_pos[0] - centre[0],
                        cell_pos[1] - centre[1],
                        cell_pos[2] - centre[2],
                    ];
                    let r =
                        (dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]).sqrt() / r200_h;
                    if r < R_MIN_FRAC || r >= R_MAX_FRAC {
                        continue;
                    }
                    let bin = ((r.ln() - log_rmin) / dlog) as usize;
                    if bin >= N_BINS {
                        continue;
                    }
                    let idx = ci * n_grid * n_grid + cj * n_grid + ck;
                    rho_gas_sum[bin] += gas_rho[idx];
                    t_sum[bin] += gas_u[idx] * u_to_t;
                    counts[bin] += 1;
                }
            }
        }

        for bin in 0..N_BINS {
            rho_dm_sum[bin] += dm_bin_counts[bin] as Real;
        }
    }

    // Normalise
    let mut r_mid   = [0.0f64; N_BINS];
    let mut rho_gas = [0.0f64; N_BINS];
    let mut t_mean  = [0.0f64; N_BINS];
    let mut rho_dm  = [0.0f64; N_BINS];
    for bin in 0..N_BINS {
        r_mid[bin] = (log_rmin + (bin as Real + 0.5) * dlog).exp();
        if counts[bin] > 0 {
            rho_gas[bin] = rho_gas_sum[bin] / counts[bin] as Real;
            t_mean[bin] = t_sum[bin] / counts[bin] as Real;
        }
        // DM: average particle count per bin across halos (not normalised by gas cell count)
        rho_dm[bin] = rho_dm_sum[bin] / n_halos.max(1) as Real;
    }

    (r_mid, rho_gas, t_mean, rho_dm, counts, n_halos)
}

/// Write profiles to a text file.
pub fn write_profiles(
    path:      &str,
    r_mid:     &[Real; N_BINS],
    rho_gas:   &[Real; N_BINS],
    t_k:       &[Real; N_BINS],
    rho_dm:    &[Real; N_BINS],
    counts:    &[usize; N_BINS],
    n_halos:   usize,
    mass_label: &str,
) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).expect("Cannot create profile file");
    writeln!(f, "# mass_bin={mass_label}  n_halos={n_halos}").unwrap();
    writeln!(f, "# r/R200  rho_gas/rho_mean  T_K  rho_dm/rho_mean  N_cells").unwrap();
    for i in 0..N_BINS {
        writeln!(
            f,
            "{:.4e} {:.4e} {:.4e} {:.4e} {}",
            r_mid[i], rho_gas[i], t_k[i], rho_dm[i], counts[i]
        )
        .unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fof_finds_single_cluster() {
        // 8 particles tightly grouped + 1 isolated particle
        let mut pos = vec![[0.0f64; 3]; 8];
        for (i, p) in pos.iter_mut().enumerate() {
            p[0] = i as f64 * 0.01; // all within 0.07 of each other
        }
        pos.push([100.0, 0.0, 0.0]); // isolated

        let groups = friends_of_friends(&pos, 0.05);
        let sizes: Vec<usize> = groups.iter().map(|g| g.len()).collect();
        assert!(sizes.contains(&8), "did not find cluster of 8: {sizes:?}");
        assert!(sizes.contains(&1), "did not find isolated particle: {sizes:?}");
    }

    #[test]
    fn r200_increases_with_mass() {
        let r1 = r200(100, 1e8, 0.3);
        let r2 = r200(1000, 1e8, 0.3);
        assert!(r2 > r1, "r200 should increase with particle count");
    }
}
