//! Friends-of-Friends halo finder and stacked radial profile output.
//!
//! FoF linking length: b = 0.2 × mean inter-particle spacing.
//! Profile bins: 20 log-spaced bins from 0.1 to 3.0 × R_200.
//! Output: `halo_profiles_{idx:04}.txt` with columns:
//!   r/R200  rho_gas/rho_mean  T_K  rho_dm/rho_mean  N_cells

use crate::utils::types::Real;

// ── FoF ──────────────────────────────────────────────────────────────────────

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]]; // path compression
        x = parent[x];
    }
    x
}

fn union(parent: &mut [usize], x: usize, y: usize) {
    let rx = find(parent, x);
    let ry = find(parent, y);
    if rx != ry {
        parent[rx] = ry;
    }
}

/// Find FoF groups using a cell-list (O(N) average, O(N²) worst case for dense systems).
///
/// `b` is the linking length [Mpc/h]. `box_size` is the periodic box size [Mpc/h].
/// Pair search includes periodic boundary wrapping.
///
/// Returns a `Vec<Vec<usize>>` — each inner vec is the particle indices of one group.
pub fn friends_of_friends(pos: &[[Real; 3]], b: Real, box_size: Real) -> Vec<Vec<usize>> {
    let n = pos.len();
    if n == 0 {
        return Vec::new();
    }
    let mut parent: Vec<usize> = (0..n).collect();

    // Build cell grid with cell side ≥ b so that every pair within b is in
    // adjacent (including diagonal) cells.
    // Cap n_cells at N^(1/3) + 1 so the cell array never has more entries than
    // particles (prevents O(N²) memory for tiny b relative to box_size).
    // Reducing n_cells makes cell_size larger (≥ b) — correctness is preserved.
    let n_cells_ideal = (box_size / b).floor() as usize;
    let n_cells_cap = (n as f64).cbrt().ceil() as usize + 1;
    let n_cells = n_cells_ideal.min(n_cells_cap).max(1);
    let cell_size = box_size / n_cells as Real;

    let cell_flat = |p: [Real; 3]| -> usize {
        let ci = ((p[0] / cell_size).floor() as isize).rem_euclid(n_cells as isize) as usize;
        let cj = ((p[1] / cell_size).floor() as isize).rem_euclid(n_cells as isize) as usize;
        let ck = ((p[2] / cell_size).floor() as isize).rem_euclid(n_cells as isize) as usize;
        ci * n_cells * n_cells + cj * n_cells + ck
    };

    let cell_3d = |flat: usize| -> (usize, usize, usize) {
        let ck = flat % n_cells;
        let cj = (flat / n_cells) % n_cells;
        let ci = flat / (n_cells * n_cells);
        (ci, cj, ck)
    };

    // Assign particles to cells
    let mut cells: Vec<Vec<usize>> = vec![Vec::new(); n_cells * n_cells * n_cells];
    for i in 0..n {
        cells[cell_flat(pos[i])].push(i);
    }

    let b2 = b * b;

    // Iterate over all cell pairs (ci_flat, ni_flat) with ni_flat >= ci_flat.
    // For each pair, check all particle pairs between the two cells.
    for ci_flat in 0..n_cells * n_cells * n_cells {
        if cells[ci_flat].is_empty() {
            continue;
        }
        let (ci, cj, ck) = cell_3d(ci_flat);

        for di in 0..3usize {
            for dj in 0..3usize {
                for dk in 0..3usize {
                    let ni = (ci + di + n_cells - 1) % n_cells;
                    let nj = (cj + dj + n_cells - 1) % n_cells;
                    let nk = (ck + dk + n_cells - 1) % n_cells;
                    let ni_flat = ni * n_cells * n_cells + nj * n_cells + nk;

                    // Process each cell pair only once
                    if ni_flat < ci_flat {
                        continue;
                    }

                    for (pos_in_cell, &i) in cells[ci_flat].iter().enumerate() {
                        // Within the same cell: start after i's position to avoid
                        // duplicate pairs.  For adjacent cells: start from 0.
                        let start = if ni_flat == ci_flat {
                            pos_in_cell + 1
                        } else {
                            0
                        };

                        for &j in &cells[ni_flat][start..] {
                            // Periodic minimum-image distance
                            let mut dx = pos[i][0] - pos[j][0];
                            if dx.abs() > box_size * 0.5 {
                                dx -= dx.signum() * box_size;
                            }
                            let mut dy = pos[i][1] - pos[j][1];
                            if dy.abs() > box_size * 0.5 {
                                dy -= dy.signum() * box_size;
                            }
                            let mut dz = pos[i][2] - pos[j][2];
                            if dz.abs() > box_size * 0.5 {
                                dz -= dz.signum() * box_size;
                            }

                            if dx * dx + dy * dy + dz * dz < b2 {
                                union(&mut parent, i, j);
                            }
                        }
                    }
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

/// R_200: radius enclosing mean density = 200 × ρ_crit at z=0.
///
/// ρ_crit(z=0) = 2.775×10¹¹ h² M_sun/(Mpc/h)³  (standard formula).
pub fn r200(n_members: usize, m_part: Real, h: Real) -> Real {
    let rho_crit = 2.775e11 * h * h; // M_sun/(Mpc/h)³
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
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn stacked_profiles(
    dm_pos: &[[Real; 3]],
    gas_rho: &[Real],
    gas_u: &[Real],
    n_grid: usize,
    box_size: Real,
    m_part: Real,
    omega_m: Real,
    h: Real,
    m_min: Real,
    m_max: Real,
    groups: &[Vec<usize>],
) -> (
    [Real; N_BINS],
    [Real; N_BINS],
    [Real; N_BINS],
    [Real; N_BINS],
    [usize; N_BINS],
    usize,
) {
    let _ = omega_m; // retained for potential future use (e.g. z-dependent ρ_crit)
    let dx = box_size / n_grid as Real;
    debug_assert_eq!(
        gas_rho.len(),
        n_grid * n_grid * n_grid,
        "gas_rho length mismatch"
    );
    debug_assert_eq!(
        gas_u.len(),
        n_grid * n_grid * n_grid,
        "gas_u length mismatch"
    );

    // T [K] = (μ m_p / k_B) × u [km/s]²
    // k_B = 8.314e-3 (km/s)²/K × m_p,  μ = 0.59
    let u_to_t = 0.59 / 8.314e-3; // K / (km/s)²

    // Log-spaced bin edges
    let log_rmin = R_MIN_FRAC.ln();
    let log_rmax = R_MAX_FRAC.ln();
    let dlog = (log_rmax - log_rmin) / N_BINS as Real;

    let mut rho_gas_sum = [0.0f64; N_BINS];
    let mut t_sum = [0.0f64; N_BINS];
    let mut rho_dm_sum = [0.0f64; N_BINS];
    let mut counts = [0usize; N_BINS];
    let mut n_halos = 0usize;

    for group in groups {
        let m_halo = group.len() as Real * m_part;
        if m_halo < m_min || m_halo > m_max {
            continue;
        }
        n_halos += 1;

        let centre = group_centre(dm_pos, group);
        let r200_h = r200(group.len(), m_part, h);

        // DM: count particles in each radial bin
        let mut dm_bin_counts = [0usize; N_BINS];
        for &i in group {
            let mut dr = [
                dm_pos[i][0] - centre[0],
                dm_pos[i][1] - centre[1],
                dm_pos[i][2] - centre[2],
            ];
            // Periodic wrapping
            for d in &mut dr {
                if d.abs() > box_size * 0.5 {
                    *d -= d.signum() * box_size;
                }
            }
            let r = (dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]).sqrt() / r200_h;
            if !(R_MIN_FRAC..R_MAX_FRAC).contains(&r) {
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
                    let mut dr = [
                        cell_pos[0] - centre[0],
                        cell_pos[1] - centre[1],
                        cell_pos[2] - centre[2],
                    ];
                    for d in &mut dr {
                        if d.abs() > box_size * 0.5 {
                            *d -= d.signum() * box_size;
                        }
                    }
                    let r = (dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]).sqrt() / r200_h;
                    if !(R_MIN_FRAC..R_MAX_FRAC).contains(&r) {
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
    let mut r_mid = [0.0f64; N_BINS];
    let mut rho_gas = [0.0f64; N_BINS];
    let mut t_mean = [0.0f64; N_BINS];
    let mut rho_dm = [0.0f64; N_BINS];
    for bin in 0..N_BINS {
        r_mid[bin] = (log_rmin + (bin as Real + 0.5) * dlog).exp();
        if counts[bin] > 0 {
            rho_gas[bin] = rho_gas_sum[bin] / counts[bin] as Real;
            t_mean[bin] = t_sum[bin] / counts[bin] as Real;
        }
        // DM: average particle count per bin across halos
        rho_dm[bin] = rho_dm_sum[bin] / n_halos.max(1) as Real;
    }

    (r_mid, rho_gas, t_mean, rho_dm, counts, n_halos)
}

// ── Halo mass function ────────────────────────────────────────────────────────

const N_MF_BINS: usize = 20;

/// Compute the halo mass function dN/dlog₁₀M [h³/Mpc³] from FoF groups.
///
/// Returns `(log10_m_mid, dn_dlgm)` arrays of length `N_MF_BINS`.
/// Mass range: log₁₀(M/(M_sun/h)) from 10 to 15.
/// Bins with zero halos have `dn_dlgm = 0`.
pub fn halo_mass_function(
    groups: &[Vec<usize>],
    m_part: Real,
    box_size: Real,
) -> ([Real; N_MF_BINS], [Real; N_MF_BINS]) {
    const LOG_M_MIN: Real = 10.0; // log10(M_sun/h)
    const LOG_M_MAX: Real = 15.0;
    let dlgm = (LOG_M_MAX - LOG_M_MIN) / N_MF_BINS as Real;
    let volume = box_size * box_size * box_size; // (Mpc/h)³

    let mut counts = [0usize; N_MF_BINS];
    for group in groups {
        let log_m = (group.len() as Real * m_part).log10();
        if !(LOG_M_MIN..LOG_M_MAX).contains(&log_m) {
            continue;
        }
        let bin = ((log_m - LOG_M_MIN) / dlgm) as usize;
        if bin < N_MF_BINS {
            counts[bin] += 1;
        }
    }

    let mut log10_m_mid = [0.0f64; N_MF_BINS];
    let mut dn_dlgm = [0.0f64; N_MF_BINS];
    for i in 0..N_MF_BINS {
        log10_m_mid[i] = LOG_M_MIN + (i as Real + 0.5) * dlgm;
        dn_dlgm[i] = counts[i] as Real / (volume * dlgm);
    }

    (log10_m_mid, dn_dlgm)
}

/// Write mass function to a text file.
pub fn write_mass_function(
    path: &str,
    log10_m_mid: &[Real; N_MF_BINS],
    dn_dlgm: &[Real; N_MF_BINS],
    n_halos_total: usize,
) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).expect("Cannot create mass function file");
    writeln!(f, "# n_halos_total={n_halos_total}").unwrap();
    writeln!(f, "# log10(M [M_sun/h])  dN/dlog10M [(Mpc/h)^-3]").unwrap();
    for i in 0..N_MF_BINS {
        writeln!(f, "{:.4e} {:.4e}", log10_m_mid[i], dn_dlgm[i]).unwrap();
    }
}

/// Write profiles to a text file.
#[allow(clippy::too_many_arguments)]
pub fn write_profiles(
    path: &str,
    r_mid: &[Real; N_BINS],
    rho_gas: &[Real; N_BINS],
    t_k: &[Real; N_BINS],
    rho_dm: &[Real; N_BINS],
    counts: &[usize; N_BINS],
    n_halos: usize,
    mass_label: &str,
) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).expect("Cannot create profile file");
    writeln!(f, "# mass_bin={mass_label}  n_halos={n_halos}").unwrap();
    writeln!(
        f,
        "# r/R200  rho_gas/rho_mean  T_K  rho_dm/rho_mean  N_cells"
    )
    .unwrap();
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

        let groups = friends_of_friends(&pos, 0.05, 200.0);
        let sizes: Vec<usize> = groups.iter().map(|g| g.len()).collect();
        assert!(sizes.contains(&8), "did not find cluster of 8: {sizes:?}");
        assert!(
            sizes.contains(&1),
            "did not find isolated particle: {sizes:?}"
        );
    }

    #[test]
    fn fof_links_across_periodic_boundary() {
        // Two particles on opposite sides of the box that should be linked periodically
        let pos = vec![
            [0.1f64, 0.0, 0.0],  // near left edge
            [99.9f64, 0.0, 0.0], // near right edge → distance = 0.2 < b=0.5
        ];
        let groups = friends_of_friends(&pos, 0.5, 100.0);
        assert_eq!(
            groups.len(),
            1,
            "particles across boundary should be linked"
        );
        assert_eq!(groups[0].len(), 2);
    }

    #[test]
    fn fof_no_particles() {
        let groups = friends_of_friends(&[], 0.2, 64.0);
        assert!(groups.is_empty());
    }

    #[test]
    fn r200_increases_with_mass() {
        let r1 = r200(100, 1e8, 0.67);
        let r2 = r200(1000, 1e8, 0.67);
        assert!(r2 > r1, "r200 should increase with particle count");
    }

    #[test]
    fn r200_depends_on_h() {
        // Higher h → higher ρ_crit → smaller R_200 for same mass
        let r_lo = r200(1000, 1e10, 0.7);
        let r_hi = r200(1000, 1e10, 0.67);
        assert!(r_lo < r_hi, "R_200 should decrease with increasing h");
    }

    #[test]
    fn mass_function_counts_correctly() {
        // Two groups: one with 100 particles (M=1e11), one with 1000 (M=1e12).
        let m_part = 1.0e9_f64; // M_sun/h per particle
        let groups: Vec<Vec<usize>> = vec![
            (0..100).collect(),   // M = 1e11
            (100..1100).collect(), // M = 1e12
        ];
        let box_size = 64.0_f64;
        let (log_m, dn) = halo_mass_function(&groups, m_part, box_size);

        // Find which bins the two halos fall into
        let bin11 = ((11.0 - 10.0) / ((15.0 - 10.0) / N_MF_BINS as f64)) as usize;
        let bin12 = ((12.0 - 10.0) / ((15.0 - 10.0) / N_MF_BINS as f64)) as usize;
        assert!(
            dn[bin11] > 0.0,
            "10^11 halo not counted (bin {bin11}, log_m={:.2})",
            log_m[bin11]
        );
        assert!(
            dn[bin12] > 0.0,
            "10^12 halo not counted (bin {bin12}, log_m={:.2})",
            log_m[bin12]
        );
        // Bins for halos should be in the expected mass decade
        assert!((log_m[bin11] - 11.0).abs() < 0.5);
        assert!((log_m[bin12] - 12.0).abs() < 0.5);
    }

    #[test]
    fn mass_function_empty_returns_zeros() {
        let (_, dn) = halo_mass_function(&[], 1.0e9, 64.0);
        assert!(dn.iter().all(|&x| x == 0.0));
    }
}
