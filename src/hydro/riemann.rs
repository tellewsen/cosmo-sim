//! HLL Riemann solver for the Euler equations.
//!
//! Computes intercell flux F(U_L, U_R) in the x-direction.
//! U = [ρ, ρvx, ρvy, ρvz, E],  E = ρu + ½ρ|v|²,  P = (γ-1)(E - ½ρ|v|²).

/// MinMod slope limiter: returns 0 if slopes have opposite signs,
/// otherwise the smaller-magnitude slope.
pub fn minmod(a: f64, b: f64) -> f64 {
    if a * b <= 0.0 { 0.0 }
    else if a.abs() <= b.abs() { a }
    else { b }
}

/// Pressure from conserved variables. Returns 0 for zero/negative density or negative kinetic energy.
#[inline]
pub fn pressure(u: &[f64; 5], gamma: f64) -> f64 {
    let rho = u[0];
    if rho <= 0.0 { return 0.0; }
    let vx  = u[1] / rho;
    let vy  = u[2] / rho;
    let vz  = u[3] / rho;
    ((gamma - 1.0) * (u[4] - 0.5 * rho * (vx*vx + vy*vy + vz*vz))).max(0.0)
}

/// Physical flux in the x-direction: F = [ρvx, ρvx²+P, ρvxvy, ρvxvz, (E+P)vx].
/// Returns zero flux for zero/negative density (vacuum).
#[inline]
pub fn flux_x(u: &[f64; 5], gamma: f64) -> [f64; 5] {
    let rho = u[0];
    if rho <= 0.0 { return [0.0; 5]; }
    let vx  = u[1] / rho;
    let vy  = u[2] / rho;
    let vz  = u[3] / rho;
    let p   = pressure(u, gamma);
    [
        rho * vx,
        rho * vx * vx + p,
        rho * vx * vy,
        rho * vx * vz,
        (u[4] + p) * vx,
    ]
}

/// HLL wave speed estimates (Einfeldt, 1988).
/// Returns (S_L, S_R): left and right signal speeds.
pub fn hll_speeds(u_l: &[f64; 5], u_r: &[f64; 5], gamma: f64) -> (f64, f64) {
    let rho_l = u_l[0].max(0.0); let vx_l = if rho_l > 0.0 { u_l[1] / rho_l } else { 0.0 };
    let rho_r = u_r[0].max(0.0); let vx_r = if rho_r > 0.0 { u_r[1] / rho_r } else { 0.0 };
    let p_l = pressure(u_l, gamma);
    let p_r = pressure(u_r, gamma);
    let cs_l = if rho_l > 0.0 { (gamma * p_l / rho_l).sqrt() } else { 0.0 };
    let cs_r = if rho_r > 0.0 { (gamma * p_r / rho_r).sqrt() } else { 0.0 };
    let s_l = (vx_l - cs_l).min(vx_r - cs_r);
    let s_r = (vx_l + cs_l).max(vx_r + cs_r);
    (s_l, s_r)
}

/// HLL intercell flux at the interface between left state U_L and right state U_R.
pub fn hll_flux(u_l: &[f64; 5], u_r: &[f64; 5], gamma: f64) -> [f64; 5] {
    let (s_l, s_r) = hll_speeds(u_l, u_r, gamma);
    if s_l >= 0.0 { return flux_x(u_l, gamma); }
    if s_r <= 0.0 { return flux_x(u_r, gamma); }
    let f_l = flux_x(u_l, gamma);
    let f_r = flux_x(u_r, gamma);
    let ds = s_r - s_l;
    // Degenerate case: both wave speeds equal (e.g. uniform stationary field).
    if ds.abs() < f64::EPSILON * (s_r.abs() + s_l.abs() + 1.0) {
        return std::array::from_fn(|i| 0.5 * (f_l[i] + f_r[i]));
    }
    let mut f = [0.0f64; 5];
    for i in 0..5 {
        f[i] = (s_r * f_l[i] - s_l * f_r[i] + s_l * s_r * (u_r[i] - u_l[i])) / ds;
    }
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sod_left() -> [f64; 5] {
        // ρ=1, vx=0, vy=0, vz=0, P=1  → E = P/(γ-1) = 1.5 for γ=5/3
        let gamma = 5.0_f64 / 3.0;
        [1.0, 0.0, 0.0, 0.0, 1.0 / (gamma - 1.0)]
    }

    fn sod_right() -> [f64; 5] {
        // ρ=0.125, vx=0, vy=0, vz=0, P=0.1
        let gamma = 5.0_f64 / 3.0;
        [0.125, 0.0, 0.0, 0.0, 0.1 / (gamma - 1.0)]
    }

    #[test]
    fn minmod_same_sign_returns_smaller() {
        assert!((minmod(2.0, 3.0) - 2.0).abs() < 1e-14);
        assert!((minmod(-3.0, -1.0) - (-1.0)).abs() < 1e-14);
    }

    #[test]
    fn minmod_opposite_sign_returns_zero() {
        assert_eq!(minmod(1.0, -1.0), 0.0);
        assert_eq!(minmod(-1.0, 1.0), 0.0);
    }

    #[test]
    fn pressure_recovers_initial_p() {
        let gamma = 5.0 / 3.0;
        let u = sod_left();
        let p = pressure(&u, gamma);
        assert!((p - 1.0).abs() < 1e-12, "p={p}");
    }

    #[test]
    fn hll_flux_pure_left_state_when_supersonic_right() {
        // If both speeds positive, flux = F_L
        let gamma = 5.0 / 3.0;
        let u_l = sod_left();
        // Boost both states rightward so S_L > 0
        let mut u_r = sod_right();
        let mut ul = u_l;
        ul[1] = 10.0 * u_l[0]; // vx=10
        u_r[1] = 10.0 * u_r[0];
        let f = hll_flux(&ul, &u_r, gamma);
        let f_l = flux_x(&ul, gamma);
        for i in 0..5 {
            assert!((f[i] - f_l[i]).abs() < 1e-10, "component {i}: {:.4} vs {:.4}", f[i], f_l[i]);
        }
    }

    #[test]
    fn hll_flux_sod_produces_finite_nonneg_rho_flux() {
        let gamma = 5.0 / 3.0;
        let f = hll_flux(&sod_left(), &sod_right(), gamma);
        // Mass flux (f[0]) must be finite and non-negative (contact discontinuity moves right)
        assert!(f[0].is_finite(), "mass flux not finite: {}", f[0]);
        assert!(f[0] >= 0.0, "mass flux negative: {}", f[0]);
    }
}
