//! Benchmarks for gravity solvers.
//! Run with: cargo bench --features fft_pm

use cosmo_sim::gravity::poisson::{grad_phi, solve_poisson_mg};
use cosmo_sim::utils::types::Real;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

#[cfg(feature = "fft_pm")]
use cosmo_sim::gravity::poisson::solve_poisson_fft;

fn bench_poisson_mg(c: &mut Criterion) {
    let mut group = c.benchmark_group("poisson_multigrid");
    for &n in &[16_usize, 32, 64] {
        let dx = 1.0 / n as Real;
        let source: Vec<Real> = (0..n * n * n).map(|i| (i as Real * 0.01).sin()).collect();
        group.bench_with_input(
            BenchmarkId::new("n", n),
            &(source, dx, n),
            |b, (src, dx, n)| {
                b.iter(|| solve_poisson_mg(src, *dx, *n, 1e-6, 50, 2));
            },
        );
    }
    group.finish();
}

#[cfg(feature = "fft_pm")]
fn bench_poisson_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("poisson_fft");
    for &n in &[32_usize, 64, 128] {
        let dx = 1.0 / n as Real;
        let source: Vec<Real> = (0..n * n * n).map(|i| (i as Real * 0.01).sin()).collect();
        group.bench_with_input(
            BenchmarkId::new("n", n),
            &(source, dx, n),
            |b, (src, dx, n)| {
                b.iter(|| solve_poisson_fft(src, *dx, *n));
            },
        );
    }
    group.finish();
}

fn bench_grad_phi(c: &mut Criterion) {
    let n = 64_usize;
    let dx = 1.0 / n as Real;
    let phi: Vec<Real> = (0..n * n * n).map(|i| (i as Real * 0.001).sin()).collect();
    c.bench_function("grad_phi_64", |b| b.iter(|| grad_phi(&phi, dx, n)));
}

#[cfg(feature = "fft_pm")]
criterion_group!(benches, bench_poisson_mg, bench_poisson_fft, bench_grad_phi);
#[cfg(not(feature = "fft_pm"))]
criterion_group!(benches, bench_poisson_mg, bench_grad_phi);
criterion_main!(benches);
