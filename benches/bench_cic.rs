//! Benchmarks for CIC density assignment and force interpolation.

use cosmo_sim::particles::particles::Particles;
use cosmo_sim::utils::types::Real;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn make_particles(n: usize, l: Real) -> Particles {
    let mut p = Particles::new(n, 1.0, l);
    for i in 0..n {
        p.pos[i] = [
            l * (i as Real * 0.6180339887).fract(),
            l * (i as Real * 0.3819660113).fract(),
            l * (i as Real * 0.2360679775).fract(),
        ];
        p.vel[i] = [
            (i as Real * 0.1).sin(),
            (i as Real * 0.2).cos(),
            (i as Real * 0.3).sin(),
        ];
    }
    p
}

fn bench_cic_density(c: &mut Criterion) {
    let mut group = c.benchmark_group("cic_density");
    for &(n_part, ng) in &[(32768_usize, 32_usize), (262144, 64)] {
        let p = make_particles(n_part, ng as Real);
        group.bench_with_input(
            BenchmarkId::new(format!("n_part={n_part}/ng={ng}"), n_part),
            &(p, ng),
            |b, (p, ng)| b.iter(|| p.cic_density(*ng)),
        );
    }
    group.finish();
}

fn bench_cic_interp(c: &mut Criterion) {
    let ng = 32_usize;
    let p = make_particles(32768, ng as Real);
    let grid: Vec<[Real; 3]> = (0..ng * ng * ng)
        .map(|i| [(i as Real * 0.001).sin(); 3])
        .collect();
    c.bench_function("cic_interp_32k/ng=32", |b| {
        b.iter(|| p.cic_interpolate_force(&grid, ng))
    });
}

fn bench_drift(c: &mut Criterion) {
    let mut p = make_particles(262144, 256.0);
    c.bench_function("drift_256k", |b| b.iter(|| p.drift(0.001, 0.9)));
}

criterion_group!(benches, bench_cic_density, bench_cic_interp, bench_drift);
criterion_main!(benches);
