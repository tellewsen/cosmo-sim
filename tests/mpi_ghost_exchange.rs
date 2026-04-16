/// Multi-rank integration test for `reduce_ghost_density`.
///
/// Must be run via mpirun, not directly:
///
///   cargo build --features mpi_parallel --test mpi_ghost_exchange
///   mpirun -np 2 ./target/debug/deps/mpi_ghost_exchange-*
///
/// Or via the Makefile:
///
///   make test-mpi
///
/// The test verifies that after CIC-style ghost deposits are made in
/// adjacent rank's ghost cells, `reduce_ghost_density` correctly
/// accumulates those contributions into the owner's slab.
///
/// Layout (n=4, 2 ranks):
///   Rank 0 owns slabs [0, 2)  →  cells 0, 1
///   Rank 1 owns slabs [2, 4)  →  cells 2, 3
///
/// Setup:
///   Rank 0 deposits 5.0 into its right ghost (cell 2) for all j.
///   Rank 1 deposits 7.0 into its left ghost  (cell 1) for all j.
///
/// Expected after exchange:
///   Rank 0, cell 1 (right boundary): += 7.0  (received from rank 1's left ghost)
///   Rank 1, cell 2 (left boundary):  += 5.0  (received from rank 0's right ghost)

fn main() {
    #[cfg(feature = "mpi_parallel")]
    run_mpi_test();

    #[cfg(not(feature = "mpi_parallel"))]
    eprintln!("SKIP: build with --features mpi_parallel to run this test");
}

#[cfg(feature = "mpi_parallel")]
fn run_mpi_test() {
    use cosmo_sim::mpi::MpiContext;

    let ctx = MpiContext::init();
    let n: usize = 4;
    let n2 = n * n;
    let n3 = n * n * n;

    assert_eq!(
        ctx.size, 2,
        "this test requires exactly 2 MPI ranks (got {}); run with mpirun -np 2",
        ctx.size
    );

    // All cells start at 0.
    let mut rho = vec![0.0_f64; n3];

    if ctx.rank == 0 {
        // Rank 0 right ghost: cell 2 (i_hi = 2, so right_ghost = 2 % n = 2)
        for j in 0..n2 {
            rho[2 * n2 + j] = 5.0;
        }
    } else {
        // Rank 1 left ghost: cell 1 (i_lo = 2, so left_ghost = (2 + 4 - 1) % 4 = 1)
        for j in 0..n2 {
            rho[1 * n2 + j] = 7.0;
        }
    }

    ctx.reduce_ghost_density(&mut rho, n);

    if ctx.rank == 0 {
        // Cell 1 is rank 0's right boundary (i_hi - 1 = 1).
        // It should have received rank 1's left ghost = 7.0.
        for j in 0..n2 {
            let got = rho[1 * n2 + j];
            assert!(
                (got - 7.0).abs() < 1e-12,
                "rank 0 cell 1 j={j}: expected 7.0, got {got}"
            );
        }
        // Cell 0 (i_lo): receives rank 1's right ghost (cell 0).
        // Rank 1 wrote nothing into cell 0, so it should stay 0.
        for j in 0..n2 {
            let got = rho[0 * n2 + j];
            assert!(
                got.abs() < 1e-12,
                "rank 0 cell 0 j={j}: expected 0.0, got {got}"
            );
        }
        println!("rank 0: PASS");
    } else {
        // Cell 2 is rank 1's left boundary (i_lo = 2).
        // It should have received rank 0's right ghost = 5.0.
        for j in 0..n2 {
            let got = rho[2 * n2 + j];
            assert!(
                (got - 5.0).abs() < 1e-12,
                "rank 1 cell 2 j={j}: expected 5.0, got {got}"
            );
        }
        // Cell 3 (i_hi - 1 = 3): receives rank 0's left ghost (cell 3).
        // Rank 0 wrote nothing into cell 3, so it should stay 0.
        for j in 0..n2 {
            let got = rho[3 * n2 + j];
            assert!(
                got.abs() < 1e-12,
                "rank 1 cell 3 j={j}: expected 0.0, got {got}"
            );
        }
        println!("rank 1: PASS");
    }
}
