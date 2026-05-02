//! cosmo_sim — command-line entry point.
//!
//! Usage:
//!   cosmo_sim run  --params params.toml
//!   cosmo_sim init --params params.toml   # write example parameter file

use clap::{Parser, Subcommand};
use cosmo_sim::mpi::MpiContext;
use cosmo_sim::utils::config::Config;
use cosmo_sim::{example_config, Simulation};

#[derive(Parser)]
#[command(
    name = "cosmo_sim",
    version = "0.1.0",
    about = "Cosmological N-body + modified gravity simulation (ISIS/RAMSES port to Rust)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a simulation from a parameter file
    Run {
        #[arg(short, long, default_value = "params.toml")]
        params: String,
        #[arg(
            short = 'i',
            long,
            help = "Path to initial conditions binary (N×6 f64 LE)"
        )]
        ic: Option<String>,
        #[arg(
            short = 's',
            long,
            default_value = "42",
            help = "Random seed for Zel'dovich IC generator (used when --ic is omitted)"
        )]
        seed: u64,
        #[arg(
            short = 'r',
            long,
            help = "Restart from a snapshot binary written by this sim"
        )]
        restart: Option<String>,
    },
    /// Write an example parameter file and exit
    Init {
        #[arg(short, long, default_value = "params.toml")]
        params: String,
    },
}

fn main() {
    // MPI must be initialised before any other work.  In non-MPI builds
    // this returns a single-rank stub with no side effects.
    let mpi_ctx = MpiContext::init();

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Init { params } => {
            let cfg = example_config();
            cfg.to_file(&params)
                .expect("Failed to write parameter file");
            println!("Example parameter file written to: {params}");
            println!("Edit it, then run:  cosmo_sim run --params {params}");
        }

        Commands::Run {
            params,
            ic,
            seed,
            restart,
        } => {
            let cfg = Config::from_file(&params).unwrap_or_else(|e| {
                eprintln!("Error reading {params}: {e}");
                std::process::exit(1);
            });

            log::info!("Loaded parameters from {params}");

            let mut sim = if let Some(snap_path) = restart {
                Simulation::read_snapshot(&snap_path, cfg, mpi_ctx).unwrap_or_else(|e| {
                    eprintln!("Error reading snapshot {snap_path}: {e}");
                    std::process::exit(1);
                })
            } else {
                let mut s = Simulation::new(cfg, mpi_ctx);
                if let Some(ic_path) = ic {
                    s.load_ic(&ic_path).unwrap_or_else(|e| {
                        eprintln!("Error loading IC from {ic_path}: {e}");
                        std::process::exit(1);
                    });
                } else {
                    log::info!("No IC file given — generating Zel'dovich IC (seed={seed})");
                    s.generate_ic(seed);
                }
                s
            };

            sim.run();
        }
    }
}
