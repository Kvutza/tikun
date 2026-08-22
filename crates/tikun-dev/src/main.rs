use clap::{Parser, Subcommand};
use std::process::Command;
use tikun_core::{
    CompilationSchema, GlobalReduction, HardwarePlan, LayoutPlan, MegakernelBackend,
    MegakernelEmitter, MegakernelSpec, MlirEmitter, PointwiseOp, ScheduleIRNode,
    SchedulePipeline, WorkloadRole,
};
use tikun_cpu::KernelAutoTuner;

#[derive(Parser)]
#[command(name = "tikun")]
#[command(about = "Tikun Native Engine: High-Performance Gradient Transformation Compiler", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Auto-tune hardware parameters using pure Rust High-Dimensional TuRBO-ENN / MORBO Bayesian optimization
    Autotune {
        /// Parameter count in millions (default: 25M)
        #[arg(short, long, default_value_t = 25)]
        params_millions: usize,
        /// Number of dimension parameters to optimize jointly (e.g. 6, 100, 1000)
        #[arg(short, long, default_value_t = 100)]
        dimensions: usize,
        /// Number of Bayesian acquisition trials (default: 20)
        #[arg(short, long, default_value_t = 20)]
        trials: usize,
    },
    /// Emit target-specific Persistent Megakernel Code (ARM64 Assembly, NVIDIA PTX, Metal MSL, MLIR)
    Megakernel {
        /// Target hardware backend: arm64, ptx, metal, mlir (default: arm64)
        #[arg(short, long, default_value = "arm64")]
        target: String,
        /// Number of persistent workers (default: 8)
        #[arg(short, long, default_value_t = 8)]
        workers: usize,
    },
    /// Inspect CPU SIMD lowering plan, MLIR Dialect, and memory arena in pure Rust
    Inspect {
        /// Parameter count in millions (default: 10M)
        #[arg(short, long, default_value_t = 10)]
        params_millions: usize,
        /// Algorithm to inspect: adamw, lion, sgd (default: adamw)
        #[arg(short, long, default_value = "adamw")]
        algorithm: String,
        /// Output standardized MLIR Dialect format
        #[arg(long)]
        mlir: bool,
        /// Output standardized JSON schema (for tooling/visualizers/CI)
        #[arg(long)]
        json: bool,
    },
    /// Build target via Buck2 (e.g. //:gpu, //:cpu, //:wheel)
    Build {
        /// Target label (default: //:cpu)
        #[arg(default_value = "//:cpu")]
        target: String,
        /// Extra arguments forwarded to Buck2
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Run test targets via Buck2 or Cargo
    Test {
        /// Test label (default: //:rust_tests)
        #[arg(default_value = "//:rust_tests")]
        target: String,
        /// Extra arguments forwarded to Buck2
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Run static verification and workspace checks
    Check,
    /// Run comprehensive optimizer benchmarks across PyTorch, JAX, and Rust
    Bench,
    /// Synchronize third-party Cargo dependencies into Buck2 rules via reindeer.toml
    Sync,
    /// Execute custom Buck2 BXL script
    Bxl {
        /// BXL script path
        #[arg(default_value = "toolchains//bxl/build_matrix.bxl:matrix")]
        script: String,
    },
}

fn main() {
    let cli = Cli::parse();
    let buck2_bin = "./tools/bin/buck2";

    match cli.command {
        Commands::Autotune { params_millions, dimensions, trials } => {
            let num_params = params_millions * 1_000_000;
            KernelAutoTuner::tune_hd(num_params, dimensions, trials);
        }
        Commands::Megakernel { target, workers } => {
            let backend = match target.to_lowercase().as_str() {
                "ptx" | "cuda" => MegakernelBackend::NvidiaPtx,
                "metal" | "gpu" => MegakernelBackend::AppleMetal,
                "mlir" => MegakernelBackend::MlirVector,
                _ => MegakernelBackend::Arm64Neon,
            };

            let spec = MegakernelSpec {
                name: "tikun_persistent_megakernel".to_string(),
                target_backend: backend,
                num_persistent_workers: workers,
                tile_size_elements: 131072, // 512 KB
                ring_buffer_capacity: 64,
                register_budget_per_thread: 32,
                use_non_temporal_stores: true,
                op: PointwiseOp::AdamW {
                    step_count: 1,
                    learn_rate: 1e-3,
                    beta_one: 0.9,
                    beta_two: 0.999,
                    eps: 1e-8,
                    decay: 0.01,
                },
            };

            let code = MegakernelEmitter::emit_code(&spec);
            println!("{}", code);
        }
        Commands::Inspect { params_millions, algorithm, mlir, json } => {
            let total_params = params_millions * 1_000_000;
            let num_layers = 12;
            let params_per_layer = total_params / num_layers;
            let shapes = vec![params_per_layer; num_layers];
            let layout = LayoutPlan::from_shapes(&shapes);

            let mut pipeline = SchedulePipeline::default();
            pipeline.add_node(ScheduleIRNode::new("load_p_g", WorkloadRole::Load, 16));
            pipeline.add_node(ScheduleIRNode::new("fma_update", WorkloadRole::Compute, 16));
            pipeline.add_node(ScheduleIRNode::new("store_p", WorkloadRole::Store, 16));
            pipeline.reduction = Some(GlobalReduction::GlobalL2Norm { max_norm: 1.0 });

            let op = match algorithm.to_lowercase().as_str() {
                "lion" => PointwiseOp::Lion { learn_rate: 1e-4, beta_one: 0.9, beta_two: 0.99, decay: 0.01 },
                "sgd" => PointwiseOp::SGD { learn_rate: 1e-2, momentum: 0.9, decay: 0.0 },
                _ => PointwiseOp::AdamW { step_count: 1, learn_rate: 1e-3, beta_one: 0.9, beta_two: 0.999, eps: 1e-8, decay: 0.01 },
            };

            let profile = tikun_core::HardwareProfile::active();
            let tile_kb = profile.optimal_tile_kb;
            let unroll = profile.unroll_factor;
            let prefetch = profile.prefetch_bytes;
            let workers = profile.worker_threads;

            if mlir {
                let tile_elements = (tile_kb * 1024 / 4) as i32;
                let mlir_code = MlirEmitter::emit(&pipeline, &layout, op, tile_elements, unroll as i32, prefetch as i32);
                println!("{}", mlir_code);
            } else if json {
                let schema = CompilationSchema::build(&pipeline, &layout, op);
                match schema.to_json() {
                    Ok(j) => println!("{}", j),
                    Err(e) => eprintln!("Serialization error: {}", e),
                }
            } else {
                let report = HardwarePlan::report(&pipeline, &layout, op, tile_kb, unroll, prefetch, workers);
                println!("{}", report);
            }
        }
        Commands::Build { target, args } => {
            println!("build: target={}", target);
            let mut cmd_args = vec!["build".to_string(), "--target-platforms=//:macos-arm64-platform".to_string(), target];
            cmd_args.extend(args);
            let status = Command::new(buck2_bin)
                .args(&cmd_args)
                .status()
                .expect("Failed to execute buck2");
            std::process::exit(status.code().unwrap_or(1));
        }
        Commands::Test { target, args } => {
            println!("test: targets={}", target);
            let mut cmd_args = vec!["test".to_string(), "--target-platforms=//:macos-arm64-platform".to_string()];
            if target == "//:rust_tests" {
                cmd_args.push("//:tikun_core_tests".to_string());
                cmd_args.push("//:tikun_cpu_tests".to_string());
            } else {
                cmd_args.push(target);
            }
            cmd_args.extend(args);
            let status = Command::new(buck2_bin)
                .args(&cmd_args)
                .status()
                .expect("Failed to execute buck2");
            std::process::exit(status.code().unwrap_or(1));
        }
        Commands::Check => {
            let status = Command::new(buck2_bin)
                .args(["targets", "//..."])
                .status()
                .expect("Failed to execute buck2");
            std::process::exit(status.code().unwrap_or(1));
        }
        Commands::Bench => {
            println!("bench: running benchmark suite...");
            let status = Command::new("cargo")
                .args(["bench", "--workspace"])
                .status()
                .expect("Failed to run cargo bench");
            std::process::exit(status.code().unwrap_or(1));
        }
        Commands::Sync => {
            println!("sync: regenerating third-party Buck2 targets via reindeer.toml...");
            let status = Command::new("./tools/bin/reindeer")
                .args(["--config", "third-party/rust/reindeer.toml", "buckify"])
                .status()
                .expect("Failed to execute reindeer");
            std::process::exit(status.code().unwrap_or(1));
        }
        Commands::Bxl { script } => {
            println!("bxl: executing script={}", script);
            let status = Command::new(buck2_bin)
                .args(["bxl", &script])
                .status()
                .expect("Failed to execute buck2");
            std::process::exit(status.code().unwrap_or(1));
        }
    }
}
