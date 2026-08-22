use candle_core::{Device, Tensor, Var};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use std::time::Instant;
use tikun_core::{PointwiseOp, ScheduleIRNode, SchedulePipeline, WorkloadRole};
use tikun_cpu::{CpuLowering, TensorBuffer};

fn benchmark_scale(num_params: usize, name: &str) {
    println!("\n==========================================================================");
    println!("🦀 RUST SHOWDOWN: Hugging Face Candle vs. tikun Engine ({}: {} Params)", name, num_params);
    println!("==========================================================================");

    let device = Device::Cpu;

    // -------------------------------------------------------------
    // 1. Hugging Face Candle AdamW Setup
    // -------------------------------------------------------------
    let raw_p: Vec<f32> = vec![1.0; num_params];
    let raw_g: Vec<f32> = vec![0.1; num_params];

    let param_tensor = Tensor::from_vec(raw_p.clone(), (num_params,), &device).unwrap();
    let var_p = Var::from_tensor(&param_tensor).unwrap();
    let target = Tensor::from_vec(raw_g.clone(), (num_params,), &device).unwrap();

    let params_adamw = ParamsAdamW {
        lr: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.01,
    };

    let mut candle_opt = AdamW::new(vec![var_p.clone()], params_adamw).unwrap();

    // Generate real backward gradients in Candle
    let diff = (var_p.as_tensor() - &target).unwrap();
    let loss = diff.sqr().unwrap().sum_all().unwrap();
    let grads = loss.backward().unwrap();

    // Warmup Candle
    candle_opt.step(&grads).unwrap();

    // Time Candle step
    let start = Instant::now();
    for _ in 0..10 {
        candle_opt.step(&grads).unwrap();
    }
    let candle_time = start.elapsed().as_secs_f64() * 1000.0 / 10.0;
    println!("⏱️ 1. Hugging Face Candle (Rust Native): {:.2} ms / step", candle_time);

    // -------------------------------------------------------------
    // 2. tikun Engine Setup (Zero-Copy Resident Memory)
    // -------------------------------------------------------------
    let mut p_tikun = raw_p;
    let g_tikun = raw_g;
    let mut m1_tikun = vec![0.0f32; num_params];
    let mut m2_tikun = vec![0.0f32; num_params];

    let mut pipeline = SchedulePipeline::default();
    pipeline.add_node(ScheduleIRNode::new("load_g", WorkloadRole::Load, 8));
    pipeline.add_node(ScheduleIRNode::new("fma_adamw", WorkloadRole::Compute, 8));
    pipeline.add_node(ScheduleIRNode::new("store_p", WorkloadRole::Store, 8));

    let buf = TensorBuffer {
        param_ptr: p_tikun.as_mut_ptr() as usize,
        grad_ptr: g_tikun.as_ptr() as usize,
        m1_ptr: m1_tikun.as_mut_ptr() as usize,
        m2_ptr: m2_tikun.as_mut_ptr() as usize,
        length: num_params,
    };

    let pointwise = PointwiseOp::AdamW {
        step_count: 1,
        learn_rate: 1e-3,
        beta_one: 0.9,
        beta_two: 0.999,
        eps: 1e-8,
        decay: 0.01,
    };

    // Warmup tikun
    CpuLowering::step_two_tier(&pipeline, &[buf], None, pointwise.clone()).unwrap();

    // Time tikun step
    let start = Instant::now();
    for _ in 0..10 {
        CpuLowering::step_two_tier(&pipeline, &[buf], None, pointwise.clone()).unwrap();
    }
    let tikun_time = start.elapsed().as_secs_f64() * 1000.0 / 10.0;
    println!("⚡ 2. tikun Engine (ARM NEON SIMD):      {:.2} ms / step", tikun_time);

    let speedup = candle_time / tikun_time;
    println!("\n🏆 Speedup vs Candle: tikun is {:.2}x faster!", speedup);
}

fn main() {
    benchmark_scale(10_000_000, "10M Parameters");
    benchmark_scale(50_000_000, "50M Parameters");
    benchmark_scale(100_000_000, "100M Parameters");
}
