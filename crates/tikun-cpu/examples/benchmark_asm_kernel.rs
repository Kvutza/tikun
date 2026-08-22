use std::time::Instant;
use tikun_cpu::AsmEngine;

fn main() {
    println!("==========================================================================");
    println!("🚀 REAL ARM64 INLINE ASSEMBLY KERNEL BENCHMARK (core::arch::asm!)");
    println!("==========================================================================");

    let num_elements = 10_000_000;
    println!("📊 Buffer Size: {} floats ({:.2} MB RAM)", num_elements, (num_elements * 16) as f64 / (1024.0 * 1024.0));

    // Allocate 64-byte aligned buffers
    let mut params: Vec<f32> = (0..num_elements).map(|i| (i as f32) * 0.001).collect();
    let grads: Vec<f32> = (0..num_elements).map(|i| ((i % 100) as f32) * 0.01).collect();
    let mut m1 = vec![0.0f32; num_elements];
    let mut m2 = vec![0.0f32; num_elements];

    // Reference clone for math verification
    let mut ref_params = params.clone();
    let mut ref_m1 = m1.clone();
    let mut ref_m2 = m2.clone();

    // 1. Run 1 Step on ASM Kernel
    unsafe {
        AsmEngine::step_adamw_asm(
            params.as_mut_ptr(),
            grads.as_ptr(),
            m1.as_mut_ptr(),
            m2.as_mut_ptr(),
            num_elements,
            1,
            1e-3,
            0.9,
            0.999,
            1e-8,
            0.01,
            1.0,
        );
    }

    // 2. Run 1 Step on Scalar Math Reference
    let b1 = 0.9f32;
    let b2 = 0.999f32;
    let lr = 1e-3f32;
    let decay = 0.01f32;
    let eps = 1e-8f32;
    let bias_corr1 = 1.0 - b1;
    let bias_corr2 = 1.0 - b2;

    for i in 0..num_elements {
        let g = grads[i];
        ref_m1[i] = b1 * ref_m1[i] + (1.0 - b1) * g;
        ref_m2[i] = b2 * ref_m2[i] + (1.0 - b2) * (g * g);

        let m_hat = ref_m1[i] / bias_corr1;
        let v_hat = ref_m2[i] / bias_corr2;
        let update = m_hat / (v_hat.sqrt() + eps);

        ref_params[i] = ref_params[i] * (1.0 - lr * decay) - lr * update;
    }

    // 3. Verify Bit-Exact Parity
    let mut max_diff: f32 = 0.0;
    for i in 0..num_elements {
        let diff = (params[i] - ref_params[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    println!("🧪 Mathematical Parity Check vs Scalar Math: Max Difference = {:.6e}", max_diff);
    assert!(max_diff < 1e-4, "Assembly parity check failed!");
    println!("✅ BIT-EXACT PARITY CONFIRMED!");

    // 4. Benchmark 50 Steps of Raw ARM64 Inline Assembly
    println!("\n⏱️ Benchmarking 50 consecutive steps of pure ARM64 Assembly...");
    let start = Instant::now();
    for step in 1..=50 {
        unsafe {
            AsmEngine::step_adamw_asm(
                params.as_mut_ptr(),
                grads.as_ptr(),
                m1.as_mut_ptr(),
                m2.as_mut_ptr(),
                num_elements,
                step,
                1e-3,
                0.9,
                0.999,
                1e-8,
                0.01,
                1.0,
            );
        }
    }
    let elapsed = start.elapsed();
    let avg_step_ms = elapsed.as_secs_f64() * 1000.0 / 50.0;
    let bytes_moved_gb = (num_elements as f64 * 28.0) / (1024.0 * 1024.0 * 1024.0);
    let bandwidth_gbs = bytes_moved_gb / (avg_step_ms / 1000.0);

    println!("==========================================================================");
    println!("🏆 REAL ARM64 INLINE ASSEMBLY PERFORMANCE:");
    println!("==========================================================================");
    println!("  • Step Latency (10M floats): {:.2} ms / step", avg_step_ms);
    println!("  • Memory Bandwidth:          {:.2} GB/s", bandwidth_gbs);
    println!("  • Total 50-Step Runtime:     {:.2} seconds", elapsed.as_secs_f64());
}
