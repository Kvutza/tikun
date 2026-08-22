use tikun_core::{
    BufferSlot, GlobalReduction, HardwarePlan, LayoutPlan, MlirEmitter, PointwiseOp,
    ScheduleIRNode, SchedulePipeline, WorkloadRole,
};

fn main() {
    println!("==========================================================================");
    println!("🦀 PURE RUST TIKUN MLIR & COMPILER INSPECTION DEMO");
    println!("==========================================================================");

    // 1. Construct a pure Rust memory layout for a 12-layer Transformer (25M parameters)
    let total_params = 25_000_000;
    let num_layers = 12;
    let params_per_layer = total_params / num_layers;
    let shapes = vec![params_per_layer; num_layers];
    let layout = LayoutPlan::from_shapes(&shapes);

    // 2. Construct pure Rust Schedule Pipeline
    let mut pipeline = SchedulePipeline::default();
    pipeline.add_node(ScheduleIRNode::new("load_p_g", WorkloadRole::Load, 16));
    pipeline.add_node(ScheduleIRNode::new("fma_update", WorkloadRole::Compute, 16));
    pipeline.add_node(ScheduleIRNode::new("store_p", WorkloadRole::Store, 16));
    pipeline.reduction = Some(GlobalReduction::GlobalL2Norm { max_norm: 1.0 });

    let op = PointwiseOp::AdamW {
        step_count: 1,
        learn_rate: 1e-3,
        beta_one: 0.9,
        beta_two: 0.999,
        eps: 1e-8,
        decay: 0.01,
    };

    // 3. Emit MLIR in pure Rust
    println!("\n[1/2] Emitting Pure Rust MLIR Dialect:");
    let mlir_code = MlirEmitter::emit(&pipeline, &layout, op.clone(), 131072, 4, 128);
    println!("=== MLIR EMISSION ===");
    println!("{}", mlir_code);

    // 4. Emit Hardware Plan in pure Rust
    println!("\n[2/2] Emitting Pure Rust Hardware Inspection Plan:");
    let report = HardwarePlan::report(&pipeline, &layout, op, 512, 4, 128, 12);
    println!("{}", report);
}
