use tikun_core::{ScheduleIRNode, SchedulePipeline, WorkloadRole};
use tikun_cpu::CpuLowering;

fn main() {
    println!("Tikun Hardware Optimizer - CLI Direct Runner");

    let mut pipeline = SchedulePipeline::default();
    pipeline.add_node(ScheduleIRNode::new("load_g", WorkloadRole::Load, 8));
    pipeline.add_node(ScheduleIRNode::new("fma_adamw", WorkloadRole::Compute, 8));
    pipeline.add_node(ScheduleIRNode::new("store_p", WorkloadRole::Store, 8));

    let mut params = vec![1.0; 1000];
    let grads = vec![0.1; 1000];
    let mut m1 = vec![0.0; 1000];
    let mut m2 = vec![0.0; 1000];

    match CpuLowering::step_adamw(
        &pipeline,
        &mut params,
        &grads,
        &mut m1,
        &mut m2,
        1,     // step_count
        0.01,  // learn_rate
        0.9,   // beta_one
        0.999, // beta_two
        1e-8,  // eps
        0.01,  // decay
    ) {
        Ok(_) => println!("Successfully updated 1000 parameters via tikun-cpu."),
        Err(e) => eprintln!("Execution error: {}", e),
    }
}
