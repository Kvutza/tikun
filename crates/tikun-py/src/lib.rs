use pyo3::prelude::*;
use std::sync::OnceLock;
use tikun_core::{
    BufferSlot, GlobalReduction, HardwarePlan, LayoutPlan, MlirEmitter, PointwiseOp,
    ScheduleIRNode, SchedulePipeline, WorkloadRole,
};
use tikun_cpu::{CpuLowering, TensorBuffer, TensorEngine};
use tikun_metal::MetalEngine;

static METAL_ENGINE: OnceLock<Result<MetalEngine, String>> = OnceLock::new();

fn get_metal_engine() -> &'static Result<MetalEngine, String> {
    METAL_ENGINE.get_or_init(MetalEngine::new)
}

fn build_spec(
    param_ptrs: &[usize],
    lengths: &[usize],
    algorithm: &str,
    max_norm: f32,
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    decay: f32,
) -> (SchedulePipeline, LayoutPlan, PointwiseOp) {
    let mut slots = Vec::with_capacity(param_ptrs.len());
    let mut total_bytes = 0;

    for (i, (&ptr, &len)) in param_ptrs.iter().zip(lengths.iter()).enumerate() {
        let byte_size = len * 4;
        slots.push(BufferSlot {
            slot_id: i,
            byte_offset: ptr,
            num_elements: len,
            byte_size,
            is_persistent: true,
        });
        total_bytes += byte_size;
    }

    let layout = LayoutPlan {
        slots,
        total_bytes,
        cache_aligned: param_ptrs.iter().all(|p| p % 64 == 0),
    };

    let mut pipeline = SchedulePipeline::default();
    pipeline.add_node(ScheduleIRNode::new("load_p_g", WorkloadRole::Load, 16));
    pipeline.add_node(ScheduleIRNode::new("fma_update", WorkloadRole::Compute, 16));
    pipeline.add_node(ScheduleIRNode::new("store_p", WorkloadRole::Store, 16));
    if max_norm > 0.0 {
        pipeline.reduction = Some(GlobalReduction::GlobalL2Norm { max_norm });
    }

    let op = match algorithm.to_lowercase().as_str() {
        "lion" => PointwiseOp::Lion { learn_rate: lr, beta_one: b1, beta_two: b2, decay },
        "sgd" => PointwiseOp::SGD { learn_rate: lr, momentum: b1, decay },
        _ => PointwiseOp::AdamW { step_count: 1, learn_rate: lr, beta_one: b1, beta_two: b2, eps, decay },
    };

    (pipeline, layout, op)
}

#[pyfunction]
fn emit_mlir(
    param_ptrs: Vec<usize>,
    lengths: Vec<usize>,
    algorithm: &str,
    max_norm: f32,
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    decay: f32,
    tile_kb: Option<usize>,
    unroll: Option<usize>,
    prefetch: Option<usize>,
) -> PyResult<String> {
    let (pipeline, layout, op) = build_spec(&param_ptrs, &lengths, algorithm, max_norm, lr, b1, b2, eps, decay);
    let tile_bytes = (tile_kb.unwrap_or(512) * 1024 / 4) as i32;
    let unroll_factor = unroll.unwrap_or(4) as i32;
    let prefetch_distance = prefetch.unwrap_or(128) as i32;
    Ok(MlirEmitter::emit(&pipeline, &layout, op, tile_bytes, unroll_factor, prefetch_distance))
}

#[pyfunction]
fn emit_inspect(
    param_ptrs: Vec<usize>,
    lengths: Vec<usize>,
    algorithm: &str,
    max_norm: f32,
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    decay: f32,
    tile_kb: Option<usize>,
    unroll: Option<usize>,
    prefetch: Option<usize>,
    workers: Option<usize>,
) -> PyResult<String> {
    let (pipeline, layout, op) = build_spec(&param_ptrs, &lengths, algorithm, max_norm, lr, b1, b2, eps, decay);
    Ok(HardwarePlan::report(
        &pipeline,
        &layout,
        op,
        tile_kb.unwrap_or(512),
        unroll.unwrap_or(4),
        prefetch.unwrap_or(128),
        workers.unwrap_or(12),
    ))
}

#[pyfunction]
fn step_metal_gpu(
    py: Python,
    param_ptr: usize,
    grad_ptr: usize,
    m1_ptr: usize,
    m2_ptr: usize,
    num_elements: usize,
    step_count: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
) -> PyResult<()> {
    let engine = get_metal_engine()
        .as_ref()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.clone()))?;

    py.allow_threads(|| {
        engine
            .step_adamw_gpu(
                param_ptr,
                grad_ptr,
                m1_ptr,
                m2_ptr,
                num_elements,
                1.0,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    })?;

    Ok(())
}

#[pyfunction]
fn step_fast_buffers(
    py: Python,
    param_ptrs: Vec<usize>,
    grad_ptrs: Vec<usize>,
    m1_ptrs: Vec<usize>,
    m2_ptrs: Vec<usize>,
    lengths: Vec<usize>,
    max_norm: f32,
    opt_kind: &str,
    step_count: usize,
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    decay: f32,
) -> PyResult<f32> {
    let mut buffers = Vec::with_capacity(param_ptrs.len());
    for i in 0..param_ptrs.len() {
        buffers.push(TensorBuffer {
            param_ptr: param_ptrs[i],
            grad_ptr: grad_ptrs[i],
            m1_ptr: m1_ptrs[i],
            m2_ptr: if m2_ptrs.is_empty() { 0 } else { m2_ptrs[i] },
            length: lengths[i],
        });
    }

    let mut pipeline = SchedulePipeline::default();
    pipeline.add_node(ScheduleIRNode::new("load_g", WorkloadRole::Load, 8));
    pipeline.add_node(ScheduleIRNode::new("fma_opt", WorkloadRole::Compute, 8));
    pipeline.add_node(ScheduleIRNode::new("store_p", WorkloadRole::Store, 8));

    let reduction = if max_norm > 0.0 {
        Some(GlobalReduction::GlobalL2Norm { max_norm })
    } else {
        None
    };

    let pointwise = match opt_kind {
        "lion" => PointwiseOp::Lion {
            learn_rate: lr,
            beta_one: b1,
            beta_two: b2,
            decay,
        },
        "sgd" => PointwiseOp::SGD {
            learn_rate: lr,
            momentum: b1,
            decay,
        },
        _ => PointwiseOp::AdamW {
            step_count,
            learn_rate: lr,
            beta_one: b1,
            beta_two: b2,
            eps,
            decay,
        },
    };

    let clip_scale = py.allow_threads(move || {
        CpuLowering::step_two_tier(&pipeline, &buffers, reduction, pointwise)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    })?;

    Ok(clip_scale)
}

#[pyfunction]
fn newton_schulz_2d_ffi(
    py: Python,
    matrix_ptr: usize,
    rows: usize,
    cols: usize,
    steps: usize,
) -> PyResult<()> {
    py.allow_threads(|| unsafe {
        let slice = std::slice::from_raw_parts_mut(matrix_ptr as *mut f32, rows * cols);
        TensorEngine::polar_step(slice, rows, cols, steps);
    });
    Ok(())
}

#[pyfunction]
fn newton_schulz_3d_ffi(
    py: Python,
    tensor_ptr: usize,
    num_heads: usize,
    head_dim_out: usize,
    head_dim_in: usize,
    steps: usize,
) -> PyResult<()> {
    py.allow_threads(|| unsafe {
        let total_len = num_heads * head_dim_out * head_dim_in;
        let slice = std::slice::from_raw_parts_mut(tensor_ptr as *mut f32, total_len);
        TensorEngine::polar_batched(slice, num_heads, head_dim_out, head_dim_in, steps);
    });
    Ok(())
}

#[pymodule]
fn _tikun(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(emit_mlir, m)?)?;
    m.add_function(wrap_pyfunction!(emit_inspect, m)?)?;
    m.add_function(wrap_pyfunction!(step_metal_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(step_fast_buffers, m)?)?;
    m.add_function(wrap_pyfunction!(newton_schulz_2d_ffi, m)?)?;
    m.add_function(wrap_pyfunction!(newton_schulz_3d_ffi, m)?)?;
    Ok(())
}
