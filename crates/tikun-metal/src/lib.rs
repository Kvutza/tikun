use metal::*;
use std::ffi::c_void;
use tikun_core::{SchedulePipeline, Verifier};

const MSL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void adamw_vectorized_kernel(
    device float4* params [[buffer(0)]],
    device const float4* grads [[buffer(1)]],
    device float4* moment_one [[buffer(2)]],
    device float4* moment_two [[buffer(3)]],
    constant float& clip_scale [[buffer(4)]],
    constant float& learn_rate [[buffer(5)]],
    constant float& beta_one [[buffer(6)]],
    constant float& beta_two [[buffer(7)]],
    constant float& eps [[buffer(8)]],
    constant float& decay [[buffer(9)]],
    uint idx [[thread_position_in_grid]]
) {
    float4 p = params[idx];
    float4 g = grads[idx] * clip_scale;
    float4 m1 = moment_one[idx];
    float4 m2 = moment_two[idx];

    float4 next_m1 = beta_one * m1 + (1.0f - beta_one) * g;
    float4 next_m2 = beta_two * m2 + (1.0f - beta_two) * (g * g);

    float4 m_hat = next_m1 / (1.0f - beta_one);
    float4 v_hat = next_m2 / (1.0f - beta_two);

    float4 step_update = (m_hat / (sqrt(v_hat) + eps)) + (decay * p);
    float4 next_p = p - learn_rate * step_update;

    moment_one[idx] = next_m1;
    moment_two[idx] = next_m2;
    params[idx] = next_p;
}
"#;

pub struct MetalEngine {
    device: Device,
    command_queue: CommandQueue,
    pipeline_state: ComputePipelineState,
}

unsafe impl Send for MetalEngine {}
unsafe impl Sync for MetalEngine {}

impl MetalEngine {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or_else(|| "No Apple Silicon Metal GPU found".to_string())?;
        let command_queue = device.new_command_queue();

        let compile_options = CompileOptions::new();
        let library = device
            .new_library_with_source(MSL_SOURCE, &compile_options)
            .map_err(|e| format!("Metal shader compilation error: {}", e))?;

        let kernel_func = library
            .get_function("adamw_vectorized_kernel", None)
            .map_err(|e| format!("Failed to find Metal kernel function: {}", e))?;

        let pipeline_state = device
            .new_compute_pipeline_state_with_function(&kernel_func)
            .map_err(|e| format!("Failed to create compute pipeline state: {}", e))?;

        Ok(Self {
            device,
            command_queue,
            pipeline_state,
        })
    }

    /// True Zero-Copy Unified Memory GPU Execution
    pub fn step_adamw_gpu(
        &self,
        params_ptr: usize,
        grads_ptr: usize,
        m1_ptr: usize,
        m2_ptr: usize,
        length: usize,
        clip_scale: f32,
        learn_rate: f32,
        beta_one: f32,
        beta_two: f32,
        eps: f32,
        decay: f32,
    ) -> Result<(), String> {
        let vec4_count = length / 4;
        let byte_length = length * std::mem::size_of::<f32>();

        // Zero-Copy Unified Memory Wrapping
        let p_buf = self.device.new_buffer_with_bytes_no_copy(
            params_ptr as *mut c_void,
            byte_length as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        let g_buf = self.device.new_buffer_with_bytes_no_copy(
            grads_ptr as *mut c_void,
            byte_length as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        let m1_buf = self.device.new_buffer_with_bytes_no_copy(
            m1_ptr as *mut c_void,
            byte_length as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        let m2_buf = self.device.new_buffer_with_bytes_no_copy(
            m2_ptr as *mut c_void,
            byte_length as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        );

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline_state);
        encoder.set_buffer(0, Some(&p_buf), 0);
        encoder.set_buffer(1, Some(&g_buf), 0);
        encoder.set_buffer(2, Some(&m1_buf), 0);
        encoder.set_buffer(3, Some(&m2_buf), 0);

        encoder.set_bytes(4, std::mem::size_of::<f32>() as u64, &clip_scale as *const f32 as *const c_void);
        encoder.set_bytes(5, std::mem::size_of::<f32>() as u64, &learn_rate as *const f32 as *const c_void);
        encoder.set_bytes(6, std::mem::size_of::<f32>() as u64, &beta_one as *const f32 as *const c_void);
        encoder.set_bytes(7, std::mem::size_of::<f32>() as u64, &beta_two as *const f32 as *const c_void);
        encoder.set_bytes(8, std::mem::size_of::<f32>() as u64, &eps as *const f32 as *const c_void);
        encoder.set_bytes(9, std::mem::size_of::<f32>() as u64, &decay as *const f32 as *const c_void);

        let thread_group_count = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let grid_size = MTLSize {
            width: ((vec4_count as u64 + 255) / 256) * 256,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_threads(grid_size, thread_group_count);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(())
    }
}

pub struct MetalLowering;

impl MetalLowering {
    pub fn msl_shader(pipeline: &SchedulePipeline) -> Result<String, String> {
        Verifier::verify(pipeline).map_err(|e| format!("Static verification failed: {:?}", e))?;
        Ok(MSL_SOURCE.to_string())
    }
}
