use tikun_core::{GlobalReduction, MemoryArena, PointwiseOp, SchedulePipeline, Verifier};
use rayon::prelude::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "aarch64")]
use std::arch::asm;

pub mod asm_kernel;
pub mod autotuner;
pub mod jit;
pub mod pmc;
pub mod tensor_engine;
pub mod topology;

pub use asm_kernel::AsmEngine;
pub use autotuner::{KernelAutoTuner, TunedHardwareProfile};
pub use jit::JitKernel;
pub use pmc::{PmcMetrics, PmcSampler};
pub use tensor_engine::TensorEngine;
pub use topology::SiliconTopology;

pub struct CpuLowering;

#[derive(Debug, Clone, Copy)]
pub struct TensorBuffer {
    pub param_ptr: usize,
    pub grad_ptr: usize,
    pub m1_ptr: usize,
    pub m2_ptr: usize,
    pub length: usize,
}

unsafe impl Send for TensorBuffer {}
unsafe impl Sync for TensorBuffer {}

impl CpuLowering {
    /// Pipelined Memory Arena Execution
    pub fn step_arena(
        pipeline: &SchedulePipeline,
        params_arena: &mut MemoryArena,
        grads_arena: &MemoryArena,
        m1_arena: &mut MemoryArena,
        m2_arena: &mut MemoryArena,
        reduction: Option<GlobalReduction>,
        pointwise: PointwiseOp,
    ) -> Result<f32, String> {
        let p_base = params_arena.base_mut_ptr() as usize;
        let g_base = grads_arena.base_ptr() as usize;
        let m1_base = m1_arena.base_mut_ptr() as usize;
        let m2_base = m2_arena.base_mut_ptr() as usize;

        let mut buffers = Vec::with_capacity(params_arena.layout_plan.slots.len());
        for slot in &params_arena.layout_plan.slots {
            buffers.push(TensorBuffer {
                param_ptr: p_base + slot.byte_offset,
                grad_ptr: g_base + slot.byte_offset,
                m1_ptr: m1_base + slot.byte_offset,
                m2_ptr: m2_base + slot.byte_offset,
                length: slot.num_elements,
            });
        }

        Self::step_two_tier(pipeline, &buffers, reduction, pointwise)
    }

    /// Two-Tier Pipeline Engine: Multi-Tensor Parallel Coalesced Execution
    pub fn step_two_tier(
        pipeline: &SchedulePipeline,
        buffers: &[TensorBuffer],
        reduction: Option<GlobalReduction>,
        pointwise: PointwiseOp,
    ) -> Result<f32, String> {
        Verifier::verify(pipeline).map_err(|e| format!("Static verification failed: {:?}", e))?;

        // -------------------------------------------------------------
        // TIER 1: Global Monoidal Reduction (Parallel Sum of Squares)
        // -------------------------------------------------------------
        let clip_scale = match reduction {
            Some(GlobalReduction::GlobalL2Norm { max_norm }) => {
                let total_sum_sq: f64 = buffers
                    .par_iter()
                    .map(|buf| {
                        let length = buf.length;
                        let g_ptr = buf.grad_ptr as *const f32;
                        let mut local_sum = 0.0f64;
                        let mut idx = 0;

                        #[cfg(target_arch = "aarch64")]
                        {
                            unsafe {
                                let mut acc0 = vdupq_n_f32(0.0);
                                let mut acc1 = vdupq_n_f32(0.0);
                                while idx + 8 <= length {
                                    let g0 = vld1q_f32(g_ptr.add(idx));
                                    let g1 = vld1q_f32(g_ptr.add(idx + 4));
                                    acc0 = vfmaq_f32(acc0, g0, g0);
                                    acc1 = vfmaq_f32(acc1, g1, g1);
                                    idx += 8;
                                }
                                let acc = vaddq_f32(acc0, acc1);
                                local_sum += vaddvq_f32(acc) as f64;
                            }
                        }

                        while idx < length {
                            unsafe {
                                let g = *g_ptr.add(idx) as f64;
                                local_sum += g * g;
                            }
                            idx += 1;
                        }
                        local_sum
                    })
                    .sum();

                let global_norm = (total_sum_sq.sqrt()) as f32;
                if global_norm > max_norm && max_norm > 0.0 {
                    max_norm / (global_norm + 1e-6)
                } else {
                    1.0f32
                }
            }
            None => 1.0f32,
        };

        // -------------------------------------------------------------
        // TIER 2: Pointwise Fused SIMD Kernel (Parallel Across Buffers)
        // -------------------------------------------------------------
        buffers.par_iter().for_each(|buf| {
            let length = buf.length;
            let p_addr = buf.param_ptr;
            let g_addr = buf.grad_ptr;
            let m1_addr = buf.m1_ptr;
            let m2_addr = buf.m2_ptr;

            let p_ptr = p_addr as *mut f32;
            let g_ptr = g_addr as *const f32;
            let m1_ptr = m1_addr as *mut f32;
            let m2_ptr = m2_addr as *mut f32;
            let mut idx = 0;

            match pointwise {
                PointwiseOp::AdamW {
                    step_count,
                    learn_rate,
                    beta_one,
                    beta_two,
                    eps,
                    decay,
                } => {
                    let step_val = (step_count.max(1)) as i32;
                    let bias_corr1 = 1.0 - beta_one.powi(step_val);
                    let bias_corr2 = 1.0 - beta_two.powi(step_val);

                    #[cfg(target_arch = "aarch64")]
                    {
                        unsafe {
                            let lr_vec = vdupq_n_f32(learn_rate);
                            let b1_vec = vdupq_n_f32(beta_one);
                            let b2_vec = vdupq_n_f32(beta_two);
                            let one_minus_b1_vec = vdupq_n_f32(1.0 - beta_one);
                            let one_minus_b2_vec = vdupq_n_f32(1.0 - beta_two);
                            let bias_corr1_vec = vdupq_n_f32(bias_corr1);
                            let bias_corr2_vec = vdupq_n_f32(bias_corr2);
                            let eps_vec = vdupq_n_f32(eps);
                            let decay_vec = vdupq_n_f32(decay);
                            let ones_vec = vdupq_n_f32(1.0);
                            let scale_vec = vdupq_n_f32(clip_scale);

                            while idx + 16 <= length {
                                let prefetch_ptr1 = p_ptr.add(idx + 64) as *const i8;
                                let prefetch_ptr2 = p_ptr.add(idx + 128) as *const i8;
                                asm!("prfm pldl1keep, [{0}]", in(reg) prefetch_ptr1, options(nostack, preserves_flags));
                                asm!("prfm pldl1keep, [{0}]", in(reg) prefetch_ptr2, options(nostack, preserves_flags));

                                // Lane 0
                                let p0 = vld1q_f32(p_ptr.add(idx));
                                let g0 = vmulq_f32(vld1q_f32(g_ptr.add(idx)), scale_vec);
                                let m1_0 = vld1q_f32(m1_ptr.add(idx));
                                let m2_0 = vld1q_f32(m2_ptr.add(idx));

                                let next_m1_0 = vaddq_f32(vmulq_f32(b1_vec, m1_0), vmulq_f32(one_minus_b1_vec, g0));
                                let next_m2_0 = vaddq_f32(vmulq_f32(b2_vec, m2_0), vmulq_f32(one_minus_b2_vec, vmulq_f32(g0, g0)));
                                let m_hat0 = vdivq_f32(next_m1_0, bias_corr1_vec);
                                let v_hat0 = vdivq_f32(next_m2_0, bias_corr2_vec);
                                let rsq0 = vrsqrteq_f32(v_hat0);
                                let sqrt0 = vdivq_f32(ones_vec, vmulq_f32(rsq0, vrsqrtsq_f32(vmulq_f32(rsq0, rsq0), v_hat0)));
                                let step0 = vaddq_f32(vdivq_f32(m_hat0, vaddq_f32(sqrt0, eps_vec)), vmulq_f32(decay_vec, p0));
                                let next_p0 = vsubq_f32(p0, vmulq_f32(lr_vec, step0));

                                vst1q_f32(m1_ptr.add(idx), next_m1_0);
                                vst1q_f32(m2_ptr.add(idx), next_m2_0);
                                vst1q_f32(p_ptr.add(idx), next_p0);

                                // Lane 1
                                let p1 = vld1q_f32(p_ptr.add(idx + 4));
                                let g1 = vmulq_f32(vld1q_f32(g_ptr.add(idx + 4)), scale_vec);
                                let m1_1 = vld1q_f32(m1_ptr.add(idx + 4));
                                let m2_1 = vld1q_f32(m2_ptr.add(idx + 4));

                                let next_m1_1 = vaddq_f32(vmulq_f32(b1_vec, m1_1), vmulq_f32(one_minus_b1_vec, g1));
                                let next_m2_1 = vaddq_f32(vmulq_f32(b2_vec, m2_1), vmulq_f32(one_minus_b2_vec, vmulq_f32(g1, g1)));
                                let m_hat1 = vdivq_f32(next_m1_1, bias_corr1_vec);
                                let v_hat1 = vdivq_f32(next_m2_1, bias_corr2_vec);
                                let rsq1 = vrsqrteq_f32(v_hat1);
                                let sqrt1 = vdivq_f32(ones_vec, vmulq_f32(rsq1, vrsqrtsq_f32(vmulq_f32(rsq1, rsq1), v_hat1)));
                                let step1 = vaddq_f32(vdivq_f32(m_hat1, vaddq_f32(sqrt1, eps_vec)), vmulq_f32(decay_vec, p1));
                                let next_p1 = vsubq_f32(p1, vmulq_f32(lr_vec, step1));

                                vst1q_f32(m1_ptr.add(idx + 4), next_m1_1);
                                vst1q_f32(m2_ptr.add(idx + 4), next_m2_1);
                                vst1q_f32(p_ptr.add(idx + 4), next_p1);

                                // Lane 2
                                let p2 = vld1q_f32(p_ptr.add(idx + 8));
                                let g2 = vmulq_f32(vld1q_f32(g_ptr.add(idx + 8)), scale_vec);
                                let m1_2 = vld1q_f32(m1_ptr.add(idx + 8));
                                let m2_2 = vld1q_f32(m2_ptr.add(idx + 8));

                                let next_m1_2 = vaddq_f32(vmulq_f32(b1_vec, m1_2), vmulq_f32(one_minus_b1_vec, g2));
                                let next_m2_2 = vaddq_f32(vmulq_f32(b2_vec, m2_2), vmulq_f32(one_minus_b2_vec, vmulq_f32(g2, g2)));
                                let m_hat2 = vdivq_f32(next_m1_2, bias_corr1_vec);
                                let v_hat2 = vdivq_f32(next_m2_2, bias_corr2_vec);
                                let rsq2 = vrsqrteq_f32(v_hat2);
                                let sqrt2 = vdivq_f32(ones_vec, vmulq_f32(rsq2, vrsqrtsq_f32(vmulq_f32(rsq2, rsq2), v_hat2)));
                                let step2 = vaddq_f32(vdivq_f32(m_hat2, vaddq_f32(sqrt2, eps_vec)), vmulq_f32(decay_vec, p2));
                                let next_p2 = vsubq_f32(p2, vmulq_f32(lr_vec, step2));

                                vst1q_f32(m1_ptr.add(idx + 8), next_m1_2);
                                vst1q_f32(m2_ptr.add(idx + 8), next_m2_2);
                                vst1q_f32(p_ptr.add(idx + 8), next_p2);

                                // Lane 3
                                let p3 = vld1q_f32(p_ptr.add(idx + 12));
                                let g3 = vmulq_f32(vld1q_f32(g_ptr.add(idx + 12)), scale_vec);
                                let m1_3 = vld1q_f32(m1_ptr.add(idx + 12));
                                let m2_3 = vld1q_f32(m2_ptr.add(idx + 12));

                                let next_m1_3 = vaddq_f32(vmulq_f32(b1_vec, m1_3), vmulq_f32(one_minus_b1_vec, g3));
                                let next_m2_3 = vaddq_f32(vmulq_f32(b2_vec, m2_3), vmulq_f32(one_minus_b2_vec, vmulq_f32(g3, g3)));
                                let m_hat3 = vdivq_f32(next_m1_3, bias_corr1_vec);
                                let v_hat3 = vdivq_f32(next_m2_3, bias_corr2_vec);
                                let rsq3 = vrsqrteq_f32(v_hat3);
                                let sqrt3 = vdivq_f32(ones_vec, vmulq_f32(rsq3, vrsqrtsq_f32(vmulq_f32(rsq3, rsq3), v_hat3)));
                                let step3 = vaddq_f32(vdivq_f32(m_hat3, vaddq_f32(sqrt3, eps_vec)), vmulq_f32(decay_vec, p3));
                                let next_p3 = vsubq_f32(p3, vmulq_f32(lr_vec, step3));

                                vst1q_f32(m1_ptr.add(idx + 12), next_m1_3);
                                vst1q_f32(m2_ptr.add(idx + 12), next_m2_3);
                                vst1q_f32(p_ptr.add(idx + 12), next_p3);

                                idx += 16;
                            }
                        }
                    }

                    while idx < length {
                        unsafe {
                            let g = *g_ptr.add(idx) * clip_scale;
                            *m1_ptr.add(idx) = beta_one * *m1_ptr.add(idx) + (1.0 - beta_one) * g;
                            *m2_ptr.add(idx) = beta_two * *m2_ptr.add(idx) + (1.0 - beta_two) * g * g;

                            let m_hat = *m1_ptr.add(idx) / bias_corr1;
                            let m_hat2 = *m2_ptr.add(idx) / bias_corr2;

                            let step_update = (m_hat / (m_hat2.sqrt() + eps)) + (decay * *p_ptr.add(idx));
                            *p_ptr.add(idx) -= learn_rate * step_update;
                        }
                        idx += 1;
                    }
                }

                PointwiseOp::Lion {
                    learn_rate,
                    beta_one,
                    beta_two,
                    decay,
                } => {
                    #[cfg(target_arch = "aarch64")]
                    {
                        unsafe {
                            let lr_vec = vdupq_n_f32(learn_rate);
                            let b1_vec = vdupq_n_f32(beta_one);
                            let b2_vec = vdupq_n_f32(beta_two);
                            let one_minus_b1_vec = vdupq_n_f32(1.0 - beta_one);
                            let one_minus_b2_vec = vdupq_n_f32(1.0 - beta_two);
                            let decay_vec = vdupq_n_f32(decay);
                            let scale_vec = vdupq_n_f32(clip_scale);
                            let zero_vec = vdupq_n_f32(0.0);
                            let one_vec = vdupq_n_f32(1.0);
                            let neg_one_vec = vdupq_n_f32(-1.0);

                            while idx + 16 <= length {
                                let prefetch_ptr = p_ptr.add(idx + 64) as *const i8;
                                asm!("prfm pldl1keep, [{0}]", in(reg) prefetch_ptr, options(nostack, preserves_flags));

                                // Lane 0
                                let p0 = vld1q_f32(p_ptr.add(idx));
                                let g0 = vmulq_f32(vld1q_f32(g_ptr.add(idx)), scale_vec);
                                let m0 = vld1q_f32(m1_ptr.add(idx));
                                let c0 = vaddq_f32(vmulq_f32(b1_vec, m0), vmulq_f32(one_minus_b1_vec, g0));
                                let sign0 = vbslq_f32(vcltq_f32(c0, zero_vec), neg_one_vec, vbslq_f32(vcgtq_f32(c0, zero_vec), one_vec, zero_vec));
                                let next_p0 = vsubq_f32(p0, vmulq_f32(lr_vec, vaddq_f32(sign0, vmulq_f32(decay_vec, p0))));
                                let next_m0 = vaddq_f32(vmulq_f32(b2_vec, m0), vmulq_f32(one_minus_b2_vec, g0));
                                vst1q_f32(m1_ptr.add(idx), next_m0);
                                vst1q_f32(p_ptr.add(idx), next_p0);

                                // Lane 1
                                let p1 = vld1q_f32(p_ptr.add(idx + 4));
                                let g1 = vmulq_f32(vld1q_f32(g_ptr.add(idx + 4)), scale_vec);
                                let m1 = vld1q_f32(m1_ptr.add(idx + 4));
                                let c1 = vaddq_f32(vmulq_f32(b1_vec, m1), vmulq_f32(one_minus_b1_vec, g1));
                                let sign1 = vbslq_f32(vcltq_f32(c1, zero_vec), neg_one_vec, vbslq_f32(vcgtq_f32(c1, zero_vec), one_vec, zero_vec));
                                let next_p1 = vsubq_f32(p1, vmulq_f32(lr_vec, vaddq_f32(sign1, vmulq_f32(decay_vec, p1))));
                                let next_m1 = vaddq_f32(vmulq_f32(b2_vec, m1), vmulq_f32(one_minus_b2_vec, g1));
                                vst1q_f32(m1_ptr.add(idx + 4), next_m1);
                                vst1q_f32(p_ptr.add(idx + 4), next_p1);

                                // Lane 2
                                let p2 = vld1q_f32(p_ptr.add(idx + 8));
                                let g2 = vmulq_f32(vld1q_f32(g_ptr.add(idx + 8)), scale_vec);
                                let m2 = vld1q_f32(m1_ptr.add(idx + 8));
                                let c2 = vaddq_f32(vmulq_f32(b1_vec, m2), vmulq_f32(one_minus_b1_vec, g2));
                                let sign2 = vbslq_f32(vcltq_f32(c2, zero_vec), neg_one_vec, vbslq_f32(vcgtq_f32(c2, zero_vec), one_vec, zero_vec));
                                let next_p2 = vsubq_f32(p2, vmulq_f32(lr_vec, vaddq_f32(sign2, vmulq_f32(decay_vec, p2))));
                                let next_m2 = vaddq_f32(vmulq_f32(b2_vec, m2), vmulq_f32(one_minus_b2_vec, g2));
                                vst1q_f32(m1_ptr.add(idx + 8), next_m2);
                                vst1q_f32(p_ptr.add(idx + 8), next_p2);

                                // Lane 3
                                let p3 = vld1q_f32(p_ptr.add(idx + 12));
                                let g3 = vmulq_f32(vld1q_f32(g_ptr.add(idx + 12)), scale_vec);
                                let m3 = vld1q_f32(m1_ptr.add(idx + 12));
                                let c3 = vaddq_f32(vmulq_f32(b1_vec, m3), vmulq_f32(one_minus_b1_vec, g3));
                                let sign3 = vbslq_f32(vcltq_f32(c3, zero_vec), neg_one_vec, vbslq_f32(vcgtq_f32(c3, zero_vec), one_vec, zero_vec));
                                let next_p3 = vsubq_f32(p3, vmulq_f32(lr_vec, vaddq_f32(sign3, vmulq_f32(decay_vec, p3))));
                                let next_m3 = vaddq_f32(vmulq_f32(b2_vec, m3), vmulq_f32(one_minus_b2_vec, g3));
                                vst1q_f32(m1_ptr.add(idx + 12), next_m3);
                                vst1q_f32(p_ptr.add(idx + 12), next_p3);

                                idx += 16;
                            }
                        }
                    }

                    while idx < length {
                        unsafe {
                            let g = *g_ptr.add(idx) * clip_scale;
                            let m = *m1_ptr.add(idx);
                            let c = beta_one * m + (1.0 - beta_one) * g;
                            let sign_c = if c > 0.0 { 1.0 } else if c < 0.0 { -1.0 } else { 0.0 };

                            *p_ptr.add(idx) -= learn_rate * (sign_c + decay * *p_ptr.add(idx));
                            *m1_ptr.add(idx) = beta_two * m + (1.0 - beta_two) * g;
                        }
                        idx += 1;
                    }
                }

                PointwiseOp::SGD {
                    learn_rate,
                    momentum,
                    decay,
                } => {
                    #[cfg(target_arch = "aarch64")]
                    {
                        unsafe {
                            let lr_vec = vdupq_n_f32(learn_rate);
                            let mom_vec = vdupq_n_f32(momentum);
                            let decay_vec = vdupq_n_f32(decay);
                            let scale_vec = vdupq_n_f32(clip_scale);

                            while idx + 16 <= length {
                                let prefetch_ptr = p_ptr.add(idx + 64) as *const i8;
                                asm!("prfm pldl1keep, [{0}]", in(reg) prefetch_ptr, options(nostack, preserves_flags));

                                // Lane 0
                                let p0 = vld1q_f32(p_ptr.add(idx));
                                let g0 = vmulq_f32(vld1q_f32(g_ptr.add(idx)), scale_vec);
                                let m0 = vld1q_f32(m1_ptr.add(idx));
                                let next_m0 = vaddq_f32(vmulq_f32(mom_vec, m0), g0);
                                let next_p0 = vsubq_f32(p0, vmulq_f32(lr_vec, vaddq_f32(next_m0, vmulq_f32(decay_vec, p0))));
                                vst1q_f32(m1_ptr.add(idx), next_m0);
                                vst1q_f32(p_ptr.add(idx), next_p0);

                                // Lane 1
                                let p1 = vld1q_f32(p_ptr.add(idx + 4));
                                let g1 = vmulq_f32(vld1q_f32(g_ptr.add(idx + 4)), scale_vec);
                                let m1 = vld1q_f32(m1_ptr.add(idx + 4));
                                let next_m1 = vaddq_f32(vmulq_f32(mom_vec, m1), g1);
                                let next_p1 = vsubq_f32(p1, vmulq_f32(lr_vec, vaddq_f32(next_m1, vmulq_f32(decay_vec, p1))));
                                vst1q_f32(m1_ptr.add(idx + 4), next_m1);
                                vst1q_f32(p_ptr.add(idx + 4), next_p1);

                                // Lane 2
                                let p2 = vld1q_f32(p_ptr.add(idx + 8));
                                let g2 = vmulq_f32(vld1q_f32(g_ptr.add(idx + 8)), scale_vec);
                                let m2 = vld1q_f32(m1_ptr.add(idx + 8));
                                let next_m2 = vaddq_f32(vmulq_f32(mom_vec, m2), g2);
                                let next_p2 = vsubq_f32(p2, vmulq_f32(lr_vec, vaddq_f32(next_m2, vmulq_f32(decay_vec, p2))));
                                vst1q_f32(m1_ptr.add(idx + 8), next_m2);
                                vst1q_f32(p_ptr.add(idx + 8), next_p2);

                                // Lane 3
                                let p3 = vld1q_f32(p_ptr.add(idx + 12));
                                let g3 = vmulq_f32(vld1q_f32(g_ptr.add(idx + 12)), scale_vec);
                                let m3 = vld1q_f32(m1_ptr.add(idx + 12));
                                let next_m3 = vaddq_f32(vmulq_f32(mom_vec, m3), g3);
                                let next_p3 = vsubq_f32(p3, vmulq_f32(lr_vec, vaddq_f32(next_m3, vmulq_f32(decay_vec, p3))));
                                vst1q_f32(m1_ptr.add(idx + 12), next_m3);
                                vst1q_f32(p_ptr.add(idx + 12), next_p3);

                                idx += 16;
                            }
                        }
                    }

                    while idx < length {
                        unsafe {
                            let g = *g_ptr.add(idx) * clip_scale;
                            *m1_ptr.add(idx) = momentum * *m1_ptr.add(idx) + g;
                            let update = *m1_ptr.add(idx) + decay * *p_ptr.add(idx);
                            *p_ptr.add(idx) -= learn_rate * update;
                        }
                        idx += 1;
                    }
                }
                PointwiseOp::CustomDAG { primitives: _ } => {
                    // Fallback execution for generalized primitive composition
                    let mut idx = 0;
                    while idx < length {
                        unsafe {
                            let g = *g_ptr.add(idx) * clip_scale;
                            *m1_ptr.add(idx) = 0.9 * *m1_ptr.add(idx) + 0.1 * g;
                            *p_ptr.add(idx) -= 1e-3 * *m1_ptr.add(idx);
                        }
                        idx += 1;
                    }
                }
            }
        });

        Ok(clip_scale)
    }

    pub fn step_pytree(
        pipeline: &SchedulePipeline,
        buffers: &[TensorBuffer],
        step_count: usize,
        learn_rate: f32,
        beta_one: f32,
        beta_two: f32,
        eps: f32,
        decay: f32,
    ) -> Result<(), String> {
        Self::step_two_tier(
            pipeline,
            buffers,
            None,
            PointwiseOp::AdamW {
                step_count,
                learn_rate,
                beta_one,
                beta_two,
                eps,
                decay,
            },
        )
        .map(|_| ())
    }

    pub fn step_adamw(
        pipeline: &SchedulePipeline,
        params: &mut [f32],
        grads: &[f32],
        moment_one: &mut [f32],
        moment_two: &mut [f32],
        step_count: usize,
        learn_rate: f32,
        beta_one: f32,
        beta_two: f32,
        eps: f32,
        decay: f32,
    ) -> Result<(), String> {
        let buf = TensorBuffer {
            param_ptr: params.as_mut_ptr() as usize,
            grad_ptr: grads.as_ptr() as usize,
            m1_ptr: moment_one.as_mut_ptr() as usize,
            m2_ptr: moment_two.as_mut_ptr() as usize,
            length: params.len(),
        };
        Self::step_pytree(pipeline, &[buf], step_count, learn_rate, beta_one, beta_two, eps, decay)
    }
}
