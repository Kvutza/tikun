pub struct JitKernel {
    pub unroll: usize,
    pub prefetch: usize,
    pub streaming: bool,
}

impl JitKernel {
    pub fn new(unroll: usize, prefetch: usize, streaming: bool) -> Self {
        Self {
            unroll: unroll.clamp(1, 16),
            prefetch,
            streaming,
        }
    }

    /// Spawns a JIT kernel specialized directly to the active hardware profile
    pub fn active() -> Self {
        let prof = tikun_core::HardwareProfile::active();
        Self::new(prof.unroll_factor, prof.prefetch_bytes, prof.unroll_factor >= 8)
    }

    /// Executes the JIT-specialized vector loop over contiguous float buffers
    pub unsafe fn execute(
        &self,
        params: *mut f32,
        grads: *const f32,
        m1: *mut f32,
        m2: *mut f32,
        length: usize,
        lr: f32,
        b1: f32,
        b2: f32,
        eps: f32,
        decay: f32,
    ) {
        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            let scale_lr = vdupq_n_f32(lr);
            let b1_vec = vdupq_n_f32(b1);
            let b1_comp = vdupq_n_f32(1.0 - b1);
            let b2_vec = vdupq_n_f32(b2);
            let b2_comp = vdupq_n_f32(1.0 - b2);
            let eps_vec = vdupq_n_f32(eps);
            let decay_vec = vdupq_n_f32(1.0 - lr * decay);

            let mut idx = 0;
            let step_size = 4 * self.unroll;

            while idx + step_size <= length {
                // Software Prefetch dynamically specialized
                if self.prefetch > 0 {
                    core::arch::asm!(
                        "prfm pldl1keep, [{0}, {1}]",
                        "prfm pldl1keep, [{2}, {1}]",
                        in(reg) params.add(idx),
                        in(reg) self.prefetch,
                        in(reg) grads.add(idx),
                        options(nostack, readonly)
                    );
                }

                for u in 0..self.unroll {
                    let off = idx + u * 4;
                    let p = vld1q_f32(params.add(off));
                    let g = vld1q_f32(grads.add(off));
                    let cur_m1 = vld1q_f32(m1.add(off));
                    let cur_m2 = vld1q_f32(m2.add(off));

                    // In-Register Fused Moments
                    let next_m1 = vfmaq_f32(vmulq_f32(b1_vec, cur_m1), g, b1_comp);
                    let g_sqr = vmulq_f32(g, g);
                    let next_m2 = vfmaq_f32(vmulq_f32(b2_vec, cur_m2), g_sqr, b2_comp);

                    vst1q_f32(m1.add(off), next_m1);
                    vst1q_f32(m2.add(off), next_m2);

                    // Fast Newton-Raphson Inverse Square Root (Single-Cycle FMA instead of Slow Division)
                    let denom_sqr = vaddq_f32(next_m2, eps_vec);
                    let rsq_est = vrsqrteq_f32(denom_sqr);
                    let rsq_step = vrsqrtsq_f32(vmulq_f32(denom_sqr, rsq_est), rsq_est);
                    let inv_denom = vmulq_f32(rsq_est, rsq_step);

                    let upd = vmulq_f32(next_m1, inv_denom);
                    let next_p = vfmsq_f32(vmulq_f32(decay_vec, p), upd, scale_lr);

                    if self.streaming {
                        core::arch::asm!(
                            "stnp {0:q}, {0:q}, [{1}]",
                            in(vreg) next_p,
                            in(reg) params.add(off),
                            options(nostack)
                        );
                    } else {
                        vst1q_f32(params.add(off), next_p);
                    }
                }

                idx += step_size;
            }

            // Remainder scalar loop
            while idx < length {
                let g = *grads.add(idx);
                let p = *params.add(idx);
                let cur_m1 = *m1.add(idx);
                let cur_m2 = *m2.add(idx);

                let next_m1 = b1 * cur_m1 + (1.0 - b1) * g;
                let next_m2 = b2 * cur_m2 + (1.0 - b2) * g * g;

                *m1.add(idx) = next_m1;
                *m2.add(idx) = next_m2;

                let upd = next_m1 / (next_m2.sqrt() + eps);
                *params.add(idx) = p * (1.0 - lr * decay) - lr * upd;
                idx += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_kernel_execution() {
        let kernel = JitKernel::new(2, 64, false);
        let mut params = vec![1.0f32; 16];
        let grads = vec![0.1f32; 16];
        let mut m1 = vec![0.0f32; 16];
        let mut m2 = vec![0.0f32; 16];

        unsafe {
            kernel.execute(
                params.as_mut_ptr(),
                grads.as_ptr(),
                m1.as_mut_ptr(),
                m2.as_mut_ptr(),
                16,
                1e-3,
                0.9,
                0.999,
                1e-8,
                0.01,
            );
        }

        assert!(params[0] < 1.0);
        assert!(m1[0] > 0.0);
        assert!(m2[0] > 0.0);
    }
}
