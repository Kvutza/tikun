#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "aarch64")]
use core::arch::asm;

pub struct AsmEngine;

impl AsmEngine {
    /// Executes the real, hand-tuned ARM64 assembly kernel across raw pointers
    #[cfg(target_arch = "aarch64")]
    pub unsafe fn step_adamw_asm(
        param_ptr: *mut f32,
        grad_ptr: *const f32,
        m1_ptr: *mut f32,
        m2_ptr: *mut f32,
        num_elements: usize,
        step_count: usize,
        lr: f32,
        b1: f32,
        b2: f32,
        eps: f32,
        decay: f32,
        clip_scale: f32,
    ) {
        let b1_pow = b1.powi(step_count as i32);
        let b2_pow = b2.powi(step_count as i32);
        let bias_corr1 = (1.0 - b1_pow).max(1e-8);
        let bias_corr2 = (1.0 - b2_pow).max(1e-8);

        let decay_factor = 1.0 - lr * decay;
        let one_minus_b1 = (1.0 - b1) * clip_scale;
        let one_minus_b2 = (1.0 - b2) * (clip_scale * clip_scale);

        let mut p = param_ptr;
        let mut g = grad_ptr;
        let mut m1 = m1_ptr;
        let mut m2 = m2_ptr;

        let num_blocks = num_elements / 16;
        let remainder = num_elements % 16;

        for _ in 0..num_blocks {
            asm!(
                // 1. Dual-line cache prefetch
                "prfm pldl1keep, [{g}, #128]",
                "prfm pldl1keep, [{p}, #128]",

                // 2. Load 16 floats (64 bytes) per array into hardware registers
                "ld1 {{v0.4s, v1.4s, v2.4s, v3.4s}}, [{p}]",
                "ld1 {{v4.4s, v5.4s, v6.4s, v7.4s}}, [{g}]",
                "ld1 {{v8.4s, v9.4s, v10.4s, v11.4s}}, [{m1}]",
                "ld1 {{v12.4s, v13.4s, v14.4s, v15.4s}}, [{m2}]",

                // 3. Momentum EMA: M1 = Beta1 * M1 + (1-Beta1)*G
                "fmul v8.4s, v8.4s, {v_b1:v}.4s",
                "fmla v8.4s, v4.4s, {v_1mb1:v}.4s",
                "fmul v9.4s, v9.4s, {v_b1:v}.4s",
                "fmla v9.4s, v5.4s, {v_1mb1:v}.4s",
                "fmul v10.4s, v10.4s, {v_b1:v}.4s",
                "fmla v10.4s, v6.4s, {v_1mb1:v}.4s",
                "fmul v11.4s, v11.4s, {v_b1:v}.4s",
                "fmla v11.4s, v7.4s, {v_1mb1:v}.4s",

                // 4. Variance EMA: M2 = Beta2 * M2 + (1-Beta2)*(G*G)
                "fmul v4.4s, v4.4s, v4.4s",
                "fmul v12.4s, v12.4s, {v_b2:v}.4s",
                "fmla v12.4s, v4.4s, {v_1mb2:v}.4s",
                "fmul v5.4s, v5.4s, v5.4s",
                "fmul v13.4s, v13.4s, {v_b2:v}.4s",
                "fmla v13.4s, v5.4s, {v_1mb2:v}.4s",
                "fmul v6.4s, v6.4s, v6.4s",
                "fmul v14.4s, v14.4s, {v_b2:v}.4s",
                "fmla v14.4s, v6.4s, {v_1mb2:v}.4s",
                "fmul v7.4s, v7.4s, v7.4s",
                "fmul v15.4s, v15.4s, {v_b2:v}.4s",
                "fmla v15.4s, v7.4s, {v_1mb2:v}.4s",

                // 5. Store updated M1 and M2 back to memory
                "st1 {{v8.4s, v9.4s, v10.4s, v11.4s}}, [{m1}], #64",
                "st1 {{v12.4s, v13.4s, v14.4s, v15.4s}}, [{m2}], #64",

                // 6. Bias Correction Scaling: M_hat = M1 / bias_corr1, V_hat = M2 / bias_corr2
                "fdiv v8.4s,  v8.4s,  {v_bc1:v}.4s",
                "fdiv v9.4s,  v9.4s,  {v_bc1:v}.4s",
                "fdiv v10.4s, v10.4s, {v_bc1:v}.4s",
                "fdiv v11.4s, v11.4s, {v_bc1:v}.4s",

                "fdiv v12.4s, v12.4s, {v_bc2:v}.4s",
                "fdiv v13.4s, v13.4s, {v_bc2:v}.4s",
                "fdiv v14.4s, v14.4s, {v_bc2:v}.4s",
                "fdiv v15.4s, v15.4s, {v_bc2:v}.4s",

                // 7. Hardware Square Root: denom = sqrt(V_hat) + eps
                "fsqrt v16.4s, v12.4s",
                "fadd  v16.4s, v16.4s, {v_eps:v}.4s",
                "fsqrt v17.4s, v13.4s",
                "fadd  v17.4s, v17.4s, {v_eps:v}.4s",
                "fsqrt v18.4s, v14.4s",
                "fadd  v18.4s, v18.4s, {v_eps:v}.4s",
                "fsqrt v19.4s, v15.4s",
                "fadd  v19.4s, v19.4s, {v_eps:v}.4s",

                // 8. Update: step = M_hat / denom
                "fdiv v8.4s,  v8.4s,  v16.4s",
                "fdiv v9.4s,  v9.4s,  v17.4s",
                "fdiv v10.4s, v10.4s, v18.4s",
                "fdiv v11.4s, v11.4s, v19.4s",

                // 9. Weight Decay + Parameter Step
                "fmul v0.4s, v0.4s, {v_decay:v}.4s",
                "fmls v0.4s, v8.4s,  {v_lr:v}.4s",

                "fmul v1.4s, v1.4s, {v_decay:v}.4s",
                "fmls v1.4s, v9.4s,  {v_lr:v}.4s",

                "fmul v2.4s, v2.4s, {v_decay:v}.4s",
                "fmls v2.4s, v10.4s, {v_lr:v}.4s",

                "fmul v3.4s, v3.4s, {v_decay:v}.4s",
                "fmls v3.4s, v11.4s, {v_lr:v}.4s",

                // 10. Write updated parameters back to memory
                "st1 {{v0.4s, v1.4s, v2.4s, v3.4s}}, [{p}], #64",
                "add {g}, {g}, #64",

                p = inout(reg) p,
                g = inout(reg) g,
                m1 = inout(reg) m1,
                m2 = inout(reg) m2,
                v_b1 = in(vreg) vdupq_n_f32(b1),
                v_b2 = in(vreg) vdupq_n_f32(b2),
                v_1mb1 = in(vreg) vdupq_n_f32(one_minus_b1),
                v_1mb2 = in(vreg) vdupq_n_f32(one_minus_b2),
                v_bc1 = in(vreg) vdupq_n_f32(bias_corr1),
                v_bc2 = in(vreg) vdupq_n_f32(bias_corr2),
                v_eps = in(vreg) vdupq_n_f32(eps),
                v_lr = in(vreg) vdupq_n_f32(lr),
                v_decay = in(vreg) vdupq_n_f32(decay_factor),
                out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                out("v8") _, out("v9") _, out("v10") _, out("v11") _,
                out("v12") _, out("v13") _, out("v14") _, out("v15") _,
                out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            );
        }

        // Remainder scalar loop
        for i in 0..remainder {
            let g_val = *g.add(i) * clip_scale;
            let m1_val = b1 * *m1.add(i) + (1.0 - b1) * g_val;
            let m2_val = b2 * *m2.add(i) + (1.0 - b2) * (g_val * g_val);
            *m1.add(i) = m1_val;
            *m2.add(i) = m2_val;

            let m_hat = m1_val / bias_corr1;
            let v_hat = m2_val / bias_corr2;
            let update = m_hat / (v_hat.sqrt() + eps);

            let p_val = *p.add(i) * decay_factor - lr * update;
            *p.add(i) = p_val;
        }
    }
}
