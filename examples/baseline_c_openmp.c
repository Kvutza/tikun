#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <arm_neon.h>

void step_adamw_c(
    float* p,
    const float* g,
    float* m1,
    float* m2,
    size_t length,
    float lr,
    float b1,
    float b2,
    float eps,
    float decay
) {
    #pragma omp parallel for
    for (size_t i = 0; i < length; i += 4) {
        float32x4_t p_val = vld1q_f32(&p[i]);
        float32x4_t g_val = vld1q_f32(&g[i]);
        float32x4_t m1_val = vld1q_f32(&m1[i]);
        float32x4_t m2_val = vld1q_f32(&m2[i]);

        float32x4_t next_m1 = vaddq_f32(vmulq_n_f32(m1_val, b1), vmulq_n_f32(g_val, 1.0f - b1));
        float32x4_t next_m2 = vaddq_f32(vmulq_n_f32(m2_val, b2), vmulq_n_f32(vmulq_f32(g_val, g_val), 1.0f - b2));

        float32x4_t m_hat = vmulq_n_f32(next_m1, 1.0f / (1.0f - b1));
        float32x4_t v_hat = vmulq_n_f32(next_m2, 1.0f / (1.0f - b2));

        // Scalar sqrt on vector elements for C baseline
        float m_hat_arr[4], v_hat_arr[4], p_arr[4];
        vst1q_f32(m_hat_arr, m_hat);
        vst1q_f32(v_hat_arr, v_hat);
        vst1q_f32(p_arr, p_val);

        for (int j = 0; j < 4; j++) {
            float update = (m_hat_arr[j] / (sqrtf(v_hat_arr[j]) + eps)) + (decay * p_arr[j]);
            p_arr[j] -= lr * update;
        }

        vst1q_f32(&m1[i], next_m1);
        vst1q_f32(&m2[i], next_m2);
        vst1q_f32(&p[i], vld1q_f32(p_arr));
    }
}

int main() {
    size_t sizes[] = {10000000, 50000000, 100000000};
    const char* names[] = {"10M", "50M", "100M"};

    printf("==========================================================================\n");
    printf("🏎️ Bare-Metal Clang -O3 C Baseline (Handwritten C Kernel)\n");
    printf("==========================================================================\n");

    for (int s = 0; s < 3; s++) {
        size_t n = sizes[s];
        float* p = (float*)aligned_alloc(64, n * sizeof(float));
        float* g = (float*)aligned_alloc(64, n * sizeof(float));
        float* m1 = (float*)calloc(n, sizeof(float));
        float* m2 = (float*)calloc(n, sizeof(float));

        for (size_t i = 0; i < n; i++) {
            p[i] = 1.0f;
            g[i] = 0.1f;
        }

        // Warmup
        step_adamw_c(p, g, m1, m2, n, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        for (int iter = 0; iter < 10; iter++) {
            step_adamw_c(p, g, m1, m2, n, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);
        }
        clock_gettime(CLOCK_MONOTONIC, &end);

        double elapsed_ms = ((end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1000000.0) / 10.0;
        printf("  • %s Parameters (%zu elements): %.2f ms / step\n", names[s], n, elapsed_ms);

        free(p);
        free(g);
        free(m1);
        free(m2);
    }
    return 0;
}
