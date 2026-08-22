#include <metal_stdlib>
using namespace metal;

kernel void adamw_kernel(
    device float* params [[buffer(0)]],
    device const float* grads [[buffer(1)]],
    device float* m [[buffer(2)]],
    device float* v [[buffer(3)]],
    constant float& lr [[buffer(4)]],
    constant float& beta1 [[buffer(5)]],
    constant float& beta2 [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    constant float& weight_decay [[buffer(8)]],
    uint id [[thread_position_in_grid]]
) {
    float g = grads[id];
    m[id] = beta1 * m[id] + (1.0f - beta1) * g;
    v[id] = beta2 * v[id] + (1.0f - beta2) * g * g;

    float m_hat = m[id] / (1.0f - beta1);
    float v_hat = v[id] / (1.0f - beta2);

    float update = m_hat / (sqrt(v_hat) + eps) + weight_decay * params[id];
    params[id] -= lr * update;
}
