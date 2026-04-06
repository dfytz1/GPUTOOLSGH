#include <metal_stdlib>
using namespace metal;

// Float atomic add via uint bit reinterpretation (CAS loop); Av_u must be zero-initialized.
inline void atomic_add_float(device atomic_uint* slot, float v)
{
    uint u_old = atomic_load_explicit(slot, memory_order_relaxed);
    for (;;) {
        float f_old = as_type<float>(u_old);
        uint u_new = as_type<uint>(f_old + v);
        uint u_cmp = u_old;
        if (atomic_compare_exchange_weak_explicit(
                slot, &u_cmp, u_new, memory_order_relaxed, memory_order_relaxed))
            break;
        u_old = u_cmp;
    }
}

// One thread per element; scatter-add into global Av (uint buffer, zero before dispatch).
kernel void fem_matvec(
    device const float* Ke [[buffer(0)]],
    device const int* dofMap [[buffer(1)]],
    device const float* rho [[buffer(2)]],
    device const float* v [[buffer(3)]],
    device atomic_uint* Av_u [[buffer(4)]],
    device const int* nElem [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= *nElem)
        return;

    int base = (int)gid * 24 * 24;
    float r = rho[gid];

    float ve[24];
    for (int b = 0; b < 24; b++)
        ve[b] = v[dofMap[gid * 24 + b]];

    for (int a = 0; a < 24; a++) {
        float s = 0.f;
        for (int b = 0; b < 24; b++)
            s += Ke[base + a * 24 + b] * ve[b];
        atomic_add_float(&Av_u[dofMap[gid * 24 + a]], r * s);
    }
}

// One thread per global DOF: add penalty * v once per fixed DOF (matches CPU MatVec).
kernel void fem_apply_fixed_penalty(
    device atomic_uint* Av_u [[buffer(0)]],
    device const uchar* fixedMask [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* penaltyVal [[buffer(3)]],
    device const int* ndofBuf [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    int n = *ndofBuf;
    if ((int)gid >= n)
        return;
    if (!fixedMask[gid])
        return;
    atomic_add_float(&Av_u[gid], (*penaltyVal) * v[gid]);
}
