#include <metal_stdlib>
using namespace metal;

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

kernel void mg_jacobi_update(
    device float* x [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* Ax [[buffer(2)]],
    device const float* diag [[buffer(3)]],
    device const float* omega [[buffer(4)]],
    device const int* n [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= *n)
        return;
    float d = diag[gid];
    if (d < 1e-30f)
        return;
    x[gid] += (*omega) * (b[gid] - Ax[gid]) / d;
}

kernel void mg_residual(
    device float* r [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* Ax [[buffer(2)]],
    device const int* n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= *n)
        return;
    r[gid] = b[gid] - Ax[gid];
}

kernel void mg_restrict(
    device const float* r_fine [[buffer(0)]],
    device atomic_uint* rhs_coarse_u [[buffer(1)]],
    device const int* prolongCoarse [[buffer(2)]],
    device const float* prolongW [[buffer(3)]],
    device const int* nFineDof [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= *nFineDof)
        return;
    float val = r_fine[gid];
    for (int k = 0; k < 8; k++) {
        int cdof = prolongCoarse[gid * 8 + k];
        if (cdof < 0)
            continue;
        atomic_add_float(&rhs_coarse_u[cdof], prolongW[gid * 8 + k] * val);
    }
}

kernel void mg_prolongate(
    device float* x_fine [[buffer(0)]],
    device const float* x_coarse [[buffer(1)]],
    device const int* prolongCoarse [[buffer(2)]],
    device const float* prolongW [[buffer(3)]],
    device const int* nFineDof [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= *nFineDof)
        return;
    float corr = 0.f;
    for (int k = 0; k < 8; k++) {
        int cdof = prolongCoarse[gid * 8 + k];
        if (cdof < 0)
            continue;
        corr += prolongW[gid * 8 + k] * x_coarse[cdof];
    }
    x_fine[gid] += corr;
}

kernel void mg_zero(
    device float* buf [[buffer(0)]],
    device const int* n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= *n)
        return;
    buf[gid] = 0.f;
}

kernel void mg_zero_uint(
    device atomic_uint* buf [[buffer(0)]],
    device const int* n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= *n)
        return;
    atomic_store_explicit(&buf[gid], 0u, memory_order_relaxed);
}

kernel void pcg_axpy_mg(
    device float* out [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* y [[buffer(2)]],
    device const float* a [[buffer(3)]],
    device const float* b [[buffer(4)]],
    device const int* n [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= *n)
        return;
    out[gid] = (*a) * x[gid] + (*b) * y[gid];
}

kernel void pcg_precond_mg(
    device float* z [[buffer(0)]],
    device const float* r [[buffer(1)]],
    device const float* diag [[buffer(2)]],
    device const int* n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= *n)
        return;
    float d = diag[gid];
    z[gid] = (d > 1e-30f) ? r[gid] / d : r[gid];
}

kernel void mg_pcg_dot_partial(
    device const float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* partials [[buffer(2)]],
    device const int* n [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    threadgroup float* sdata [[threadgroup(0)]])
{
    int i = (int)gid;
    sdata[lid] = (i < *n) ? x[i] * y[i] : 0.f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = lsize / 2; s > 0; s >>= 1) {
        if (lid < s)
            sdata[lid] += sdata[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0)
        partials[gid / lsize] = sdata[0];
}

kernel void mg_pcg_reduce_level(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    device const int* nIn [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]],
    threadgroup float* sdata [[threadgroup(0)]])
{
    uint gidx = gid / tpg;
    int base = (int)(gidx * tpg);
    int ii = base + (int)lid;
    int n = *nIn;
    sdata[lid] = (ii < n) ? in[ii] : 0.f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (lid < s)
            sdata[lid] += sdata[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0)
        out[gidx] = sdata[0];
}

kernel void mg_pcg_reduce(
    device const float* partials [[buffer(0)]],
    device float* result [[buffer(1)]],
    device const int* n [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    threadgroup float* sdata [[threadgroup(0)]])
{
    sdata[lid] = ((int)gid < *n) ? partials[gid] : 0.f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = lsize / 2; s > 0; s >>= 1) {
        if (lid < s)
            sdata[lid] += sdata[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0)
        result[0] = sdata[0];
}
