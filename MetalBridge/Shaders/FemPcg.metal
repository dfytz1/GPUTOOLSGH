#include <metal_stdlib>
using namespace metal;

// --- axpy: out[i] = a*x[i] + b*y[i]
kernel void pcg_axpy(
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

// --- z[i] = r[i] / diag[i]
kernel void pcg_precond(
    device float* z [[buffer(0)]],
    device const float* r [[buffer(1)]],
    device const float* diag [[buffer(2)]],
    device const int* n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= *n)
        return;
    float d = diag[gid];
    z[gid] = d > 1e-30f ? r[gid] / d : r[gid];
}

// --- partial dot: one sum per threadgroup
kernel void pcg_dot_partial(
    device const float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* partials [[buffer(2)]],
    device const int* n [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]],
    threadgroup float* sdata [[threadgroup(0)]])
{
    uint gidx = gid / tpg;
    int base = (int)(gidx * tpg);
    int i = base + (int)lid;
    sdata[lid] = (i < *n) ? x[i] * y[i] : 0.f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (lid < s)
            sdata[lid] += sdata[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0)
        partials[gidx] = sdata[0];
}

// --- reduce partial sums (chunked); output count = ceil(nIn / tpg)
kernel void pcg_reduce_level(
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
    int i = base + (int)lid;
    int n = *nIn;
    sdata[lid] = (i < n) ? in[i] : 0.f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (lid < s)
            sdata[lid] += sdata[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0)
        out[gidx] = sdata[0];
}

// --- single-threadgroup tree reduce: partials[0..n-1] -> result[0]; n <= threads per group
kernel void pcg_reduce(
    device const float* partials [[buffer(0)]],
    device float* result [[buffer(1)]],
    device const int* n [[buffer(2)]],
    uint lid [[thread_index_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]],
    threadgroup float* sdata [[threadgroup(0)]])
{
    sdata[lid] = ((int)lid < *n) ? partials[lid] : 0.f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (lid < s)
            sdata[lid] += sdata[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0)
        result[0] = sdata[0];
}

kernel void pcg_copy(
    device float* dst [[buffer(0)]],
    device const float* src [[buffer(1)]],
    device const int* n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= *n)
        return;
    dst[gid] = src[gid];
}

// MatVec accumulates float bits as uint; safe to read as uint after kernel completes.
kernel void pcg_uint_to_float(
    device const uint* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    device const int* n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= *n)
        return;
    dst[gid] = as_type<float>(src[gid]);
}
