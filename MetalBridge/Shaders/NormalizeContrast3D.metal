#include <metal_stdlib>
using namespace metal;

struct NormParams3D { uint n; float dMin, dMax; int invert; float exponent; };

kernel void normalize_contrast_3d(
    device       float* data   [[buffer(0)]],
    device const float* inside [[buffer(1)]],
    constant NormParams3D& p   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n || inside[gid] < 0.5f) return;

    float range = p.dMax - p.dMin;
    float v = range > 1e-30f ? (data[gid] - p.dMin) / range : 0.f;
    v = clamp(v, 0.f, 1.f);
    if (p.invert) v = 1.f - v;
    data[gid] = powr(v, p.exponent);
}
