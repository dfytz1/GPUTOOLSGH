#include <metal_stdlib>
using namespace metal;

struct NormParams {
    uint n;
    float dMin;
    float dMax;
    int invert;
    float exp;
};

kernel void normalize_contrast_3d(
    device float* data [[buffer(0)]],
    device const float* inside [[buffer(1)]],
    constant NormParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n || inside[gid] < 0.5f)
        return;
    float r = p.dMax - p.dMin;
    float v = r > 1e-30f ? (data[gid] - p.dMin) / r : 0.f;
    v = clamp(v, 0.f, 1.f);
    if (p.invert != 0)
        v = 1.f - v;
    data[gid] = powr(v, p.exp);
}
