#include <metal_stdlib>
using namespace metal;

struct GradParams {
    uint nx;
    uint ny;
    uint nz;
    float iDx;
    float iDy;
    float iDz;
};

kernel void gradient_magnitude_3d(
    device const float* phi [[buffer(0)]],
    device const float* inside [[buffer(1)]],
    device float* grad [[buffer(2)]],
    constant GradParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.nx * p.ny * p.nz || inside[gid] < 0.5f) {
        grad[gid] = 0.f;
        return;
    }
    uint i = gid / (p.ny * p.nz);
    uint j = (gid / p.nz) % p.ny;
    uint k = gid % p.nz;
    auto at = [&](uint a, uint b, uint c) { return phi[a * p.ny * p.nz + b * p.nz + c]; };
    float gx = (i > 0 && i < p.nx - 1) ? (at(i + 1, j, k) - at(i - 1, j, k)) * 0.5f * p.iDx : 0.f;
    float gy = (j > 0 && j < p.ny - 1) ? (at(i, j + 1, k) - at(i, j - 1, k)) * 0.5f * p.iDy : 0.f;
    float gz = (k > 0 && k < p.nz - 1) ? (at(i, j, k + 1) - at(i, j, k - 1)) * 0.5f * p.iDz : 0.f;
    grad[gid] = sqrt(gx * gx + gy * gy + gz * gz);
}
