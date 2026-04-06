#include <metal_stdlib>
using namespace metal;

struct GradParams3D { uint nx, ny, nz; float iDx, iDy, iDz; };

kernel void gradient_magnitude_3d(
    device const float* phi    [[buffer(0)]],
    device const float* inside [[buffer(1)]],
    device       float* grad   [[buffer(2)]],
    constant GradParams3D& p   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.nx * p.ny * p.nz || inside[gid] < 0.5f) { grad[gid] = 0.f; return; }

    uint i = gid / (p.ny * p.nz);
    uint j = (gid / p.nz) % p.ny;
    uint k = gid % p.nz;

    float gx = (i > 0 && i < p.nx-1) ?
        (phi[(i+1)*p.ny*p.nz + j*p.nz + k] - phi[(i-1)*p.ny*p.nz + j*p.nz + k]) * 0.5f * p.iDx : 0.f;
    float gy = (j > 0 && j < p.ny-1) ?
        (phi[i*p.ny*p.nz + (j+1)*p.nz + k] - phi[i*p.ny*p.nz + (j-1)*p.nz + k]) * 0.5f * p.iDy : 0.f;
    float gz = (k > 0 && k < p.nz-1) ?
        (phi[i*p.ny*p.nz + j*p.nz + (k+1)] - phi[i*p.ny*p.nz + j*p.nz + (k-1)]) * 0.5f * p.iDz : 0.f;

    grad[gid] = sqrt(gx*gx + gy*gy + gz*gz);
}
