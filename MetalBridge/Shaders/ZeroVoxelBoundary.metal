#include <metal_stdlib>
using namespace metal;

struct BoundaryParams { uint nx, ny, nz; };

kernel void zero_voxel_boundary(
    device float* data           [[buffer(0)]],
    constant BoundaryParams& p   [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.nx * p.ny * p.nz) return;
    uint i = gid / (p.ny * p.nz);
    uint j = (gid / p.nz) % p.ny;
    uint k = gid % p.nz;
    if (i == 0 || i == p.nx-1 || j == 0 || j == p.ny-1 || k == 0 || k == p.nz-1)
        data[gid] = 0.f;
}
