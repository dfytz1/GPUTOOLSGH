#include <metal_stdlib>
using namespace metal;

kernel void voxel_sample_kernel(
    device const float* ptX [[buffer(0)]],
    device const float* ptY [[buffer(1)]],
    device const float* ptZ [[buffer(2)]],
    device const float* charge [[buffer(3)]],
    device float* grid [[buffer(4)]],
    device const float* bbMin [[buffer(5)]],
    device const float* cellSz [[buffer(6)]],
    device const int* dims [[buffer(7)]],
    device const int* nPoints [[buffer(8)]],
    device const float* range [[buffer(9)]],
    device const int* flags [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    int nx = dims[0], ny = dims[1], nz = dims[2];
    int total = nx * ny * nz;
    if ((int)gid >= total)
        return;

    int iz = (int)gid % nz;
    int iy = ((int)gid / nz) % ny;
    int ix = (int)gid / (ny * nz);

    float cx = bbMin[0] + (ix + 0.5f) * cellSz[0];
    float cy = bbMin[1] + (iy + 0.5f) * cellSz[1];
    float cz = bbMin[2] + (iz + 0.5f) * cellSz[2];

    float r = *range;
    float r2 = r * r;
    int np = *nPoints;
    int lin = flags[0] & 1;
    int bulge = (flags[0] >> 1) & 1;

    float val = 0.f;
    int cnt = 0;
    for (int p = 0; p < np; p++) {
        float dx = cx - ptX[p];
        float dy = cy - ptY[p];
        float dz = cz - ptZ[p];
        float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 > r2)
            continue;
        float d = sqrt(d2);
        float w = lin ? (1.f - d / r) : exp(-3.f * d / r);
        val += charge[p] * w;
        cnt++;
    }
    if (bulge && cnt > 1)
        val *= (float)cnt;
    grid[gid] = val;
}
