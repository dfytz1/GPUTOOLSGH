#include <metal_stdlib>
using namespace metal;

kernel void laplacianConstrainedKernel(
    device const float* inX [[buffer(0)]],
    device const float* inY [[buffer(1)]],
    device const float* inZ [[buffer(2)]],
    device float* outX [[buffer(3)]],
    device float* outY [[buffer(4)]],
    device float* outZ [[buffer(5)]],
    device const int* adjFlat [[buffer(6)]],
    device const int* offsets [[buffer(7)]],
    device const int* vertCount [[buffer(8)]],
    device const float* strength [[buffer(9)]],
    device const uchar* fixed [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    int n = *vertCount;
    if ((int)gid >= n)
        return;

    if (fixed[gid]) {
        outX[gid] = inX[gid];
        outY[gid] = inY[gid];
        outZ[gid] = inZ[gid];
        return;
    }

    int start = offsets[gid];
    int end = offsets[gid + 1];
    int count = end - start;
    if (count == 0) {
        outX[gid] = inX[gid];
        outY[gid] = inY[gid];
        outZ[gid] = inZ[gid];
        return;
    }

    float ax = 0, ay = 0, az = 0;
    for (int i = start; i < end; i++) {
        int nb = adjFlat[i];
        ax += inX[nb];
        ay += inY[nb];
        az += inZ[nb];
    }
    ax /= count;
    ay /= count;
    az /= count;

    float s = *strength;
    outX[gid] = inX[gid] + s * (ax - inX[gid]);
    outY[gid] = inY[gid] + s * (ay - inY[gid]);
    outZ[gid] = inZ[gid] + s * (az - inZ[gid]);
}
