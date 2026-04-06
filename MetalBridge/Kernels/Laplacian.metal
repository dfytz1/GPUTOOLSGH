#include <metal_stdlib>
using namespace metal;

kernel void laplacianSmoothKernel(
    device const float* posXIn [[buffer(0)]],
    device const float* posYIn [[buffer(1)]],
    device const float* posZIn [[buffer(2)]],
    device float* posXOut [[buffer(3)]],
    device float* posYOut [[buffer(4)]],
    device float* posZOut [[buffer(5)]],
    device const int* adjFlat [[buffer(6)]],
    device const int* rowOffsets [[buffer(7)]],
    constant int& vertexCount [[buffer(8)]],
    constant float& strength [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    int v = (int)gid;
    if (v >= vertexCount)
        return;

    int rowStart = rowOffsets[v];
    int rowEnd = rowOffsets[v + 1];
    int cnt = rowEnd - rowStart;
    float px = posXIn[v];
    float py = posYIn[v];
    float pz = posZIn[v];

    if (cnt <= 0)
    {
        posXOut[v] = px;
        posYOut[v] = py;
        posZOut[v] = pz;
        return;
    }

    float sx = 0.0f;
    float sy = 0.0f;
    float sz = 0.0f;
    for (int k = rowStart; k < rowEnd; k++)
    {
        int j = adjFlat[k];
        sx += posXIn[j];
        sy += posYIn[j];
        sz += posZIn[j];
    }

    float inv = 1.0f / (float)cnt;
    float mx = sx * inv;
    float my = sy * inv;
    float mz = sz * inv;
    float s = strength;
    posXOut[v] = px + s * (mx - px);
    posYOut[v] = py + s * (my - py);
    posZOut[v] = pz + s * (mz - pz);
}
