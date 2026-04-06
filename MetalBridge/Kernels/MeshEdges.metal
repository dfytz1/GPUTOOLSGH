#include <metal_stdlib>
using namespace metal;

kernel void csrDirectedWeightedEdgesKernel(
    device const float* vx [[buffer(0)]],
    device const float* vy [[buffer(1)]],
    device const float* vz [[buffer(2)]],
    device const int* rowOffsets [[buffer(3)]],
    device const int* adjFlat [[buffer(4)]],
    device const int* edgeWriteBase [[buffer(5)]],
    device int* edgeU [[buffer(6)]],
    device int* edgeV [[buffer(7)]],
    device float* edgeW [[buffer(8)]],
    constant int& vertexCount [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
    int v = (int)gid;
    if (v >= vertexCount)
        return;

    int a = rowOffsets[v];
    int b = rowOffsets[v + 1];
    int base = edgeWriteBase[v];
    float x0 = vx[v];
    float y0 = vy[v];
    float z0 = vz[v];
    int o = 0;
    for (int k = a; k < b; k++)
    {
        int u = adjFlat[k];
        float dx = vx[u] - x0;
        float dy = vy[u] - y0;
        float dz = vz[u] - z0;
        float len = sqrt(dx * dx + dy * dy + dz * dz);
        edgeU[base + o] = v;
        edgeV[base + o] = u;
        edgeW[base + o] = len;
        o++;
    }
}
