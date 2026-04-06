#include <metal_stdlib>
using namespace metal;

kernel void closestPointCloudKernel(
    device const float* qx [[buffer(0)]],
    device const float* qy [[buffer(1)]],
    device const float* qz [[buffer(2)]],
    device const float* px [[buffer(3)]],
    device const float* py [[buffer(4)]],
    device const float* pz [[buffer(5)]],
    constant int& queryCount [[buffer(6)]],
    constant int& targetCount [[buffer(7)]],
    device float* outCx [[buffer(8)]],
    device float* outCy [[buffer(9)]],
    device float* outCz [[buffer(10)]],
    device float* outDistSq [[buffer(11)]],
    device int* outIndex [[buffer(12)]],
    uint gid [[thread_position_in_grid]])
{
    int qi = (int)gid;
    if (qi >= queryCount)
        return;

    float3 q = float3(qx[qi], qy[qi], qz[qi]);
    float best = INFINITY;
    float3 bestP = float3(0.0f);
    int bestIdx = -1;

    for (int t = 0; t < targetCount; t++)
    {
        float3 p = float3(px[t], py[t], pz[t]);
        float3 d = q - p;
        float d2 = dot(d, d);
        if (d2 < best)
        {
            best = d2;
            bestP = p;
            bestIdx = t;
        }
    }

    outCx[qi] = bestP.x;
    outCy[qi] = bestP.y;
    outCz[qi] = bestP.z;
    outDistSq[qi] = best;
    outIndex[qi] = bestIdx;
}
