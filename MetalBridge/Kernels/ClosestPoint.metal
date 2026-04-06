#include <metal_stdlib>
using namespace metal;

static inline float distSq(float3 u)
{
    return dot(u, u);
}

/// Squared distance from p to triangle abc; writes closest point on triangle.
static float pointTriangleSqDistance(float3 p, float3 a, float3 b, float3 c, thread float3* closest)
{
    float3 ab = b - a;
    float3 ac = c - a;
    float3 ap = p - a;
    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f)
    {
        *closest = a;
        return distSq(p - a);
    }

    float3 bp = p - b;
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3)
    {
        *closest = b;
        return distSq(p - b);
    }

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        float denom = d1 - d3;
        float t = denom > 0.0f ? (d1 / denom) : 0.0f;
        *closest = a + t * ab;
        return distSq(p - *closest);
    }

    float3 cp = p - c;
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6)
    {
        *closest = c;
        return distSq(p - c);
    }

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        float denom = d2 - d6;
        float w = denom > 0.0f ? (d2 / denom) : 0.0f;
        *closest = a + w * ac;
        return distSq(p - *closest);
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        float denom = (d4 - d3) + (d5 - d6);
        float w = denom > 0.0f ? ((d4 - d3) / denom) : 0.0f;
        *closest = b + w * (c - b);
        return distSq(p - *closest);
    }

    float denom = va + vb + vc;
    if (fabs(denom) < 1e-20f)
    {
        *closest = a;
        return distSq(p - a);
    }

    float v = vb / denom;
    float w = vc / denom;
    *closest = a + ab * v + ac * w;
    return distSq(p - *closest);
}

kernel void closestPointMeshKernel(
    device const float* qx [[buffer(0)]],
    device const float* qy [[buffer(1)]],
    device const float* qz [[buffer(2)]],
    device const float* vx [[buffer(3)]],
    device const float* vy [[buffer(4)]],
    device const float* vz [[buffer(5)]],
    device const int* triIdx [[buffer(6)]],
    constant int& queryCount [[buffer(7)]],
    constant int& triangleCount [[buffer(8)]],
    device float* outCx [[buffer(9)]],
    device float* outCy [[buffer(10)]],
    device float* outCz [[buffer(11)]],
    device float* outDistSq [[buffer(12)]],
    device int* outTriIdx [[buffer(13)]],
    uint gid [[thread_position_in_grid]])
{
    int qi = (int)gid;
    if (qi >= queryCount)
        return;

    float3 q = float3(qx[qi], qy[qi], qz[qi]);
    float best = INFINITY;
    float3 bestC = float3(0.0f);
    int bestT = -1;

    for (int t = 0; t < triangleCount; t++)
    {
        int i0 = triIdx[3 * t];
        int i1 = triIdx[3 * t + 1];
        int i2 = triIdx[3 * t + 2];
        float3 a = float3(vx[i0], vy[i0], vz[i0]);
        float3 b = float3(vx[i1], vy[i1], vz[i1]);
        float3 c = float3(vx[i2], vy[i2], vz[i2]);
        float3 cp;
        float d2 = pointTriangleSqDistance(q, a, b, c, &cp);
        if (d2 < best)
        {
            best = d2;
            bestC = cp;
            bestT = t;
        }
    }

    outCx[qi] = bestC.x;
    outCy[qi] = bestC.y;
    outCz[qi] = bestC.z;
    outDistSq[qi] = best;
    outTriIdx[qi] = bestT;
}
