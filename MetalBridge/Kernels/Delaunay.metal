#include <metal_stdlib>
using namespace metal;

static int inCircumcircle(float2 a, float2 b, float2 c, float2 d)
{
    float ax = a.x - d.x;
    float ay = a.y - d.y;
    float bx = b.x - d.x;
    float by = b.y - d.y;
    float cx = c.x - d.x;
    float cy = c.y - d.y;
    float det = (ax * ax + ay * ay) * (bx * cy - by * cx) - (bx * bx + by * by) * (ax * cy - ay * cx)
        + (cx * cx + cy * cy) * (ax * by - ay * bx);
    return det > 0.0f ? 1 : 0;
}

kernel void delaunayMarkBadTrisKernel(
    device const float* px [[buffer(0)]],
    device const float* py [[buffer(1)]],
    device const int* triFlat [[buffer(2)]],
    constant float& queryX [[buffer(3)]],
    constant float& queryY [[buffer(4)]],
    device int* outBad [[buffer(5)]],
    constant int& triCount [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    int t = (int)gid;
    if (t >= triCount)
        return;

    int ia = triFlat[3 * t];
    int ib = triFlat[3 * t + 1];
    int ic = triFlat[3 * t + 2];
    float2 pa = float2(px[ia], py[ia]);
    float2 pb = float2(px[ib], py[ib]);
    float2 pc = float2(px[ic], py[ic]);
    float2 q = float2(queryX, queryY);
    float orient = (pb.x - pa.x) * (pc.y - pa.y) - (pb.y - pa.y) * (pc.x - pa.x);
    int bad = 0;
    if (fabs(orient) > 1e-20f)
    {
        if (orient < 0.0f)
            bad = inCircumcircle(pc, pb, pa, q);
        else
            bad = inCircumcircle(pa, pb, pc, q);
    }

    outBad[t] = bad;
}
