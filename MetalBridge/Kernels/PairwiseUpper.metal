#include <metal_stdlib>
using namespace metal;

/// Upper triangle only (i < j). Pair index k in [0, n*(n-1)/2) uses row-major over j.
static void decode_pair(uint k, int n, thread int& i, thread int& j)
{
    int lo = 0;
    int hi = n - 2;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        int baseMid = mid * (2 * n - mid - 1) / 2;
        if (static_cast<uint>(baseMid) <= k)
            lo = mid;
        else
            hi = mid - 1;
    }
    i = lo;
    int baseI = i * (2 * n - i - 1) / 2;
    j = i + 1 + static_cast<int>(k - static_cast<uint>(baseI));
}

kernel void pairwiseUpperDistSqKernel(
    device const float* px [[buffer(0)]],
    device const float* py [[buffer(1)]],
    device const float* pz [[buffer(2)]],
    constant int& n [[buffer(3)]],
    device float* outDistSq [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (n < 2)
        return;
    uint P = static_cast<uint>(n) * static_cast<uint>(n - 1u) / 2u;
    if (gid >= P)
        return;

    int i, j;
    decode_pair(gid, n, i, j);

    float3 a = float3(px[i], py[i], pz[i]);
    float3 b = float3(px[j], py[j], pz[j]);
    float3 d = a - b;
    outDistSq[gid] = dot(d, d);
}
