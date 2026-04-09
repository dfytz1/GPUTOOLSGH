// Mesh–mesh triangle intersection: one thread per triangle pair. Per-thread broad-phase is an
// AABB overlap of the two triangles before the SAT test (no mesh-level BVH).
#include <metal_stdlib>
using namespace metal;

static inline float3 v3(device const float* x, device const float* y, device const float* z, int i)
{
    return float3(x[i], y[i], z[i]);
}

static inline void triVerts(
    device const float* px,
    device const float* py,
    device const float* pz,
    device const int* tri,
    int ti,
    thread float3& a,
    thread float3& b,
    thread float3& c)
{
    int o = ti * 3;
    int i0 = tri[o];
    int i1 = tri[o + 1];
    int i2 = tri[o + 2];
    a = v3(px, py, pz, i0);
    b = v3(px, py, pz, i1);
    c = v3(px, py, pz, i2);
}

static inline void triAabb(float3 a, float3 b, float3 c, thread float3& mn, thread float3& mx)
{
    mn = min(min(a, b), c);
    mx = max(max(a, b), c);
}

static inline bool aabbOverlap(float3 amin, float3 amax, float3 bmin, float3 bmax)
{
    return all(amin <= bmax) && all(bmin <= amax);
}

/// true if intervals [minA,maxA] and [minB,maxB] are disjoint
static inline bool separated1D(float minA, float maxA, float minB, float maxB)
{
    return (maxA < minB) || (maxB < minA);
}

static inline bool separatedByAxis(float3 axis, float3 a0, float3 a1, float3 a2, float3 b0, float3 b1, float3 b2)
{
    float al2 = dot(axis, axis);
    if (al2 < 1e-30f)
        return false;
    float p0 = dot(a0, axis);
    float p1 = dot(a1, axis);
    float p2 = dot(a2, axis);
    float minA = min(min(p0, p1), p2);
    float maxA = max(max(p0, p1), p2);
    float q0 = dot(b0, axis);
    float q1 = dot(b1, axis);
    float q2 = dot(b2, axis);
    float minB = min(min(q0, q1), q2);
    float maxB = max(max(q0, q1), q2);
    return separated1D(minA, maxA, minB, maxB);
}

static bool trianglesIntersectSat(float3 a0, float3 a1, float3 a2, float3 b0, float3 b1, float3 b2)
{
    float3 n1 = cross(a1 - a0, a2 - a0);
    float3 n2 = cross(b1 - b0, b2 - b0);
    if (dot(n1, n1) < 1e-30f || dot(n2, n2) < 1e-30f)
        return false;

    if (separatedByAxis(n1, a0, a1, a2, b0, b1, b2))
        return false;
    if (separatedByAxis(n2, a0, a1, a2, b0, b1, b2))
        return false;

    float3 e1a = a1 - a0;
    float3 e1b = a2 - a1;
    float3 e1c = a0 - a2;
    float3 e2a = b1 - b0;
    float3 e2b = b2 - b1;
    float3 e2c = b0 - b2;

    float3 e1[3] = { e1a, e1b, e1c };
    float3 e2[3] = { e2a, e2b, e2c };

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float3 ax = cross(e1[i], e2[j]);
            if (dot(ax, ax) < 1e-30f)
                continue;
            if (separatedByAxis(ax, a0, a1, a2, b0, b1, b2))
                return false;
        }
    }
    return true;
}

/// Upper-triangle linear index for pairs (i < j), n = triangle count
static void decodeTriPair(uint k, int n, thread int& i, thread int& j)
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

kernel void meshMeshTriangleHitsKernel(
    device const float* ax [[buffer(0)]],
    device const float* ay [[buffer(1)]],
    device const float* az [[buffer(2)]],
    device const int* triA [[buffer(3)]],
    constant int& nVertA [[buffer(4)]],
    constant int& nTriA [[buffer(5)]],
    device const float* bx [[buffer(6)]],
    device const float* by [[buffer(7)]],
    device const float* bz [[buffer(8)]],
    device const int* triB [[buffer(9)]],
    constant int& nVertB [[buffer(10)]],
    constant int& nTriB [[buffer(11)]],
    constant int& selfCollision [[buffer(12)]],
    constant int& skipSameIndex [[buffer(13)]],
    constant int& maxHits [[buffer(14)]],
    device atomic_uint* hitCounter [[buffer(15)]],
    device int* outTriA [[buffer(16)]],
    device int* outTriB [[buffer(17)]],
    uint gid [[thread_position_in_grid]])
{
    int na = nTriA;
    int nb = nTriB;
    if (na < 1 || nb < 1)
        return;

    uint totalPairs;
    int ia;
    int ib;

    if (selfCollision != 0) {
        totalPairs = static_cast<uint>(na) * static_cast<uint>(na - 1) / 2u;
        if (gid >= totalPairs)
            return;
        decodeTriPair(gid, na, ia, ib);
    } else {
        totalPairs = static_cast<uint>(na) * static_cast<uint>(nb);
        if (gid >= totalPairs)
            return;
        ia = static_cast<int>(gid / static_cast<uint>(nb));
        ib = static_cast<int>(gid % static_cast<uint>(nb));
        if (skipSameIndex != 0 && ia == ib)
            return;
    }

    float3 a0, a1, a2;
    float3 b0, b1, b2;
    triVerts(ax, ay, az, triA, ia, a0, a1, a2);
    triVerts(bx, by, bz, triB, ib, b0, b1, b2);

    float3 amin, amax, bmin, bmax;
    triAabb(a0, a1, a2, amin, amax);
    triAabb(b0, b1, b2, bmin, bmax);
    if (!aabbOverlap(amin, amax, bmin, bmax))
        return;

    if (!trianglesIntersectSat(a0, a1, a2, b0, b1, b2))
        return;

    uint slot = atomic_fetch_add_explicit(hitCounter, 1u, memory_order_relaxed);
    int mh = maxHits;
    if (mh > 0 && static_cast<int>(slot) < mh) {
        outTriA[slot] = ia;
        outTriB[slot] = ib;
    }
}
