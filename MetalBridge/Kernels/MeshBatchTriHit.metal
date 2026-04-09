// One thread per (meshA index, meshB index). Each thread scans triangle pairs for that mesh pair only
// (Cartesian product, or upper-triangle pairs when same packed list and ma == mb).
#include <metal_stdlib>
using namespace metal;

static inline float3 v3(device const float* x, device const float* y, device const float* z, int i)
{
    return float3(x[i], y[i], z[i]);
}

static inline void triVertsGlobal(
    device const float* px,
    device const float* py,
    device const float* pz,
    device const int* tri,
    int globalTriIndex,
    thread float3& a,
    thread float3& b,
    thread float3& c)
{
    int o = globalTriIndex * 3;
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

static inline void testPair(
    device const float* ax,
    device const float* ay,
    device const float* az,
    device const int* triA,
    device const float* bx,
    device const float* by,
    device const float* bz,
    device const int* triB,
    int gta,
    int gtb,
    int ma,
    int mb,
    int localTa,
    int localTb,
    int maxHits,
    device atomic_uint* hitCounter,
    device int* outMeshA,
    device int* outMeshB,
    device int* outTriA,
    device int* outTriB)
{
    float3 a0, a1, a2, b0, b1, b2;
    triVertsGlobal(ax, ay, az, triA, gta, a0, a1, a2);
    triVertsGlobal(bx, by, bz, triB, gtb, b0, b1, b2);

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
        outMeshA[slot] = ma;
        outMeshB[slot] = mb;
        outTriA[slot] = localTa;
        outTriB[slot] = localTb;
    }
}

kernel void meshBatchTriangleHitsKernel(
    device const float* ax [[buffer(0)]],
    device const float* ay [[buffer(1)]],
    device const float* az [[buffer(2)]],
    device const int* triA [[buffer(3)]],
    device const int* meshTriStartA [[buffer(4)]],
    constant int& nMeshA [[buffer(5)]],
    device const float* bx [[buffer(6)]],
    device const float* by [[buffer(7)]],
    device const float* bz [[buffer(8)]],
    device const int* triB [[buffer(9)]],
    device const int* meshTriStartB [[buffer(10)]],
    constant int& nMeshB [[buffer(11)]],
    constant int& samePackedList [[buffer(12)]],
    constant int& skipIntraMeshPair [[buffer(13)]],
    constant int& maxHits [[buffer(14)]],
    device atomic_uint* hitCounter [[buffer(15)]],
    device int* outMeshA [[buffer(16)]],
    device int* outMeshB [[buffer(17)]],
    device int* outTriA [[buffer(18)]],
    device int* outTriB [[buffer(19)]],
    uint gid [[thread_position_in_grid]])
{
    int nA = nMeshA;
    int nB = nMeshB;
    if (nA < 1 || nB < 1)
        return;

    uint gridW = static_cast<uint>(nA) * static_cast<uint>(nB);
    if (gid >= gridW)
        return;

    int ma = static_cast<int>(gid / static_cast<uint>(nB));
    int mb = static_cast<int>(gid % static_cast<uint>(nB));

    if (skipIntraMeshPair != 0 && ma == mb && samePackedList != 0)
        return;

    int t0a = meshTriStartA[ma];
    int t1a = meshTriStartA[ma + 1];
    int t0b = meshTriStartB[mb];
    int t1b = meshTriStartB[mb + 1];
    int na = t1a - t0a;
    int nb = t1b - t0b;

    bool intra = (samePackedList != 0 && ma == mb);

    if (intra) {
        for (int ta = 0; ta < na; ta++) {
            for (int tb = ta + 1; tb < na; tb++) {
                int gta = t0a + ta;
                int gtb = t0a + tb;
                testPair(ax, ay, az, triA, ax, ay, az, triA, gta, gtb, ma, mb, ta, tb, maxHits, hitCounter,
                    outMeshA, outMeshB, outTriA, outTriB);
            }
        }
    } else {
        for (int ta = 0; ta < na; ta++) {
            for (int tb = 0; tb < nb; tb++) {
                int gta = t0a + ta;
                int gtb = t0b + tb;
                testPair(ax, ay, az, triA, bx, by, bz, triB, gta, gtb, ma, mb, ta, tb, maxHits, hitCounter,
                    outMeshA, outMeshB, outTriA, outTriB);
            }
        }
    }
}
