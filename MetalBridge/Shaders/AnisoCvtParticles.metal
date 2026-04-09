#include <metal_stdlib>
using namespace metal;

#ifndef ANISO_CVT_DEG_MAX
#define ANISO_CVT_DEG_MAX 64
#endif

// --- Per-topology-vertex anisotropic metric (9 floats row-major 3x3 symmetric stored full) ---

kernel void aniso_metric_topo_kernel(
    device const float* px [[buffer(0)]],
    device const float* py [[buffer(1)]],
    device const float* pz [[buffer(2)]],
    device const float* nx [[buffer(3)]],
    device const float* ny [[buffer(4)]],
    device const float* nz [[buffer(5)]],
    device const int* adjFlat [[buffer(6)]],
    device const float* cotW [[buffer(7)]],
    device const int* rowOff [[buffer(8)]],
    device const float* mixedArea [[buffer(9)]],
    device const float* angleSum [[buffer(10)]],
    device const uchar* isBoundary [[buffer(11)]],
    constant float& alpha [[buffer(12)]],
    constant int& nTopo [[buffer(13)]],
    device float* outM [[buffer(14)]],
    uint gid [[thread_position_in_grid]])
{
    int i = (int)gid;
    if (i >= nTopo)
        return;

    float3 pi = float3(px[i], py[i], pz[i]);
    float3 lap = float3(0.0f);
    int a0 = rowOff[i];
    int a1 = rowOff[i + 1];
    for (int k = a0; k < a1; k++) {
        int j = adjFlat[k];
        float w = cotW[k];
        float3 pj = float3(px[j], py[j], pz[j]);
        lap += w * (pi - pj);
    }

    float A = mixedArea[i];
    if (A < 1e-30f)
        A = 1e-30f;

    float Hmag = length(lap) / (2.0f * A);
    float sumAng = angleSum[i];
    float defect = isBoundary[i] ? (float(M_PI_F) - sumAng) : (2.0f * float(M_PI_F) - sumAng);
    float gauss = defect / A;
    float disc = Hmag * Hmag - gauss;
    if (disc < 0.0f)
        disc = 0.0f;
    float s = sqrt(disc);
    float k1 = Hmag - s;
    float k2 = Hmag + s;

    float3 nrm = float3(nx[i], ny[i], nz[i]);
    if (length_squared(nrm) < 1e-20f)
        nrm = float3(0.0f, 0.0f, 1.0f);
    else
        nrm = normalize(nrm);
    if (dot(nrm, lap) < 0.0f)
        nrm = -nrm;

    float3 t1 = float3(0.0f);
    for (int k = a0; k < a1; k++) {
        int j = adjFlat[k];
        float3 e = float3(px[j], py[j], pz[j]) - pi;
        float3 tg = e - dot(e, nrm) * nrm;
        if (length_squared(tg) > 1e-20f) {
            t1 = normalize(tg);
            break;
        }
    }
    if (length_squared(t1) < 1e-20f) {
        float3 ax = float3(1.0f, 0.0f, 0.0f);
        float3 tg = ax - dot(ax, nrm) * nrm;
        if (length_squared(tg) < 1e-20f)
            tg = float3(0.0f, 1.0f, 0.0f) - dot(float3(0.0f, 1.0f, 0.0f), nrm) * nrm;
        t1 = length_squared(tg) > 1e-20f ? normalize(tg) : float3(1.0f, 0.0f, 0.0f);
    }
    float3 t2 = normalize(cross(nrm, t1));

    float f1 = 1.0f + alpha * fabs(k1);
    float f2 = 1.0f + alpha * fabs(k2);

    float3x3 R = float3x3(t1, t2, nrm);
    float3x3 D = float3x3(float3(f1, 0.0f, 0.0f), float3(0.0f, f2, 0.0f), float3(0.0f, 0.0f, 1.0f));
    float3x3 Mm = R * D * transpose(R);

    int o = i * 9;
    outM[o + 0] = Mm[0][0];
    outM[o + 1] = Mm[0][1];
    outM[o + 2] = Mm[0][2];
    outM[o + 3] = Mm[1][0];
    outM[o + 4] = Mm[1][1];
    outM[o + 5] = Mm[1][2];
    outM[o + 6] = Mm[2][0];
    outM[o + 7] = Mm[2][1];
    outM[o + 8] = Mm[2][2];
}

// --- Spatial hash ---

static inline int3 cell_coord(float3 p, float3 bbMin, float invCell, int3 dims)
{
    int ix = (int)floor((p.x - bbMin.x) * invCell);
    int iy = (int)floor((p.y - bbMin.y) * invCell);
    int iz = (int)floor((p.z - bbMin.z) * invCell);
    ix = clamp(ix, 0, max(dims.x - 1, 0));
    iy = clamp(iy, 0, max(dims.y - 1, 0));
    iz = clamp(iz, 0, max(dims.z - 1, 0));
    return int3(ix, iy, iz);
}

static inline int cell_index(int3 c, int3 dims)
{
    return c.x + dims.x * (c.y + dims.y * c.z);
}

kernel void ac_clear_atomic_int_kernel(
    device atomic_int* buf [[buffer(0)]],
    constant int& n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    int gi = (int)gid;
    if (gi < n)
        atomic_store_explicit(buf + gi, 0, memory_order_relaxed);
}

kernel void ac_count_cells_kernel(
    device const float* posX [[buffer(0)]],
    device const float* posY [[buffer(1)]],
    device const float* posZ [[buffer(2)]],
    device atomic_int* cellCounts [[buffer(3)]],
    constant int& np [[buffer(4)]],
    constant float& bbMinX [[buffer(5)]],
    constant float& bbMinY [[buffer(6)]],
    constant float& bbMinZ [[buffer(7)]],
    constant float& invCell [[buffer(8)]],
    constant int& dimX [[buffer(9)]],
    constant int& dimY [[buffer(10)]],
    constant int& dimZ [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    int i = (int)gid;
    if (i >= np)
        return;
    float3 p = float3(posX[i], posY[i], posZ[i]);
    int3 dims = int3(dimX, dimY, dimZ);
    float3 bbMin = float3(bbMinX, bbMinY, bbMinZ);
    int3 c = cell_coord(p, bbMin, invCell, dims);
    int ci = cell_index(c, dims);
    atomic_fetch_add_explicit(cellCounts + ci, 1, memory_order_relaxed);
}

kernel void ac_scatter_kernel(
    device const float* posX [[buffer(0)]],
    device const float* posY [[buffer(1)]],
    device const float* posZ [[buffer(2)]],
    device atomic_int* cellHead [[buffer(3)]],
    device int* sortedParticle [[buffer(4)]],
    constant int& np [[buffer(5)]],
    constant float& bbMinX [[buffer(6)]],
    constant float& bbMinY [[buffer(7)]],
    constant float& bbMinZ [[buffer(8)]],
    constant float& invCell [[buffer(9)]],
    constant int& dimX [[buffer(10)]],
    constant int& dimY [[buffer(11)]],
    constant int& dimZ [[buffer(12)]],
    uint gid [[thread_position_in_grid]])
{
    int i = (int)gid;
    if (i >= np)
        return;
    float3 p = float3(posX[i], posY[i], posZ[i]);
    int3 dims = int3(dimX, dimY, dimZ);
    float3 bbMin = float3(bbMinX, bbMinY, bbMinZ);
    int3 c = cell_coord(p, bbMin, invCell, dims);
    int ci = cell_index(c, dims);
    int slot = atomic_fetch_add_explicit(cellHead + ci, 1, memory_order_relaxed);
    sortedParticle[slot] = i;
}

static inline float3 sym3_mul(const thread float* M, float3 v)
{
    float x = M[0] * v.x + M[1] * v.y + M[2] * v.z;
    float y = M[3] * v.x + M[4] * v.y + M[5] * v.z;
    float z = M[6] * v.x + M[7] * v.y + M[8] * v.z;
    return float3(x, y, z);
}

kernel void ac_aniso_repulse_kernel(
    device const float* posX [[buffer(0)]],
    device const float* posY [[buffer(1)]],
    device const float* posZ [[buffer(2)]],
    device float* outX [[buffer(3)]],
    device float* outY [[buffer(4)]],
    device float* outZ [[buffer(5)]],
    device const float* metric9 [[buffer(6)]],
    device const uchar* fixedMask [[buffer(7)]],
    device const int* cellStart [[buffer(8)]],
    device const int* cellPop [[buffer(9)]],
    device const int* sortedParticle [[buffer(10)]],
    constant int& np [[buffer(11)]],
    constant float& bbMinX [[buffer(12)]],
    constant float& bbMinY [[buffer(13)]],
    constant float& bbMinZ [[buffer(14)]],
    constant float& invCell [[buffer(15)]],
    constant int& dimX [[buffer(16)]],
    constant int& dimY [[buffer(17)]],
    constant int& dimZ [[buffer(18)]],
    constant float& targetSpacing [[buffer(19)]],
    constant float& repulsionStrength [[buffer(20)]],
    constant float& epsilon [[buffer(21)]],
    uint gid [[thread_position_in_grid]])
{
    int i = (int)gid;
    if (i >= np)
        return;

    if (fixedMask[i]) {
        outX[i] = posX[i];
        outY[i] = posY[i];
        outZ[i] = posZ[i];
        return;
    }

    float3 pi = float3(posX[i], posY[i], posZ[i]);
    int3 dims = int3(dimX, dimY, dimZ);
    float3 bbMin = float3(bbMinX, bbMinY, bbMinZ);
    int3 ci = cell_coord(pi, bbMin, invCell, dims);

    float dxi = 0.0f, dyi = 0.0f, dzi = 0.0f;
    device const float* Mi = metric9 + i * 9;

    for (int oz = -1; oz <= 1; oz++) {
        for (int oy = -1; oy <= 1; oy++) {
            for (int ox = -1; ox <= 1; ox++) {
                int3 cc = ci + int3(ox, oy, oz);
                if (cc.x < 0 || cc.y < 0 || cc.z < 0 || cc.x >= dims.x || cc.y >= dims.y || cc.z >= dims.z)
                    continue;
                int cidx = cell_index(cc, dims);
                int start = cellStart[cidx];
                int pop = cellPop[cidx];
                for (int t = 0; t < pop; t++) {
                    int j = sortedParticle[start + t];
                    if (j == i)
                        continue;
                    float3 pj = float3(posX[j], posY[j], posZ[j]);
                    float3 diff = pi - pj;
                    device const float* Mj = metric9 + j * 9;
                    float Mavg[9];
                    for (int k = 0; k < 9; k++)
                        Mavg[k] = 0.5f * (Mi[k] + Mj[k]);
                    float3 t1 = sym3_mul(Mavg, diff);
                    float dm2 = dot(diff, t1);
                    if (dm2 <= epsilon * epsilon)
                        continue;
                    float dM = sqrt(dm2);
                    if (dM < targetSpacing && dM > epsilon) {
                        float s = repulsionStrength * (1.0f - dM / targetSpacing) / dM;
                        dxi += s * diff.x;
                        dyi += s * diff.y;
                        dzi += s * diff.z;
                    }
                }
            }
        }
    }

    outX[i] = posX[i] + dxi;
    outY[i] = posY[i] + dyi;
    outZ[i] = posZ[i] + dzi;
}

kernel void ac_build_adj_deg_kernel(
    device const float* posX [[buffer(0)]],
    device const float* posY [[buffer(1)]],
    device const float* posZ [[buffer(2)]],
    device int* adjFlat [[buffer(3)]],
    device int* degOut [[buffer(4)]],
    device const int* cellStart [[buffer(5)]],
    device const int* cellPop [[buffer(6)]],
    device const int* sortedParticle [[buffer(7)]],
    constant int& np [[buffer(8)]],
    constant float& bbMinX [[buffer(9)]],
    constant float& bbMinY [[buffer(10)]],
    constant float& bbMinZ [[buffer(11)]],
    constant float& invCell [[buffer(12)]],
    constant int& dimX [[buffer(13)]],
    constant int& dimY [[buffer(14)]],
    constant int& dimZ [[buffer(15)]],
    constant float& neighRadius [[buffer(16)]],
    constant int& degMax [[buffer(17)]],
    uint gid [[thread_position_in_grid]])
{
    int i = (int)gid;
    if (i >= np)
        return;

    float3 pi = float3(posX[i], posY[i], posZ[i]);
    int3 dims = int3(dimX, dimY, dimZ);
    float3 bbMin = float3(bbMinX, bbMinY, bbMinZ);
    int3 ci = cell_coord(pi, bbMin, invCell, dims);
    float r2 = neighRadius * neighRadius;
    int base = i * degMax;
    int cnt = 0;

    for (int oz = -1; oz <= 1; oz++) {
        for (int oy = -1; oy <= 1; oy++) {
            for (int ox = -1; ox <= 1; ox++) {
                int3 cc = ci + int3(ox, oy, oz);
                if (cc.x < 0 || cc.y < 0 || cc.z < 0 || cc.x >= dims.x || cc.y >= dims.y || cc.z >= dims.z)
                    continue;
                int cidx = cell_index(cc, dims);
                int start = cellStart[cidx];
                int pop = cellPop[cidx];
                for (int t = 0; t < pop; t++) {
                    int j = sortedParticle[start + t];
                    if (j == i)
                        continue;
                    float3 pj = float3(posX[j], posY[j], posZ[j]);
                    float d2 = distance_squared(pi, pj);
                    if (d2 <= r2 && d2 > 0.0f) {
                        if (cnt < degMax) {
                            adjFlat[base + cnt] = j;
                            cnt++;
                        }
                    }
                }
            }
        }
    }
    for (int k = cnt; k < degMax; k++)
        adjFlat[base + k] = -1;
    degOut[i] = cnt;
}

kernel void ac_laplacian_constrained_deg_kernel(
    device const float* inX [[buffer(0)]],
    device const float* inY [[buffer(1)]],
    device const float* inZ [[buffer(2)]],
    device float* outX [[buffer(3)]],
    device float* outY [[buffer(4)]],
    device float* outZ [[buffer(5)]],
    device const int* adjFlat [[buffer(6)]],
    device const int* degArr [[buffer(7)]],
    constant int& np [[buffer(8)]],
    constant float& strength [[buffer(9)]],
    constant int& degMax [[buffer(10)]],
    device const uchar* fixedMask [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    int i = (int)gid;
    if (i >= np)
        return;

    if (fixedMask[i]) {
        outX[i] = inX[i];
        outY[i] = inY[i];
        outZ[i] = inZ[i];
        return;
    }

    int deg = degArr[i];
    int base = i * degMax;
    if (deg <= 0) {
        outX[i] = inX[i];
        outY[i] = inY[i];
        outZ[i] = inZ[i];
        return;
    }

    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    int cnt = 0;
    for (int k = 0; k < deg; k++) {
        int j = adjFlat[base + k];
        if (j < 0)
            break;
        ax += inX[j];
        ay += inY[j];
        az += inZ[j];
        cnt++;
    }
    if (cnt <= 0) {
        outX[i] = inX[i];
        outY[i] = inY[i];
        outZ[i] = inZ[i];
        return;
    }
    ax /= (float)cnt;
    ay /= (float)cnt;
    az /= (float)cnt;
    float s = strength;
    outX[i] = inX[i] + s * (ax - inX[i]);
    outY[i] = inY[i] + s * (ay - inY[i]);
    outZ[i] = inZ[i] + s * (az - inZ[i]);
}

kernel void ac_copy_xyz_kernel(
    device const float* inX [[buffer(0)]],
    device const float* inY [[buffer(1)]],
    device const float* inZ [[buffer(2)]],
    device float* outX [[buffer(3)]],
    device float* outY [[buffer(4)]],
    device float* outZ [[buffer(5)]],
    constant int& np [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    int i = (int)gid;
    if (i >= np)
        return;
    outX[i] = inX[i];
    outY[i] = inY[i];
    outZ[i] = inZ[i];
}

/// Closest point on segment ab to p; returns squared distance.
static float dist_sq_point_segment(float3 p, float3 a, float3 b, thread float3* closest)
{
    float3 ab = b - a;
    float t = dot(p - a, ab) / max(dot(ab, ab), 1e-30f);
    t = clamp(t, 0.0f, 1.0f);
    *closest = a + t * ab;
    return distance_squared(p, *closest);
}

kernel void ac_project_boundary_segments_kernel(
    device float* posX [[buffer(0)]],
    device float* posY [[buffer(1)]],
    device float* posZ [[buffer(2)]],
    device const uchar* boundaryParticle [[buffer(3)]],
    device const float* segAx [[buffer(4)]],
    device const float* segAy [[buffer(5)]],
    device const float* segAz [[buffer(6)]],
    device const float* segBx [[buffer(7)]],
    device const float* segBy [[buffer(8)]],
    device const float* segBz [[buffer(9)]],
    constant int& np [[buffer(10)]],
    constant int& nSeg [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    int i = (int)gid;
    if (i >= np)
        return;
    if (!boundaryParticle[i])
        return;

    float3 p = float3(posX[i], posY[i], posZ[i]);
    float best = INFINITY;
    float3 bestC = p;
    for (int s = 0; s < nSeg; s++) {
        float3 a = float3(segAx[s], segAy[s], segAz[s]);
        float3 b = float3(segBx[s], segBy[s], segBz[s]);
        float3 cp;
        float d2 = dist_sq_point_segment(p, a, b, &cp);
        if (d2 < best) {
            best = d2;
            bestC = cp;
        }
    }
    posX[i] = bestC.x;
    posY[i] = bestC.y;
    posZ[i] = bestC.z;
}
