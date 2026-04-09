#include <metal_stdlib>
using namespace metal;

/// Per Delaunay triangle: keep iff empty circumdisk radius ≤ alphaRadiusMax (planar px/py).
kernel void alpha_shape_tri_keep_kernel(
    device const float* px [[buffer(0)]],
    device const float* py [[buffer(1)]],
    device const int* triIdx [[buffer(2)]],
    device uchar* keepOut [[buffer(3)]],
    constant int& triangleCount [[buffer(4)]],
    constant float& alphaRadiusMax [[buffer(5)]],
    constant int& pointCount [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    int nt = triangleCount;
    if ((int)tid >= nt)
        return;

    int i0 = triIdx[tid * 3 + 0];
    int i1 = triIdx[tid * 3 + 1];
    int i2 = triIdx[tid * 3 + 2];
    int pc = pointCount;
    if (i0 < 0 || i1 < 0 || i2 < 0 || i0 >= pc || i1 >= pc || i2 >= pc) {
        keepOut[tid] = 0;
        return;
    }

    float ax = px[i0], ay = py[i0];
    float bx = px[i1], by = py[i1];
    float cx = px[i2], cy = py[i2];

    float abx = bx - ax, aby = by - ay;
    float bcx = cx - bx, bcy = cy - by;
    float cax = ax - cx, cay = ay - cy;
    float lab = sqrt(abx * abx + aby * aby);
    float lbc = sqrt(bcx * bcx + bcy * bcy);
    float lca = sqrt(cax * cax + cay * cay);

    float cross = abx * (cy - ay) - aby * (cx - ax);
    float area2 = fabs(cross);
    const float eps = 1e-20f;
    if (area2 < eps) {
        keepOut[tid] = 0;
        return;
    }

    float R = (lab * lbc * lca) / area2;
    float lim = alphaRadiusMax;
    keepOut[tid] = (R <= lim) ? 1 : 0;
}
