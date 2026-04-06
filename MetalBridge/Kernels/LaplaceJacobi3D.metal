#include <metal_stdlib>
using namespace metal;

struct LaplaceParams3D {
    uint nx;
    uint ny;
    uint nz;
    float sv;
    float lv;
};

kernel void laplace_jacobi_3d(
    device const float* inside [[buffer(0)]],
    device const float* support [[buffer(1)]],
    device const float* load [[buffer(2)]],
    device const float* src [[buffer(3)]],
    device float* dst [[buffer(4)]],
    constant LaplaceParams3D& p [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.nx * p.ny * p.nz)
        return;
    if (inside[gid] < 0.5f) {
        dst[gid] = 0.f;
        return;
    }
    if (support[gid] > 0.5f) {
        dst[gid] = p.sv;
        return;
    }
    if (load[gid] > 0.5f) {
        dst[gid] = p.lv;
        return;
    }
    uint i = gid / (p.ny * p.nz);
    uint j = (gid / p.nz) % p.ny;
    uint k = gid % p.nz;
    float s = 0.f;
    int n = 0;
    if (i > 0) {
        s += src[(i - 1) * p.ny * p.nz + j * p.nz + k];
        n++;
    }
    if (i < p.nx - 1) {
        s += src[(i + 1) * p.ny * p.nz + j * p.nz + k];
        n++;
    }
    if (j > 0) {
        s += src[i * p.ny * p.nz + (j - 1) * p.nz + k];
        n++;
    }
    if (j < p.ny - 1) {
        s += src[i * p.ny * p.nz + (j + 1) * p.nz + k];
        n++;
    }
    if (k > 0) {
        s += src[i * p.ny * p.nz + j * p.nz + (k - 1)];
        n++;
    }
    if (k < p.nz - 1) {
        s += src[i * p.ny * p.nz + j * p.nz + (k + 1)];
        n++;
    }
    dst[gid] = n > 0 ? s / float(n) : src[gid];
}
