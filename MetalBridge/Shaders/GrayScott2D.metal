#include <metal_stdlib>
using namespace metal;

// Gray–Scott model (Pearson): A + 2B → 3B, feed f, kill k.
// Grid index: gid = ix * ny + iy, ix in [0, nx), iy in [0, ny) — matches C# float[nx,ny] layout.
// Neumann boundaries via clamped neighbor indices.

kernel void gray_scott_step_kernel(
    device const float* aIn [[buffer(0)]],
    device const float* bIn [[buffer(1)]],
    device float* aOut [[buffer(2)]],
    device float* bOut [[buffer(3)]],
    device const float* params [[buffer(4)]],
    device const int* dims [[buffer(5)]],
    device const int* totalN [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    int n = *totalN;
    if ((int)gid >= n)
        return;

    int nx = dims[0];
    int ny = dims[1];
    int iy = (int)gid % ny;
    int ix = (int)gid / ny;

    float dt = params[0];
    float f = params[1];
    float k = params[2];
    float dA = params[3];
    float dB = params[4];

    int ixm = max(ix - 1, 0);
    int ixp = min(ix + 1, nx - 1);
    int iym = max(iy - 1, 0);
    int iyp = min(iy + 1, ny - 1);

    int c = ix * ny + iy;
    float aC = aIn[c];
    float bC = bIn[c];

    float lapA = aIn[ixm * ny + iy] + aIn[ixp * ny + iy] + aIn[ix * ny + iym] + aIn[ix * ny + iyp] - 4.f * aC;
    float lapB = bIn[ixm * ny + iy] + bIn[ixp * ny + iy] + bIn[ix * ny + iym] + bIn[ix * ny + iyp] - 4.f * bC;

    float r = aC * bC * bC;
    float na = aC + (dA * lapA - r + f * (1.f - aC)) * dt;
    float nb = bC + (dB * lapB + r - (f + k) * bC) * dt;

    aOut[c] = clamp(na, 0.f, 1.f);
    bOut[c] = clamp(nb, 0.f, 1.f);
}
