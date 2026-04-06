#include <metal_stdlib>
using namespace metal;

kernel void benchmarkKernel(
    device float* buffer [[buffer(0)]],
    constant int* params [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    int count = params[0];
    int innerIters = params[1];
    if ((int)gid >= count)
        return;

    float x = buffer[gid];
    for (int i = 0; i < innerIters; i++)
        x = x * 1.0000001f + 0.0000001f;
    buffer[gid] = x;
}
