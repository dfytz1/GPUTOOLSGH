#include <metal_stdlib>
using namespace metal;

// Each cell stores the index of the nearest seed point (-1 = unset)
kernel void jfaInitKernel(
    device int* grid [[buffer(0)]],
    device const float* px [[buffer(1)]],
    device const float* py [[buffer(2)]],
    device const int* seedCount [[buffer(3)]],
    device const int* gridRes [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    int n = *seedCount;
    int res = *gridRes;
    if ((int)id >= n)
        return;

    int gx = (int)(px[id] * (float)(res - 1) + 0.5f);
    int gy = (int)(py[id] * (float)(res - 1) + 0.5f);
    gx = clamp(gx, 0, res - 1);
    gy = clamp(gy, 0, res - 1);
    grid[gy * res + gx] = (int)id;
}

kernel void jfaStepKernel(
    device const int* gridIn [[buffer(0)]],
    device int* gridOut [[buffer(1)]],
    device const float* px [[buffer(2)]],
    device const float* py [[buffer(3)]],
    device const int* stepSize [[buffer(4)]],
    device const int* gridRes [[buffer(5)]],
    uint2 pos [[thread_position_in_grid]])
{
    int res = *gridRes;
    int step = *stepSize;
    int x = (int)pos.x;
    int y = (int)pos.y;
    if (x >= res || y >= res)
        return;

    int best = gridIn[y * res + x];
    float bestDist = FLT_MAX;
    if (best >= 0)
    {
        float fx0 = (float)x / (float)(res - 1);
        float fy0 = (float)y / (float)(res - 1);
        float dx = px[best] - fx0;
        float dy = py[best] - fy0;
        bestDist = dx * dx + dy * dy;
    }

    const int offsets[9][2] = {
        { -step, -step }, { 0, -step }, { step, -step }, { -step, 0 }, { 0, 0 }, { step, 0 }, { -step, step }, { 0, step }, { step, step }
    };

    float fx = (float)x / (float)(res - 1);
    float fy = (float)y / (float)(res - 1);

    for (int k = 0; k < 9; k++)
    {
        int nx = x + offsets[k][0];
        int ny = y + offsets[k][1];
        if (nx < 0 || nx >= res || ny < 0 || ny >= res)
            continue;
        int candidate = gridIn[ny * res + nx];
        if (candidate < 0)
            continue;
        float dx = px[candidate] - fx;
        float dy = py[candidate] - fy;
        float d = dx * dx + dy * dy;
        if (d < bestDist)
        {
            bestDist = d;
            best = candidate;
        }
    }

    gridOut[y * res + x] = best;
}

// edgesOut: 4 ints per cell — right edge (a,b), down edge (a,b); -1 when absent
kernel void jfaExtractEdgesKernel(
    device const int* grid [[buffer(0)]],
    device int* edgesOut [[buffer(1)]],
    device const int* gridRes [[buffer(2)]],
    uint2 pos [[thread_position_in_grid]])
{
    int res = *gridRes;
    int x = (int)pos.x;
    int y = (int)pos.y;
    int idx = y * res + x;
    int base = idx * 4;

    edgesOut[base + 0] = -1;
    edgesOut[base + 1] = -1;
    edgesOut[base + 2] = -1;
    edgesOut[base + 3] = -1;

    if (x >= res || y >= res)
        return;

    int me = grid[idx];
    if (me < 0)
        return;

    if (x + 1 < res)
    {
        int right = grid[y * res + (x + 1)];
        if (right >= 0 && right != me)
        {
            edgesOut[base + 0] = min(me, right);
            edgesOut[base + 1] = max(me, right);
        }
    }
    if (y + 1 < res)
    {
        int down = grid[(y + 1) * res + x];
        if (down >= 0 && down != me)
        {
            edgesOut[base + 2] = min(me, down);
            edgesOut[base + 3] = max(me, down);
        }
    }
}
