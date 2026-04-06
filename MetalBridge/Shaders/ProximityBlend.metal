#include <metal_stdlib>
using namespace metal;

// params[0]=mix, [1]=contrastExp, [2]=invProxMax, [3]=rslInv, [4]=wc, [5]=rcInv,
// [6]=includeSl, [7]=useCenter, [8]=distSentinel,
// [9..11]=origin (PointAt 0,0,0), [12..14]=ex/nx, [15..17]=ey/ny, [18..20]=ez/nz,
// [21..23]=boxCenter xyz
kernel void proximity_blend_kernel(
    device const float* gradField [[buffer(0)]],
    device const float* distSL [[buffer(1)]],
    device const float* inside [[buffer(2)]],
    device float* density [[buffer(3)]],
    device const float* params [[buffer(4)]],
    device const int* totalN [[buffer(5)]],
    device const int* dims [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    int n = *totalN;
    if ((int)gid >= n)
        return;

    if (inside[gid] < 0.5f) {
        density[gid] = 0.f;
        return;
    }

    float mix = params[0];
    float contrastExp = params[1];
    float invProxMax = params[2];
    float rslInv = params[3];
    float wc = params[4];
    float rcInv = params[5];
    float includeSl = params[6];
    float useCenter = params[7];
    float distSentinel = params[8];
    float ox = params[9], oy = params[10], oz = params[11];
    float exx = params[12], exy = params[13], exz = params[14];
    float eyx = params[15], eyy = params[16], eyz = params[17];
    float ezx = params[18], ezy = params[19], ezz = params[20];
    float bcx = params[21], bcy = params[22], bcz = params[23];

    int ny = dims[1], nz = dims[2];

    int iz = (int)gid % nz;
    int iy = ((int)gid / nz) % ny;
    int ix = (int)gid / (ny * nz);

    float cx = ox + (ix + 0.5f) * exx + (iy + 0.5f) * eyx + (iz + 0.5f) * ezx;
    float cy = oy + (ix + 0.5f) * exy + (iy + 0.5f) * eyy + (iz + 0.5f) * ezy;
    float cz = oz + (ix + 0.5f) * exz + (iy + 0.5f) * eyz + (iz + 0.5f) * ezz;

    float d = distSL[gid];
    float pSl = 0.f;
    if (includeSl > 0.5f && rslInv > 1e-20f && d < distSentinel * 0.5f)
        pSl = exp(-d * rslInv);

    float p = pSl;
    if (useCenter > 0.5f && rcInv > 1e-20f) {
        float dx = cx - bcx;
        float dy = cy - bcy;
        float dz = cz - bcz;
        float dc = sqrt(dx * dx + dy * dy + dz * dz);
        float pC = exp(-dc * rcInv);
        p = max(p, wc * pC);
    }

    float proxNorm = p * invProxMax;
    proxNorm = clamp(proxNorm, 0.f, 1.f);

    float g = gradField[gid];
    g = clamp(g, 0.f, 1.f);
    float blended = (1.f - mix) * g + mix * proxNorm;
    blended = clamp(blended, 0.f, 1.f);
    density[gid] = powr(blended, contrastExp);
}
