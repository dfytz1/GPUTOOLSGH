using System;

namespace GHGPUPlugin.Components.Field;

/// <summary>Gray–Scott 2D with C# layout <c>float[nx,ny]</c> → linear index <c>ix * ny + iy</c> (matches Metal).</summary>
internal static class ReactionDiffusionGrayScott
{
    internal static float[] Flatten(float[,] src)
    {
        int nx = src.GetLength(0);
        int ny = src.GetLength(1);
        var dst = new float[nx * ny];
        for (int ix = 0; ix < nx; ix++)
        {
            for (int iy = 0; iy < ny; iy++)
                dst[ix * ny + iy] = src[ix, iy];
        }

        return dst;
    }

    internal static float[,] Unflatten(float[] src, int nx, int ny)
    {
        var dst = new float[nx, ny];
        for (int ix = 0; ix < nx; ix++)
        {
            for (int iy = 0; iy < ny; iy++)
                dst[ix, iy] = src[ix * ny + iy];
        }

        return dst;
    }

    internal static void RunCpu(float[] a, float[] b, int nx, int ny, int iterations, float dt, float f, float k, float dA, float dB)
    {
        int n = nx * ny;
        var aN = new float[n];
        var bN = new float[n];
        float[] curA = a, curB = b, nxtA = aN, nxtB = bN;

        for (int it = 0; it < iterations; it++)
        {
            Step(curA, curB, nxtA, nxtB, nx, ny, dt, f, k, dA, dB);
            (curA, nxtA) = (nxtA, curA);
            (curB, nxtB) = (nxtB, curB);
        }

        if (!ReferenceEquals(curA, a))
        {
            Array.Copy(curA, a, n);
            Array.Copy(curB, b, n);
        }
    }

    private static void Step(float[] aIn, float[] bIn, float[] aOut, float[] bOut, int nx, int ny,
        float dt, float f, float k, float dA, float dB)
    {
        int Idx(int ix, int iiy) => ix * ny + iiy;

        for (int ix = 0; ix < nx; ix++)
        {
            int ixm = Math.Max(ix - 1, 0);
            int ixp = Math.Min(ix + 1, nx - 1);
            for (int iy = 0; iy < ny; iy++)
            {
                int iym = Math.Max(iy - 1, 0);
                int iyp = Math.Min(iy + 1, ny - 1);
                int c = Idx(ix, iy);
                float aC = aIn[c];
                float bC = bIn[c];
                float lapA = aIn[Idx(ixm, iy)] + aIn[Idx(ixp, iy)] + aIn[Idx(ix, iym)] + aIn[Idx(ix, iyp)] - 4f * aC;
                float lapB = bIn[Idx(ixm, iy)] + bIn[Idx(ixp, iy)] + bIn[Idx(ix, iym)] + bIn[Idx(ix, iyp)] - 4f * bC;
                float r = aC * bC * bC;
                float na = aC + (dA * lapA - r + f * (1f - aC)) * dt;
                float nb = bC + (dB * lapB + r - (f + k) * bC) * dt;
                aOut[c] = Math.Clamp(na, 0f, 1f);
                bOut[c] = Math.Clamp(nb, 0f, 1f);
            }
        }
    }
}
