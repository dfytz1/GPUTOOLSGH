using GHGPUPlugin.NativeInterop;
using Grasshopper.Kernel;
using System;

namespace GHGPUPlugin.Chromodoris;

internal static class VoxelGpuHelper
{
    public static float[] Flatten(float[,,] a)
    {
        int nx = a.GetLength(0), ny = a.GetLength(1), nz = a.GetLength(2);
        var f = new float[nx * ny * nz];
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < ny; j++)
                for (int k = 0; k < nz; k++)
                    f[i * ny * nz + j * nz + k] = a[i, j, k];
        return f;
    }

    public static float[,,] Unflatten(float[] f, int nx, int ny, int nz)
    {
        var a = new float[nx, ny, nz];
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < ny; j++)
                for (int k = 0; k < nz; k++)
                    a[i, j, k] = f[i * ny * nz + j * nz + k];
        return a;
    }

    public static void DomainMinMax(float[] data, float[] inside, int n, out float dMin, out float dMax)
    {
        dMin = float.MaxValue;
        dMax = float.MinValue;
        for (int i = 0; i < n; i++)
        {
            if (inside[i] < 0.5f)
                continue;
            if (data[i] < dMin)
                dMin = data[i];
            if (data[i] > dMax)
                dMax = data[i];
        }

        if (dMin > dMax)
        {
            dMin = 0f;
            dMax = 0f;
        }
    }

    public static bool TryLaplaceGpu(
        GH_Component c,
        float[] inside,
        float[] support,
        float[] load,
        float[] phi,
        int nx,
        int ny,
        int nz,
        float sv,
        float lv,
        int iters)
    {
        if (!MetalSharedContext.TryGetContext(out IntPtr ctx))
            return false;
        try
        {
            int code = MetalBridge.LaplaceJacobi3D(ctx, inside, support, load, phi, nx, ny, nz, sv, lv, iters);
            if (code != 0)
            {
                c.AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"GPU Jacobi error {code} — CPU fallback.");
                return false;
            }

            return true;
        }
        catch (Exception ex)
        {
            c.AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"GPU Jacobi: {ex.Message} — CPU fallback.");
            return false;
        }
    }

    public static bool TryGradientGpu(
        GH_Component c,
        float[] phi,
        float[] inside,
        float[] gradOut,
        int nx,
        int ny,
        int nz,
        float iDx,
        float iDy,
        float iDz)
    {
        if (!MetalSharedContext.TryGetContext(out IntPtr ctx))
            return false;
        try
        {
            int code = MetalBridge.GradientMagnitude3D(ctx, phi, inside, gradOut, nx, ny, nz, iDx, iDy, iDz);
            if (code != 0)
            {
                c.AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"GPU Gradient error {code} — CPU fallback.");
                return false;
            }

            return true;
        }
        catch (Exception ex)
        {
            c.AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"GPU Gradient: {ex.Message} — CPU fallback.");
            return false;
        }
    }

    public static bool TryNormalizeGpu(
        GH_Component c,
        float[] data,
        float[] inside,
        int nx,
        int ny,
        int nz,
        float dMin,
        float dMax,
        bool invert,
        double exp)
    {
        if (!MetalSharedContext.TryGetContext(out IntPtr ctx))
            return false;
        try
        {
            int code = MetalBridge.NormalizeContrast3D(
                ctx,
                data,
                inside,
                nx,
                ny,
                nz,
                dMin,
                dMax,
                invert ? 1 : 0,
                (float)exp);
            if (code != 0)
            {
                c.AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"GPU Normalize error {code} — CPU fallback.");
                return false;
            }

            return true;
        }
        catch (Exception ex)
        {
            c.AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"GPU Normalize: {ex.Message} — CPU fallback.");
            return false;
        }
    }

    public static bool TryZeroBoundaryGpu(GH_Component c, float[] data, int nx, int ny, int nz)
    {
        if (!MetalSharedContext.TryGetContext(out IntPtr ctx))
            return false;
        try
        {
            int code = MetalBridge.ZeroVoxelBoundary(ctx, data, nx, ny, nz);
            if (code != 0)
            {
                c.AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"GPU ZeroBoundary error {code} — CPU fallback.");
                return false;
            }

            return true;
        }
        catch (Exception ex)
        {
            c.AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"GPU ZeroBoundary: {ex.Message} — CPU fallback.");
            return false;
        }
    }
}
