using System.Collections.Generic;
using Grasshopper.Kernel;
using GHGPUPlugin.NativeInterop;
using Rhino.Geometry;

namespace GHGPUPlugin.Components.Field;

internal static class GrayScottField2DSolver
{
    internal const int MaxCells = 8_000_000;

    /// <summary>Runs Gray–Scott; returns false if validation failed (owner gets a warning).</summary>
    internal static bool TrySolve(
        GH_Component owner,
        int nx,
        int ny,
        int nIters,
        double f,
        double k,
        double dA,
        double dB,
        double dt,
        Plane plane,
        double sx,
        double sy,
        List<Point3d> seedPoints,
        List<Curve> seedCurves,
        double seedR,
        float[,]? initialB,
        bool useDefaultCenterSeed,
        Mesh? meshSeedVertices,
        bool useGpu,
        out float[,] aOut,
        out float[,] bOut)
    {
        aOut = null!;
        bOut = null!;

        if (nx < 2 || ny < 2)
        {
            owner.AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Resolution must be at least 2×2.");
            return false;
        }

        if ((long)nx * ny > MaxCells)
        {
            owner.AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                $"Grid has {nx * ny} cells; maximum supported is {MaxCells}. Reduce resolution.");
            return false;
        }

        if (nIters < 1)
        {
            owner.AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Iterations must be at least 1.");
            return false;
        }

        if (sx <= 0 || sy <= 0 || seedR < 0)
        {
            owner.AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "SizeX, SizeY must be positive; SeedRadius must be non-negative.");
            return false;
        }

        var a = new float[nx, ny];
        var b = new float[nx, ny];
        for (int ix = 0; ix < nx; ix++)
        {
            for (int iy = 0; iy < ny; iy++)
            {
                a[ix, iy] = 1f;
                b[ix, iy] = 0f;
            }
        }

        if (initialB != null)
        {
            for (int ix = 0; ix < nx; ix++)
            {
                for (int iy = 0; iy < ny; iy++)
                    b[ix, iy] = initialB[ix, iy];
            }
        }

        bool anySeed =
            seedPoints.Count > 0
            || seedCurves.Count > 0
            || meshSeedVertices != null
            || initialB != null;

        ReactionDiffusion2DSeeding.SplatAll(
            b, nx, ny, plane, sx, sy, seedPoints, seedCurves, seedR,
            useDefaultCenter: useDefaultCenterSeed && !anySeed,
            meshVerticesToSplat: meshSeedVertices);

        float[] aL = ReactionDiffusionGrayScott.Flatten(a);
        float[] bL = ReactionDiffusionGrayScott.Flatten(b);

        bool gpuOk = false;
        NativeLoader.EnsureLoaded();
        if (useGpu && MetalGuard.EnsureReady(owner) && MetalSharedContext.TryGetContext(out IntPtr ctx))
        {
            int code = MetalBridge.GrayScott2D(ctx, aL, bL, nx, ny, nIters, (float)dt, (float)f, (float)k, (float)dA, (float)dB);
            gpuOk = code == 0;
            if (!gpuOk)
                owner.AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"Metal GrayScott2D returned error code {code}; using CPU.");
        }
        else if (useGpu)
        {
            owner.AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "GPU unavailable; using CPU.");
        }

        if (!gpuOk)
            ReactionDiffusionGrayScott.RunCpu(aL, bL, nx, ny, nIters, (float)dt, (float)f, (float)k, (float)dA, (float)dB);

        aOut = ReactionDiffusionGrayScott.Unflatten(aL, nx, ny);
        bOut = ReactionDiffusionGrayScott.Unflatten(bL, nx, ny);
        return true;
    }
}
