using GHGPUPlugin.NativeInterop;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using System;

namespace GHGPUPlugin.Chromodoris;

/// <summary>
/// Separable 3D Gaussian blur on a voxel density field, with optional Metal Laplace–Jacobi approximation.
/// </summary>
public class VoxelDensitySmoothComponent : GH_Component
{
    public VoxelDensitySmoothComponent()
        : base("Voxel Density Smooth GPU", "DensitySmoothGPU",
            "Smooth a float[x,y,z] density field with separable Gaussian (CPU) or Laplace–Jacobi diffusion on GPU (Metal).",
            "GPUTools", "Voxel")
    {
    }

    protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
    {
        pManager.AddGenericParameter("DensityField", "D", "float[x,y,z] voxel density.", GH_ParamAccess.item);
        pManager.AddNumberParameter("Sigma", "Sig", "Gaussian sigma in voxels (CPU); ignored for GPU Laplace path.", GH_ParamAccess.item, 0.8);
        pManager.AddIntegerParameter("Iterations", "It", "Outer passes (CPU: XYZ repeats); GPU: Jacobi iteration count.", GH_ParamAccess.item, 1);
        pManager.AddBooleanParameter("PreserveVoid", "PVoid", "After blur, pin voxels with original rho below 1e-6 to 0.", GH_ParamAccess.item, true);
        pManager.AddBooleanParameter("PreserveSolid", "PSol", "After blur, pin voxels with original rho above 1-1e-6 to 1.", GH_ParamAccess.item, false);
        pManager.AddBooleanParameter("UseGPU", "GPU", "Use Metal Laplace–Jacobi when available.", GH_ParamAccess.item, true);
        pManager[pManager.ParamCount - 1].Optional = true;
    }

    protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
    {
        pManager.AddGenericParameter("SmoothedField", "S", "float[x,y,z] smoothed density.", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        float[,,] density = null;
        double sigma = 0.8;
        int iterations = 1;
        bool preserveVoid = true, preserveSolid = false, useGpu = true;

        if (!VoxelMaskGoo.TryGetFloatTensor3(DA, 0, this, out density, "DensityField"))
            return;
        DA.GetData(1, ref sigma);
        DA.GetData(2, ref iterations);
        DA.GetData(3, ref preserveVoid);
        DA.GetData(4, ref preserveSolid);
        DA.GetData(5, ref useGpu);

        NativeLoader.EnsureLoaded();

        int nx = density.GetLength(0);
        int ny = density.GetLength(1);
        int nz = density.GetLength(2);

        if (iterations < 1)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Iterations must be at least 1.");
            return;
        }

        float[,,] srcOrig = density;
        float[,,] result;

        bool tryGpu = useGpu && NativeLoader.IsMetalAvailable;
        if (useGpu && !NativeLoader.IsMetalAvailable)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                "Metal not available — CPU separable Gaussian. " + (NativeLoader.LoadError ?? string.Empty));
        }

        if (tryGpu && MetalSharedContext.TryGetContext(out IntPtr ctx))
        {
            int total = nx * ny * nz;
            var inside = new float[total];
            for (int i = 0; i < total; i++)
                inside[i] = 1f;
            var support = new float[total];
            var load = new float[total];
            float[] phi = VoxelGpuHelper.Flatten(srcOrig);

            try
            {
                int code = MetalBridge.LaplaceJacobi3D(ctx, inside, support, load, phi, nx, ny, nz, 0f, 0f, iterations);
                if (code == 0)
                {
                    result = VoxelGpuHelper.Unflatten(phi, nx, ny, nz);
                    ApplyPreservePins(srcOrig, result, preserveVoid, preserveSolid);
                    DA.SetData(0, new GH_ObjectWrapper(result));
                    return;
                }

                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"GPU Laplace–Jacobi returned {code} — CPU separable Gaussian.");
            }
            catch (Exception ex)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "GPU Laplace–Jacobi failed — CPU separable Gaussian: " + ex.Message);
            }
        }
        else if (tryGpu)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Metal context unavailable — CPU separable Gaussian.");
        }

        result = CpuSeparableGaussian(srcOrig, nx, ny, nz, sigma, iterations);
        ApplyPreservePins(srcOrig, result, preserveVoid, preserveSolid);
        DA.SetData(0, new GH_ObjectWrapper(result));
    }

    private static void ApplyPreservePins(float[,,] srcOrig, float[,,] blurred, bool preserveVoid, bool preserveSolid)
    {
        int nx = srcOrig.GetLength(0);
        int ny = srcOrig.GetLength(1);
        int nz = srcOrig.GetLength(2);
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                for (int k = 0; k < nz; k++)
                {
                    float s = srcOrig[i, j, k];
                    if (preserveVoid && s < 1e-6f)
                        blurred[i, j, k] = 0f;
                    else if (preserveSolid && s > 1f - 1e-6f)
                        blurred[i, j, k] = 1f;
                }
            }
        }
    }

    private static float[,,] CpuSeparableGaussian(float[,,] src, int nx, int ny, int nz, double sigma, int iterations)
    {
        float[,,] cur = CloneTensor3(src);
        if (sigma <= 0 || iterations < 1)
            return cur;

        float[] kernel = BuildGaussianKernel(sigma, out int radius);
        float[,,] nxt = new float[nx, ny, nz];

        for (int it = 0; it < iterations; it++)
        {
            ConvolveAxis(cur, nxt, kernel, radius, nx, ny, nz, 0);
            ConvolveAxis(nxt, cur, kernel, radius, nx, ny, nz, 1);
            ConvolveAxis(cur, nxt, kernel, radius, nx, ny, nz, 2);
            float[,,] tmp = cur;
            cur = nxt;
            nxt = tmp;
        }

        return cur;
    }

    /// <summary>axis: 0=X, 1=Y, 2=Z</summary>
    private static void ConvolveAxis(float[,,] src, float[,,] dst, float[] kernel, int radius, int nx, int ny, int nz, int axis)
    {
        int klen = kernel.Length;
        for (int i = 0; i < nx; i++)
        {
            for (int j = 0; j < ny; j++)
            {
                for (int k = 0; k < nz; k++)
                {
                    double sum = 0;
                    for (int t = 0; t < klen; t++)
                    {
                        int o = t - radius;
                        int ii = i, jj = j, kk = k;
                        if (axis == 0) ii = ClampIndex(i + o, nx);
                        else if (axis == 1) jj = ClampIndex(j + o, ny);
                        else kk = ClampIndex(k + o, nz);
                        sum += kernel[t] * src[ii, jj, kk];
                    }

                    dst[i, j, k] = (float)sum;
                }
            }
        }
    }

    private static int ClampIndex(int idx, int n)
    {
        if (idx < 0) return 0;
        if (idx >= n) return n - 1;
        return idx;
    }

    private static float[] BuildGaussianKernel(double sigma, out int radius)
    {
        radius = (int)Math.Ceiling(sigma * 3);
        if (radius < 1)
            radius = 1;
        int len = 2 * radius + 1;
        var k = new float[len];
        double sum = 0;
        for (int t = 0; t < len; t++)
        {
            double x = t - radius;
            double v = Math.Exp(-(x * x) / (2 * sigma * sigma));
            k[t] = (float)v;
            sum += v;
        }

        if (sum < 1e-30)
        {
            k[radius] = 1f;
            return k;
        }

        for (int t = 0; t < len; t++)
            k[t] = (float)(k[t] / sum);
        return k;
    }

    private static float[,,] CloneTensor3(float[,,] src)
    {
        int nx = src.GetLength(0), ny = src.GetLength(1), nz = src.GetLength(2);
        var dst = new float[nx, ny, nz];
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < ny; j++)
                for (int kk = 0; kk < nz; kk++)
                    dst[i, j, kk] = src[i, j, kk];
        return dst;
    }

    public override GH_Exposure Exposure => GH_Exposure.quinary;

    protected override System.Drawing.Bitmap Icon => null!;

    public override Guid ComponentGuid => new Guid("b7e3d1a4-6f2c-4e89-9b0d-1c4a8e7f5023");
}
