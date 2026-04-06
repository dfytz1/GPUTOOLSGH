using System.Threading.Tasks;
using Grasshopper.Kernel;
using Rhino.Geometry;
using GHGPUPlugin.NativeInterop;

namespace GHGPUPlugin.MeshTopology;

/// <summary>
/// Umbrella Laplacian on topology vertices — same numerics as Chromodoris multithreaded smooth; GPU via MetalBridge.
/// </summary>
public static class MeshLaplacianMetalSmooth
{
    public sealed class Options
    {
        public bool UseGpu { get; init; } = true;

        /// <summary>
        /// When <see cref="UseGpu"/> is true but Metal is unavailable or the kernel fails, run CPU instead (QuickSmooth-style).
        /// </summary>
        public bool CpuFallbackIfGpuUnavailable { get; init; }

        public bool WarnOnCpuFallback { get; init; } = true;
    }

    public static bool TrySmooth(
        GH_Component owner,
        Mesh meshIn,
        double strength,
        int iterations,
        Options options,
        out Mesh? result)
    {
        result = null;
        NativeLoader.EnsureLoaded();

        if (meshIn == null || !meshIn.IsValid)
            return false;

        if (iterations < 1)
            return false;

        if (strength < 0)
            strength = 0;

        float strengthF = (float)strength;
        int[][] neighbors = MeshTopologyNeighbors.NeighborsFromEdges(meshIn);
        int n = neighbors.Length;
        if (n == 0)
            return false;

        var topo = new Point3f[n];
        MeshTopologyNeighbors.TopologyPositionsToArray(meshIn, topo);
        MeshTopologyNeighbors.ToCsr(neighbors, out int[] adjFlat, out int[] rowOffsets);
        var parallelOpts = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };

        bool ranGpu = false;

        if (options.UseGpu)
        {
            IntPtr ctx = IntPtr.Zero;
            bool metalLive = NativeLoader.IsMetalAvailable &&
                             MetalSharedContext.TryGetContext(out ctx);

            if (metalLive)
            {
                var x = new float[n];
                var y = new float[n];
                var z = new float[n];
                CopySoa(topo, x, y, z);

                int code = MetalBridge.RunLaplacianIterations(
                    ctx, x, y, z, adjFlat, rowOffsets, n, strengthF, iterations);
                if (code == 0)
                {
                    CopyFromSoa(topo, x, y, z);
                    ranGpu = true;
                }
                else if (options.CpuFallbackIfGpuUnavailable)
                {
                    if (options.WarnOnCpuFallback)
                    {
                        owner.AddRuntimeMessage(
                            GH_RuntimeMessageLevel.Warning,
                            $"Metal Laplacian failed (code {code}); using CPU.");
                    }
                }
                else
                {
                    owner.AddRuntimeMessage(
                        GH_RuntimeMessageLevel.Error,
                        $"Metal Laplacian failed with code {code}.");
                    return false;
                }
            }
            else if (options.CpuFallbackIfGpuUnavailable)
            {
                if (options.WarnOnCpuFallback)
                {
                    string detail = !NativeLoader.IsMetalAvailable
                        ? (NativeLoader.LoadError ?? "MetalBridge not loaded")
                        : (MetalSharedContext.InitError ?? "Metal context failed");
                    owner.AddRuntimeMessage(
                        GH_RuntimeMessageLevel.Warning,
                        $"Metal not available ({detail}); using CPU Laplacian.");
                }
            }
            else
            {
                if (!MetalGuard.EnsureReady(owner))
                    return false;
            }
        }

        if (!ranGpu)
        {
            for (int it = 0; it < iterations; it++)
                RunLaplacianCpuParallel(topo, neighbors, strength, parallelOpts);
        }

        result = MeshTopologyNeighbors.SmoothedMeshFromTopology(meshIn, topo);
        return true;
    }

    private static void CopySoa(Point3f[] p, float[] x, float[] y, float[] z)
    {
        for (int i = 0; i < p.Length; i++)
        {
            x[i] = p[i].X;
            y[i] = p[i].Y;
            z[i] = p[i].Z;
        }
    }

    private static void CopyFromSoa(Point3f[] p, float[] x, float[] y, float[] z)
    {
        for (int i = 0; i < p.Length; i++)
            p[i] = new Point3f(x[i], y[i], z[i]);
    }

    private static void RunLaplacianCpuParallel(
        Point3f[] topo,
        int[][] neighbors,
        double step,
        ParallelOptions parallelOptions)
    {
        Parallel.For(0, topo.Length, parallelOptions, v =>
        {
            int[] nvs = neighbors[v];
            if (nvs.Length == 0)
                return;

            Point3d loc = new(topo[v].X, topo[v].Y, topo[v].Z);
            Point3d avg = new(0, 0, 0);
            for (int k = 0; k < nvs.Length; k++)
            {
                Point3f q = topo[nvs[k]];
                avg.X += q.X;
                avg.Y += q.Y;
                avg.Z += q.Z;
            }

            avg.X /= nvs.Length;
            avg.Y /= nvs.Length;
            avg.Z /= nvs.Length;

            Vector3d pos = new Vector3d(loc) + (avg - loc) * step;
            topo[v] = new Point3f((float)pos.X, (float)pos.Y, (float)pos.Z);
        });
    }
}
