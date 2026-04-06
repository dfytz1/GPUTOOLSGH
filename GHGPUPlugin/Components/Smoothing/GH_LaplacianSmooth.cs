using System.Threading.Tasks;
using Grasshopper.Kernel;
using Rhino.Geometry;
using GHGPUPlugin.MeshTopology;
using GHGPUPlugin.NativeInterop;
using System.Drawing;

namespace GHGPUPlugin.Components.Smoothing;

public class GH_LaplacianSmooth : GH_Component
{
    public GH_LaplacianSmooth()
        : base(
            "Laplacian Smooth GPU",
            "LapSmoothGPU",
            "Umbrella Laplacian on topology vertices. No input mesh duplicate; CPU matches Chromodoris; GPU uses cached Metal context + batched submits.",
            "GPUTools",
            "Smoothing")
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddMeshParameter("InputMesh", "InputMesh", "Mesh to smooth (read-only; output is a new mesh).", GH_ParamAccess.item);
        pManager.AddNumberParameter("Strength", "Strength", "Step toward neighbor centroid (Chromodoris-style).", GH_ParamAccess.item, 0.35);
        pManager.AddIntegerParameter("Iterations", "Iterations", "Number of smoothing iterations.", GH_ParamAccess.item, 8);
        pManager.AddBooleanParameter("UseGPU", "UseGPU", "Use Metal when available.", GH_ParamAccess.item, true);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddMeshParameter("SmoothedMesh", "SmoothedMesh", "Smoothed mesh copy.", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        NativeLoader.EnsureLoaded();

        Mesh? meshIn = null;
        if (!DA.GetData("InputMesh", ref meshIn) || meshIn == null)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "No mesh provided.");
            return;
        }

        if (!meshIn.IsValid)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Mesh is not valid.");
            return;
        }

        double strengthD = 0.35;
        DA.GetData("Strength", ref strengthD);
        if (strengthD < 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Strength clamped to 0.");
            strengthD = 0;
        }

        float strengthF = (float)strengthD;

        int iterations = 8;
        DA.GetData("Iterations", ref iterations);
        if (iterations < 1)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Iterations must be at least 1.");
            return;
        }

        bool useGpu = true;
        DA.GetData("UseGPU", ref useGpu);

        int[][] neighbors = MeshTopologyNeighbors.NeighborsFromEdges(meshIn);
        int n = neighbors.Length;
        if (n == 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Mesh has no topology vertices.");
            return;
        }

        var topo = new Point3f[n];
        MeshTopologyNeighbors.TopologyPositionsToArray(meshIn, topo);

        MeshTopologyNeighbors.ToCsr(neighbors, out int[] adjFlat, out int[] rowOffsets);

        var opts = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };

        bool ranGpu = false;
        if (useGpu)
        {
            if (!MetalGuard.EnsureReady(this))
                return;

            MetalSharedContext.TryGetContext(out IntPtr ctx);
            var x = new float[n];
            var y = new float[n];
            var z = new float[n];
            CopySoa(topo, x, y, z);

            int code = MetalBridge.RunLaplacianIterations(
                ctx,
                x,
                y,
                z,
                adjFlat,
                rowOffsets,
                n,
                strengthF,
                iterations);
            if (code != 0)
            {
                AddRuntimeMessage(
                    GH_RuntimeMessageLevel.Error,
                    $"Metal Laplacian failed with code {code}.");
                return;
            }

            CopyFromSoa(topo, x, y, z);
            ranGpu = true;
        }

        if (!ranGpu)
        {
            if (useGpu)
            {
                AddRuntimeMessage(
                    GH_RuntimeMessageLevel.Warning,
                    "GPU Laplacian did not run — using CPU (Chromodoris-style parallel).");
            }

            for (int it = 0; it < iterations; it++)
                RunLaplacianCpuChromodorisParallel(topo, neighbors, strengthD, opts);
        }

        Mesh outMesh = MeshTopologyNeighbors.SmoothedMeshFromTopology(meshIn, topo);
        DA.SetData("SmoothedMesh", outMesh);
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

    private static void RunLaplacianCpuChromodorisParallel(
        Point3f[] topo,
        int[][] neighbors,
        double step,
        ParallelOptions opts)
    {
        Parallel.For(0, topo.Length, opts, v =>
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

    protected override Bitmap Icon => null!;

    public override Guid ComponentGuid => new("fa8d7e40-cf38-4773-9684-a3b3b36e94e8");
}
