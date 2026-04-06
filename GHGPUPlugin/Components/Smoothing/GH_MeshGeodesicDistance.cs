using System.Drawing;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using GHGPUPlugin.Algorithms;
using GHGPUPlugin.NativeInterop;
using GHGPUPlugin.Utilities;
using Rhino.Geometry;

namespace GHGPUPlugin.Components.Smoothing;

public class GH_MeshGeodesicDistance : GH_Component
{
    public GH_MeshGeodesicDistance()
        : base(
            "Mesh Geodesic Distance GPU",
            "MeshGeoDistGPU",
            "Approximate geodesic distance from seed vertices using heat diffusion (Laplacian) plus gradient divergence and a second diffusion pass.",
            "GPUTools",
            "Mesh")
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddMeshParameter("Mesh", "M", "Triangle or quad mesh.", GH_ParamAccess.item);
        pManager.AddIntegerParameter("SeedVerts", "SV", "Mesh vertex indices where distance = 0.", GH_ParamAccess.list);
        pManager.AddIntegerParameter("Iterations", "I", "Distance diffusion iterations.", GH_ParamAccess.item, 30);
        pManager.AddNumberParameter("Strength", "S", "Distance diffusion strength per iteration.", GH_ParamAccess.item, 0.5);
        pManager.AddBooleanParameter("Normalise", "N", "Remap output distances to 0…1.", GH_ParamAccess.item, true);
        pManager.AddBooleanParameter("UseGPU", "GPU", "Use Metal Laplacian when available.", GH_ParamAccess.item, true);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddNumberParameter("Distances", "D", "Per mesh vertex scalar.", GH_ParamAccess.list);
        pManager.AddMeshParameter("ColourMesh", "CM", "Mesh with vertex colours.", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        NativeLoader.EnsureLoaded();
        AddRuntimeMessage(
            GH_RuntimeMessageLevel.Remark,
            "Approximate geodesic via heat diffusion. Results are smooth but not exact. Accuracy improves with more iterations.");

        Mesh? mesh = null;
        if (!DA.GetData("Mesh", ref mesh) || mesh == null)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Mesh is required.");
            return;
        }

        if (!mesh.IsValid)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Mesh is not valid.");
            return;
        }

        var seeds = new List<int>();
        DA.GetDataList("SeedVerts", seeds);

        int iterations = 30;
        DA.GetData("Iterations", ref iterations);

        double strength = 0.5;
        DA.GetData("Strength", ref strength);

        bool normalise = true;
        DA.GetData("Normalise", ref normalise);

        bool useGpu = true;
        DA.GetData("UseGPU", ref useGpu);

        if (useGpu && !MetalGuard.EnsureReady(this))
            return;

        if (iterations < 1)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Iterations must be at least 1.");
            return;
        }

        if (strength < 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Strength clamped to 0.");
            strength = 0;
        }

        if (seeds.Count == 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Provide at least one seed mesh vertex index.");
            return;
        }

        if (!ApproximateHeatGeodesic.TryCompute(mesh, seeds, iterations, strength, useGpu, out double[]? dist, out string? err)
            || dist == null)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, err ?? "Geodesic computation failed.");
            return;
        }

        int vc = mesh.Vertices.Count;
        if (dist.Length < vc)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Internal distance buffer too short.");
            return;
        }

        double[] outD = new double[vc];
        Array.Copy(dist, outD, vc);

        if (normalise)
        {
            double lo = double.MaxValue, hi = double.MinValue;
            for (int i = 0; i < vc; i++)
            {
                double v = outD[i];
                if (v < lo) lo = v;
                if (v > hi) hi = v;
            }

            if (hi > lo + 1e-30)
            {
                double inv = 1.0 / (hi - lo);
                for (int i = 0; i < vc; i++)
                    outD[i] = (outD[i] - lo) * inv;
            }
            else
            {
                for (int i = 0; i < vc; i++)
                    outD[i] = 0;
            }
        }

        var ghD = new List<GH_Number>(vc);
        for (int i = 0; i < vc; i++)
            ghD.Add(new GH_Number(outD[i]));
        DA.SetDataList(0, ghD);

        Mesh colourMesh = MeshColourHelper.ColourByScalar(mesh, outD, normaliseMinMax: true);
        DA.SetData(1, colourMesh);
    }

    protected override Bitmap Icon => null!;

    public override Guid ComponentGuid => new("c3d4e5f6-a7b8-9012-cdef-123456789012");
}
