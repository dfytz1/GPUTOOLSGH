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
            "Mesh")
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

        int iterations = 8;
        DA.GetData("Iterations", ref iterations);
        if (iterations < 1)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Iterations must be at least 1.");
            return;
        }

        bool useGpu = true;
        DA.GetData("UseGPU", ref useGpu);

        if (!MeshLaplacianMetalSmooth.TrySmooth(
                this,
                meshIn,
                strengthD,
                iterations,
                new MeshLaplacianMetalSmooth.Options
                {
                    UseGpu = useGpu,
                    CpuFallbackIfGpuUnavailable = false,
                    WarnOnCpuFallback = false,
                },
                out Mesh? outMesh) || outMesh == null)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Laplacian smoothing could not run (empty topology or GPU error).");
            return;
        }

        DA.SetData("SmoothedMesh", outMesh);
    }

    protected override Bitmap Icon => null!;

    public override Guid ComponentGuid => new("42f238ab-96e9-4116-a191-7d5471df7f80");
}
