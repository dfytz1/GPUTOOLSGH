/*
 * Based on ChromodorisGH by Cameron Newnham (GPL-3.0)
 * https://github.com/camnewnham/ChromodorisGH
 */

using GHGPUPlugin.Chromodoris.MeshTools;
using GHGPUPlugin.MeshTopology;
using GHGPUPlugin.NativeInterop;
using Grasshopper.Kernel;
using Rhino.Geometry;

namespace GHGPUPlugin.Chromodoris;

public class QuickSmoothComponent : GH_Component
{
    public QuickSmoothComponent()
        : base(
            "QuickSmooth GPU",
            "SmoothGPU",
            "Laplacian vertex smoothing: Metal umbrella Laplacian when UseGPU is on; otherwise Chromodoris-style multithreaded CPU.",
            "GPUTools",
            "Mesh")
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddMeshParameter("Mesh", "M", "The mesh to smooth.", GH_ParamAccess.item);
        pManager.AddNumberParameter("StepSize", "S", "Smoothing step size between 0 and 1.", GH_ParamAccess.item, 0.5);
        pManager.AddIntegerParameter("Iterations", "I", "Number of smoothing iterations.", GH_ParamAccess.item, 1);
        pManager.AddBooleanParameter("UseGPU", "GPU", "Use Metal Laplacian when available (Mac).", GH_ParamAccess.item, true);
        pManager[1].Optional = true;
        pManager[2].Optional = true;
        pManager[3].Optional = true;
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddMeshParameter("SmoothedMesh", "M", "The smoothed mesh.", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        NativeLoader.EnsureLoaded();

        Mesh? mesh = null;
        double step = 0.5;
        int iterations = 1;
        bool useGpu = true;

        if (!DA.GetData(0, ref mesh) || mesh == null)
            return;
        DA.GetData(1, ref step);
        DA.GetData(2, ref iterations);
        DA.GetData(3, ref useGpu);

        if (iterations < 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Iterations must be 0 or greater.");
            return;
        }

        if (step < 0 || step > 1)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "StepSize must be between 0 and 1.");
            return;
        }

        if (iterations == 0 || step == 0)
        {
            DA.SetData(0, mesh);
            return;
        }

        if (useGpu &&
            MeshLaplacianMetalSmooth.TrySmooth(
                this,
                mesh,
                step,
                iterations,
                new MeshLaplacianMetalSmooth.Options
                {
                    UseGpu = true,
                    CpuFallbackIfGpuUnavailable = true,
                    WarnOnCpuFallback = true,
                },
                out Mesh? metalOut) &&
            metalOut != null)
        {
            DA.SetData(0, metalOut);
            return;
        }

        var smooth = new VertexSmooth(mesh, step, iterations);
        DA.SetData(0, smooth.Compute());
    }

    public override GH_Exposure Exposure => GH_Exposure.quinary;

    protected override System.Drawing.Bitmap Icon => Icons.Smooth;

    public override Guid ComponentGuid => new("d43803b8-bc17-4172-aeb0-098124a8391a");
}
