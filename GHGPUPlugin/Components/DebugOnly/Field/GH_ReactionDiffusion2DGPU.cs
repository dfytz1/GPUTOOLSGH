using System;
using System.Collections.Generic;
using System.Drawing;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;

namespace GHGPUPlugin.Components.Field;

/// <summary>Gray–Scott reaction–diffusion on a 2D grid (Metal GPU, CPU fallback). Inspired by classic WebGL demos such as
/// <see href="https://github.com/jasonwebb/reaction-diffusion-playground">reaction-diffusion-playground</see>.</summary>
public class GH_ReactionDiffusion2DGPU : GH_Component
{
    public GH_ReactionDiffusion2DGPU()
        : base(
            "Reaction Diffusion 2D GPU",
            "ReactDiff2D",
            "Gray–Scott reaction–diffusion on a plane (Metal). Seeds: points (disks), curves (tube along curve, radius Sr), optional B0. Outputs A, B as float[nx,ny] and optional quad mesh.",
            "GPUTools",
            "Field")
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddIntegerParameter("ResolutionX", "Nx", "Grid width (cells).", GH_ParamAccess.item, 128);
        pManager.AddIntegerParameter("ResolutionY", "Ny", "Grid height (cells).", GH_ParamAccess.item, 128);
        pManager.AddIntegerParameter("Iterations", "N", "Simulation steps per solve.", GH_ParamAccess.item, 800);
        pManager.AddNumberParameter("Feed", "f", "Feed rate.", GH_ParamAccess.item, 0.055);
        pManager.AddNumberParameter("Kill", "k", "Kill rate.", GH_ParamAccess.item, 0.062);
        pManager.AddNumberParameter("DiffusionA", "dA", "Diffusion rate for A.", GH_ParamAccess.item, 1.0);
        pManager.AddNumberParameter("DiffusionB", "dB", "Diffusion rate for B.", GH_ParamAccess.item, 0.5);
        pManager.AddNumberParameter("TimeStep", "dt", "Explicit Euler step (try 1; reduce if unstable).", GH_ParamAccess.item, 1.0);
        pManager.AddPlaneParameter("Plane", "Pl", "Domain on Origin + SizeX·X + SizeY·Y (axes unitized internally).", GH_ParamAccess.item,
            Plane.WorldXY);
        pManager.AddNumberParameter("SizeX", "Sx", "Domain extent along plane X axis.", GH_ParamAccess.item, 1.0);
        pManager.AddNumberParameter("SizeY", "Sy", "Domain extent along plane Y axis.", GH_ParamAccess.item, 1.0);
        pManager.AddPointParameter("SeedPoints", "Pt", "Optional points; B is set to 1 within SeedRadius (world units).", GH_ParamAccess.list);
        pManager.AddNumberParameter("SeedRadius", "Sr", "World radius for point seeds and curve seeds.", GH_ParamAccess.item, 0.05);
        pManager.AddGenericParameter("InitialB", "B0", "Optional float[nx,ny] initial B; must match resolution.", GH_ParamAccess.item);
        pManager.AddBooleanParameter("OutputMesh", "Mesh", "Build a preview mesh colored by B.", GH_ParamAccess.item, true);
        pManager.AddBooleanParameter("UseGPU", "GPU", "Use Metal when available.", GH_ParamAccess.item, true);
        pManager.AddCurveParameter("SeedCurves", "Cv", "Optional curves; cells within Sr of the curve get B = 1.", GH_ParamAccess.list);
        pManager[11].Optional = true;
        pManager[13].Optional = true;
        pManager[16].Optional = true;
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddGenericParameter("A", "A", "float[nx,ny] chemical A.", GH_ParamAccess.item);
        pManager.AddGenericParameter("B", "B", "float[nx,ny] chemical B (pattern).", GH_ParamAccess.item);
        pManager.AddMeshParameter("Mesh", "M", "Quad mesh on the plane, vertex colors from B.", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        int nx = 128, ny = 128, nIters = 800;
        double f = 0.055, k = 0.062, dA = 1.0, dB = 0.5, dt = 1.0;
        var plane = Plane.WorldXY;
        double sx = 1.0, sy = 1.0, seedR = 0.05;
        var seeds = new List<Point3d>();
        var curves = new List<Curve>();
        bool outMesh = true, useGpu = true;

        if (!DA.GetData(0, ref nx)) return;
        if (!DA.GetData(1, ref ny)) return;
        if (!DA.GetData(2, ref nIters)) return;
        if (!DA.GetData(3, ref f)) return;
        if (!DA.GetData(4, ref k)) return;
        if (!DA.GetData(5, ref dA)) return;
        if (!DA.GetData(6, ref dB)) return;
        if (!DA.GetData(7, ref dt)) return;
        if (!DA.GetData(8, ref plane)) return;
        if (!DA.GetData(9, ref sx)) return;
        if (!DA.GetData(10, ref sy)) return;
        DA.GetDataList(11, seeds);
        DA.GetData(12, ref seedR);
        DA.GetData(14, ref outMesh);
        DA.GetData(15, ref useGpu);
        DA.GetDataList(16, curves);

        float[,]? initialB = null;
        IGH_Goo? b0Goo = null;
        if (DA.GetData(13, ref b0Goo) && b0Goo != null)
        {
            if (!Field2DDataHelper.TryUnwrapFloat2D(b0Goo, out initialB, out string err))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, err);
                return;
            }

            if (initialB!.GetLength(0) != nx || initialB.GetLength(1) != ny)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "InitialB dimensions must match Nx×Ny.");
                return;
            }
        }

        if (!GrayScottField2DSolver.TrySolve(
                this, nx, ny, nIters, f, k, dA, dB, dt, plane, sx, sy, seeds, curves, seedR, initialB,
                useDefaultCenterSeed: true, meshSeedVertices: null, useGpu, out float[,] aOut, out float[,] bOut))
            return;

        DA.SetData(0, new GH_ObjectWrapper(aOut));
        DA.SetData(1, new GH_ObjectWrapper(bOut));

        if (outMesh)
            DA.SetData(2, Field2DPreviewMesh.BuildMesh(bOut, plane, sx, sy, normalizeColors: true));
        else
            DA.SetData(2, null);
    }

    protected override Bitmap Icon => null!;

    public override Guid ComponentGuid => new("e7a1c4b2-3f5d-4e6a-9b0c-1d2e3f4a5b6c");
}
