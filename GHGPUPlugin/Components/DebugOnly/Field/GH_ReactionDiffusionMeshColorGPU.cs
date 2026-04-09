using System;
using System.Collections.Generic;
using System.Drawing;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;

namespace GHGPUPlugin.Components.Field;

/// <summary>Gray–Scott on a plane fitted to the mesh (or optional plane), domain sized to projected bounds; user scales resolution, domain padding, and seed radius.</summary>
public class GH_ReactionDiffusionMeshColorGPU : GH_Component
{
    public GH_ReactionDiffusionMeshColorGPU()
        : base(
            "Reaction Diffusion Mesh Color GPU",
            "ReactDiffMesh",
            "Fits simulation domain to the mesh (plane + Sx×Sy from projected bounds). FitPl on = auto plane; off = manual Pl. Multipliers: ResM, DomM, SrM. Outputs DPl, Sx, Sy for Mesh Displace Field 2D GPU.",
            "GPUTools",
            "Field")
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddMeshParameter("Mesh", "M", "Mesh to color (duplicated).", GH_ParamAccess.item);
        pManager.AddIntegerParameter("Resolution", "Res", "Cell count along the shorter domain side (longer side scales with aspect).", GH_ParamAccess.item, 128);
        pManager.AddNumberParameter("ResolutionMul", "ResM", "Multiplier on derived Nx, Ny.", GH_ParamAccess.item, 1.0);
        pManager.AddIntegerParameter("Iterations", "N", "Simulation steps.", GH_ParamAccess.item, 800);
        pManager.AddNumberParameter("Feed", "f", "Feed rate.", GH_ParamAccess.item, 0.055);
        pManager.AddNumberParameter("Kill", "k", "Kill rate.", GH_ParamAccess.item, 0.062);
        pManager.AddNumberParameter("DiffusionA", "dA", "Diffusion rate for A.", GH_ParamAccess.item, 1.0);
        pManager.AddNumberParameter("DiffusionB", "dB", "Diffusion rate for B.", GH_ParamAccess.item, 0.5);
        pManager.AddNumberParameter("TimeStep", "dt", "Explicit Euler step.", GH_ParamAccess.item, 1.0);
        pManager.AddBooleanParameter("FitPlane", "FitPl", "If true, plane is least-squares fit to mesh (recommended). If false, use Pl.", GH_ParamAccess.item, true);
        pManager.AddPlaneParameter("Plane", "Pl", "Used only when FitPl is false.", GH_ParamAccess.item, Plane.WorldXY);
        pManager.AddNumberParameter("DomainScale", "DomM", "Multiplies fitted width/height (1 = tight bbox, >1 = padding).", GH_ParamAccess.item, 1.12);
        pManager.AddNumberParameter("SeedRadiusMul", "SrM", "Multiplies auto seed radius (~3% of max(Sx,Sy)).", GH_ParamAccess.item, 1.0);
        pManager.AddPointParameter("SeedPoints", "Pt", "Optional extra point seeds.", GH_ParamAccess.list);
        pManager.AddCurveParameter("SeedCurves", "Cv", "Optional extra curve seeds.", GH_ParamAccess.list);
        pManager.AddGenericParameter("InitialB", "B0", "Optional float[nx,ny].", GH_ParamAccess.item);
        pManager.AddBooleanParameter("SeedFromMesh", "SeedM", "Seed B from projected mesh vertices (recommended).", GH_ParamAccess.item, true);
        pManager.AddBooleanParameter("NormalizeColors", "NormC", "Map sampled B at vertices to full color range.", GH_ParamAccess.item, true);
        pManager.AddBooleanParameter("UseGPU", "GPU", "Use Metal when available.", GH_ParamAccess.item, true);
        pManager[10].Optional = true;
        pManager[13].Optional = true;
        pManager[14].Optional = true;
        pManager[15].Optional = true;
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddMeshParameter("Mesh", "M", "Colored mesh.", GH_ParamAccess.item);
        pManager.AddGenericParameter("A", "A", "float[nx,ny] chemical A.", GH_ParamAccess.item);
        pManager.AddGenericParameter("B", "B", "float[nx,ny] chemical B.", GH_ParamAccess.item);
        pManager.AddPlaneParameter("DomainPlane", "DPl", "Simulation plane origin (corner); wire to Mesh Displace.", GH_ParamAccess.item);
        pManager.AddNumberParameter("SizeX", "Sx", "Domain extent X; wire to Mesh Displace.", GH_ParamAccess.item);
        pManager.AddNumberParameter("SizeY", "Sy", "Domain extent Y; wire to Mesh Displace.", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        Mesh? meshIn = null;
        if (!DA.GetData(0, ref meshIn) || meshIn == null || !meshIn.IsValid)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Valid mesh required.");
            return;
        }

        int res = 128, nIters = 800;
        double resMul = 1.0, f = 0.055, k = 0.062, dA = 1.0, dB = 0.5, dt = 1.0;
        double domainScale = 1.12, seedRadMul = 1.0;
        var seeds = new List<Point3d>();
        var curves = new List<Curve>();
        bool seedFromMesh = true, normColors = true, useGpu = true;

        if (!DA.GetData(1, ref res)) return;
        if (!DA.GetData(2, ref resMul)) return;
        if (!DA.GetData(3, ref nIters)) return;
        if (!DA.GetData(4, ref f)) return;
        if (!DA.GetData(5, ref k)) return;
        if (!DA.GetData(6, ref dA)) return;
        if (!DA.GetData(7, ref dB)) return;
        if (!DA.GetData(8, ref dt)) return;

        bool fitPlane = true;
        DA.GetData(9, ref fitPlane);

        Plane? userPlane = null;
        if (!fitPlane)
        {
            Plane plTmp = Plane.Unset;
            if (!DA.GetData(10, ref plTmp) || !plTmp.IsValid)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "FitPl is false but Pl is missing or invalid.");
                return;
            }

            userPlane = plTmp;
        }

        if (!DA.GetData(11, ref domainScale)) return;
        if (!DA.GetData(12, ref seedRadMul)) return;
        DA.GetDataList(13, seeds);
        DA.GetDataList(14, curves);
        DA.GetData(16, ref seedFromMesh);
        DA.GetData(17, ref normColors);
        DA.GetData(18, ref useGpu);

        if (!MeshFieldDomainFit.TryReferencePlane(meshIn, userPlane, out Plane refPlane, out string fitMsg))
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, fitMsg);
            return;
        }

        if (!string.IsNullOrEmpty(fitMsg))
            AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, fitMsg);

        if (!MeshFieldDomainFit.TryDomainFromMesh(meshIn, refPlane, domainScale, out Plane domainPlane, out double sx, out double sy))
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Could not build domain from mesh (degenerate extent).");
            return;
        }

        MeshFieldDomainFit.ResolutionForDomain(sx, sy, res, resMul, out int nx, out int ny);

        double seedR = MeshFieldDomainFit.AutoSeedRadius(sx, sy, seedRadMul);

        float[,]? initialB = null;
        IGH_Goo? b0Goo = null;
        if (DA.GetData(15, ref b0Goo) && b0Goo != null)
        {
            if (!Field2DDataHelper.TryUnwrapFloat2D(b0Goo, out initialB, out string err))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, err);
                return;
            }

            if (initialB!.GetLength(0) != nx || initialB.GetLength(1) != ny)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"InitialB must be float[{nx},{ny}] (derived from mesh).");
                return;
            }
        }

        Mesh? meshForSeed = seedFromMesh ? meshIn : null;
        if (!GrayScottField2DSolver.TrySolve(
                this, nx, ny, nIters, f, k, dA, dB, dt, domainPlane, sx, sy, seeds, curves, seedR, initialB,
                useDefaultCenterSeed: true, meshSeedVertices: meshForSeed, useGpu, out float[,] aOut, out float[,] bOut))
            return;

        Mesh colored = meshIn.DuplicateMesh();
        Field2DMeshVertexPaint.ApplyScalarFieldVertexSampledRange(colored, bOut, nx, ny, domainPlane, sx, sy, normColors);

        DA.SetData(0, colored);
        DA.SetData(1, new GH_ObjectWrapper(aOut));
        DA.SetData(2, new GH_ObjectWrapper(bOut));
        DA.SetData(3, domainPlane);
        DA.SetData(4, sx);
        DA.SetData(5, sy);
    }

    protected override Bitmap Icon => null!;

    public override Guid ComponentGuid => new("a3c8e901-2b4f-4d8e-9c1a-5e6f7d8b9a0c");
}
