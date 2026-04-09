using GHGPUPlugin.Chromodoris.Topology;
using GHGPUPlugin.NativeInterop;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;
using System;
using System.Collections.Generic;

namespace GHGPUPlugin.Chromodoris;

/// <summary>
/// SIMP topology optimisation with automatic stride, PCG count, filter radius, and penalty continuation.
/// </summary>
public class VoxelSimpAutoComponent : GH_Component
{
    public VoxelSimpAutoComponent()
        : base("Voxel SIMP Auto GPU", "VoxelAutoGPU",
            "SIMP topology optimisation with automatic parameter tuning. Provide domain, supports, loads and target volume fraction only.",
            "GPUTools", "Voxel")
    {
    }

    protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
    {
        pManager.AddBoxParameter("BoundingBox", "B", "Same box as Voxel Design Domain.", GH_ParamAccess.item);
        pManager.AddGenericParameter("InsideMask", "I", "Domain mask from Voxel Design Domain.", GH_ParamAccess.item);
        pManager.AddGenericParameter("SupportMask", "S", "Fixed-displacement voxels.", GH_ParamAccess.item);
        pManager.AddGenericParameter("LoadMask", "L", "Loaded voxels (fallback if no LoadPoints).", GH_ParamAccess.item);
        pManager.AddNumberParameter("VolumeFraction", "Vf", "Target mean SIMP design variable on free voxels (0–1).", GH_ParamAccess.item, 0.3);
        pManager.AddPointParameter("LoadPoints", "LP", "Optional world-space load points (parallel to LoadVectors).", GH_ParamAccess.list);
        pManager.AddVectorParameter("LoadVectors", "LV", "Optional force vectors.", GH_ParamAccess.list);
        pManager.AddPointParameter("SupportPoints", "SP", "Optional directional support points.", GH_ParamAccess.list);
        pManager.AddVectorParameter("SupportDirs", "SD", "Optional support direction flags per point.", GH_ParamAccess.list);
        pManager.AddBooleanParameter("UseGPU", "GPU", "Use Metal for PCG MatVec when available.", GH_ParamAccess.item, true);
        for (int i = 5; i <= 9; i++)
            pManager[i].Optional = true;
    }

    protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
    {
        pManager.AddGenericParameter("Density", "R", "float[x,y,z] upsampled density — plug into IsoSurface.", GH_ParamAccess.item);
        pManager.AddBoxParameter("BoundingBox", "B", "Passthrough for Build IsoSurface.", GH_ParamAccess.item);
        pManager.AddNumberParameter("Compliance", "C", "Final linear compliance f·u (relative units).", GH_ParamAccess.item);
        pManager.AddTextParameter("AutoParams", "AP", "Auto-derived solver parameters.", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        void FallbackOutputs(Box bbox)
        {
            DA.SetData(0, null);
            DA.SetData(1, bbox);
            DA.SetData(2, 0.0);
            DA.SetData(3, string.Empty);
        }

        Box box = new Box();
        float[,,] inside = null, support = null, load = null;
        double vf = 0.3;
        bool useGpu = true;
        var loadPts = new List<Point3d>();
        var loadVecs = new List<Vector3d>();
        var supPts = new List<Point3d>();
        var supDirs = new List<Vector3d>();

        if (!DA.GetData(0, ref box))
        {
            FallbackOutputs(new Box());
            return;
        }

        if (!VoxelMaskGoo.TryGetFloatTensor3(DA, 1, this, out inside, "InsideMask"))
        {
            FallbackOutputs(box);
            return;
        }

        if (!VoxelMaskGoo.TryGetFloatTensor3(DA, 2, this, out support, "SupportMask"))
        {
            FallbackOutputs(box);
            return;
        }

        if (!VoxelMaskGoo.TryGetFloatTensor3(DA, 3, this, out load, "LoadMask"))
        {
            FallbackOutputs(box);
            return;
        }

        DA.GetData(4, ref vf);
        DA.GetDataList(5, loadPts);
        DA.GetDataList(6, loadVecs);
        DA.GetDataList(7, supPts);
        DA.GetDataList(8, supDirs);
        DA.GetData(9, ref useGpu);

        if (useGpu)
        {
            try
            {
                NativeLoader.EnsureLoaded();
                if (!NativeLoader.IsMetalAvailable)
                {
                    useGpu = false;
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                        "GPU native library unavailable, using CPU: " + (NativeLoader.LoadError ?? "MetalBridge.dylib not loaded."));
                }
            }
            catch (Exception ex)
            {
                useGpu = false;
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                    "GPU native library unavailable, using CPU: " + ex.Message);
            }
        }

        int nx = inside.GetLength(0), ny = inside.GetLength(1), nz = inside.GetLength(2);
        if (support.GetLength(0) != nx || load.GetLength(0) != nx)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Mask dimensions must match.");
            FallbackOutputs(box);
            return;
        }

        if (vf <= 0 || vf >= 1)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "VolumeFraction must be in (0,1).");
            FallbackOutputs(box);
            return;
        }

        bool hasLp = loadPts.Count > 0;
        if (hasLp && loadPts.Count != loadVecs.Count)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "LoadPoints and LoadVectors must have the same count.");
            FallbackOutputs(box);
            return;
        }

        if (supPts.Count > 0 && supPts.Count != supDirs.Count)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "SupportPoints and SupportDirs must have the same count.");
            FallbackOutputs(box);
            return;
        }

        bool loadMaskEmpty = MaskHasNoMarkedVoxels(load);
        if (!hasLp && loadMaskEmpty)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "No loads defined.");
            FallbackOutputs(box);
            return;
        }

        bool supportMaskEmpty = MaskHasNoMarkedVoxels(support);
        if (supportMaskEmpty && supPts.Count == 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "No supports defined.");
            FallbackOutputs(box);
            return;
        }

        var sumLv = Vector3d.Zero;
        foreach (Vector3d v in loadVecs)
            sumLv += v;
        if (!hasLp && sumLv.Length < 1e-20)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "No loads defined.");
            FallbackOutputs(box);
            return;
        }

        int totalVoxels = nx * ny * nz;
        int solveStride = 1;
        if (totalVoxels > 50000) solveStride = 2;
        if (totalVoxels > 200000) solveStride = 3;
        if (totalVoxels > 500000) solveStride = 4;

        int cnx = nx / solveStride;
        int cny = ny / solveStride;
        int cnz = nz / solveStride;
        int approxElem = cnx * cny * cnz;
        int ndofEstimate = approxElem * 3;
        int pcgIterations = Math.Max(200, (int)(2.5 * Math.Sqrt(ndofEstimate)));
        pcgIterations = Math.Min(pcgIterations, 1500);

        const int outerIterations = 45;
        const double moveLimit = 0.15;
        const double voidStiffness = 1e-6;
        const double poisson = 0.3;
        const double youngModulus = 1.0;
        const int maxElements = 80000;
        double filterRadius = Math.Max(1.2, 1.5 * solveStride);
        const double simpPenaltyNominal = 3.0;

        string autoParams =
            $"AutoParams: stride={solveStride} pcgIter={pcgIterations} " +
            $"filterR={filterRadius:F1} outerIter=45 continuation=true";

        AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, autoParams);

        double dx = box.X.Length / nx;
        double dy = box.Y.Length / ny;
        double dz = box.Z.Length / nz;

        VoxelSimpOptimizer.Result res;
        try
        {
            res = VoxelSimpOptimizer.Run(
                inside, support, load, dx, dy, dz,
                box, loadPts, loadVecs, youngModulus, supPts, supDirs,
                vf,
                outerIterations, pcgIterations, simpPenaltyNominal, moveLimit, voidStiffness, poisson, maxElements, solveStride,
                useGpu, recordHistory: false, filterRadius, penaltyContinuation: true);
        }
        catch (Exception ex)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, ex.Message);
            FallbackOutputs(box);
            return;
        }

        if (!string.IsNullOrEmpty(res.Message) && res.Message.StartsWith("GPU_FALLBACK:", StringComparison.Ordinal))
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, res.Message);
        else if (!string.IsNullOrEmpty(res.Message) && res.Message.StartsWith("GPU_REMARK:", StringComparison.Ordinal))
            AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, res.Message);
        else if (res.Message != "OK")
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, res.Message);
            FallbackOutputs(box);
            return;
        }

        AddRuntimeMessage(GH_RuntimeMessageLevel.Remark,
            "Voxel SIMP Auto: tuned stride / PCG / filter / penalty continuation — not sign-off FEA.");

        if (!string.IsNullOrWhiteSpace(res.DiagMessage))
            AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, res.DiagMessage);

        if (!string.IsNullOrWhiteSpace(res.GpuDiagPreSolve))
            AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, res.GpuDiagPreSolve);

        DA.SetData(0, new GH_ObjectWrapper(res.DensityPhys));
        DA.SetData(1, box);
        DA.SetData(2, res.Compliance);
        DA.SetData(3, autoParams);
    }

    private static bool MaskHasNoMarkedVoxels(float[,,] mask)
    {
        int nx = mask.GetLength(0), ny = mask.GetLength(1), nz = mask.GetLength(2);
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < ny; j++)
                for (int k = 0; k < nz; k++)
                    if (mask[i, j, k] >= 0.5f)
                        return false;
        return true;
    }

    public override GH_Exposure Exposure => GH_Exposure.tertiary;

    protected override System.Drawing.Bitmap Icon => null;

    public override Guid ComponentGuid => new Guid("e4d8912a-3c7b-4f6e-9a1d-8b2c5e6f7089");
}
