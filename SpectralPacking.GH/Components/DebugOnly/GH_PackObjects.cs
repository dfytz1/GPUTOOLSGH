using System.Drawing;
using Grasshopper;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;
using SpectralPacking.Core.Geometry;
using SpectralPacking.Core.Metrics;
using SpectralPacking.Core.Packing;
using SpectralPacking.Core.Placement;
using SpectralPacking.GH.Interop;

namespace SpectralPacking.GH.Components.DebugOnly;

public sealed class GH_PackObjects : GH_Component
{
    public GH_PackObjects()
        : base("Pack Objects 3D GPU", "Pack3DGPU",
            "Greedy spectral 3D packing (Cui et al. TOG 2023) with FFT placement and interlocking check.",
            "GPUTools", "Spectral Pack")
    {
    }

    public override Guid ComponentGuid => new("c4e8f2a1-9b3d-4e7c-8f1a-2d6b5e7c9a01");
    protected override Bitmap Icon => null!;

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddMeshParameter("InputMeshes", "M", "Meshes to pack", GH_ParamAccess.tree);
        pManager.AddBoxParameter("TrayBox", "B", "Axis-aligned packing tray", GH_ParamAccess.item);
        pManager.AddNumberParameter("VoxelSize", "Vx", "Voxel edge length", GH_ParamAccess.item, 0.01);
        pManager.AddIntegerParameter("OrientationCount", "Ori", "Number of sampled orientations", GH_ParamAccess.item, 72);
        pManager.AddNumberParameter("GravityWeight", "G", "Penalty weight for high Z placements", GH_ParamAccess.item, 0);
        pManager.AddBooleanParameter("EnableInterlockCheck", "IL", "Resolve interlocking SCCs after packing", GH_ParamAccess.item, true);
        pManager.AddBooleanParameter("UseGPU", "GPU", "Native BFS distance field via Metal device context. 3D FFT correlation uses Accelerate vDSP in MetalBridge.dylib whenever the dylib loads (independent of this flag).", GH_ParamAccess.item, true);
        pManager.AddBooleanParameter("UseParallel", "||", "Parallel.For over orientations (CPU)", GH_ParamAccess.item, false);
        pManager.AddBooleanParameter("Run", "Run", "Execute solve", GH_ParamAccess.item, true);
        pManager.AddIntegerParameter("OrientationMode", "Om", "0 = uniform Euler, 1 = icosphere mix", GH_ParamAccess.item, 0);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddMeshParameter("PackedMeshes", "PM", "One mesh per packed object", GH_ParamAccess.tree);
        pManager.AddPlaneParameter("Placements", "Pl", "Local placement frame per object", GH_ParamAccess.tree);
        pManager.AddMeshParameter("UnpackedMeshes", "UM", "Objects that did not fit or stayed unpacked", GH_ParamAccess.list);
        pManager.AddNumberParameter("PackingDensity", "ρ", "Solid volume / tray volume", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess da)
    {
        var emptyMeshTree = new DataTree<Mesh>();
        var emptyPlaneTree = new DataTree<Plane>();
        da.SetDataTree(0, emptyMeshTree);
        da.SetDataTree(1, emptyPlaneTree);
        da.SetDataList(2, new List<Mesh>());
        da.SetData(3, 0.0);

        if (!da.GetDataTree(0, out GH_Structure<GH_Mesh>? meshStruct) || meshStruct == null)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "InputMeshes tree is missing.");
            return;
        }

        Box trayBox = default;
        if (!da.GetData(1, ref trayBox))
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "TrayBox is missing.");
            return;
        }

        double voxelSize = 0.01;
        da.GetData(2, ref voxelSize);
        if (voxelSize <= 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "VoxelSize must be positive.");
            return;
        }

        if (voxelSize < 0.05)
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                "VoxelSize is small; the voxel grid may become very large. Consider increasing VoxelSize (e.g. ≥ 0.05) for faster, more stable solves.");

        int orientationCount = 72;
        da.GetData(3, ref orientationCount);
        orientationCount = Math.Max(4, orientationCount);

        double gravityWeight = 0;
        da.GetData(4, ref gravityWeight);

        bool enableInterlock = true;
        da.GetData(5, ref enableInterlock);

        bool useGpu = true;
        da.GetData(6, ref useGpu);

        bool useParallel = false;
        da.GetData(7, ref useParallel);

        bool run = true;
        da.GetData(8, ref run);

        int orientationMode = 0;
        da.GetData(9, ref orientationMode);

        if (!run)
            return;

        var inputMeshes = new List<Mesh>();
        foreach (var path in meshStruct.Paths)
        {
            var branchList = meshStruct.get_Branch(path);
            foreach (var goo in branchList)
            {
                if (goo is GH_Mesh gm && gm.Value != null && gm.Value.IsValid)
                    inputMeshes.Add(gm.Value);
            }
        }

        if (inputMeshes.Count == 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "No valid meshes in tree.");
            return;
        }

        var bbox = trayBox.BoundingBox;
        if (!bbox.IsValid)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Invalid tray box.");
            return;
        }

        var tray = new AxisAlignedBox(bbox.Min.X, bbox.Min.Y, bbox.Min.Z, bbox.Max.X, bbox.Max.Y, bbox.Max.Z);
        var soups = inputMeshes.Select(RhinoMeshSoup.FromRhinoMesh).ToList();

        IntPtr ctx = IntPtr.Zero;
        bool gpuOk = useGpu && NativeLoader.IsMetalAvailable && MetalSharedContext.TryGetContext(out ctx);
        if (useGpu && !gpuOk)
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"GPU distance field unavailable: {MetalSharedContext.InitError ?? NativeLoader.LoadError ?? "unknown"}");

        // vDSP 3D FFT lives in MetalBridge.dylib and does not require a Metal device; always try native correlation first.
        IFFTBackend fft = new MetalFFTBackend();
        var mode = orientationMode != 0 ? OrientationSamplingMode.Icosphere : OrientationSamplingMode.UniformEuler;

        SpectralPackResult result;
        try
        {
            result = GreedyPacker.Pack(
                soups,
                tray,
                voxelSize,
                orientationCount,
                gravityWeight,
                enableInterlock,
                gpuOk,
                ctx,
                useParallel,
                fft,
                mode,
                refinementIterations: 4);
        }
        catch (Exception ex)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, ex.Message);
            return;
        }

        var packedTree = new DataTree<Mesh>();
        var planeTree = new DataTree<Rhino.Geometry.Plane>();
        int outBranch = 0;
        for (int k = 0; k < result.PackedIndices.Count; k++)
        {
            int idx = result.PackedIndices[k];
            var mIn = inputMeshes[idx];
            var R = result.Rotations[k];
            var t = result.Translations[k];
            var path = new GH_Path(outBranch);
            packedTree.Add(PackedMeshBuilder.Build(mIn, R, t), path);
            planeTree.Add(PackedMeshBuilder.ToPlacementPlane(mIn, R, t), path);
            outBranch++;
        }

        var unpacked = result.UnpackedIndices.Select(i => inputMeshes[i].DuplicateMesh()).ToList();

        da.SetDataTree(0, packedTree);
        da.SetDataTree(1, planeTree);
        da.SetDataList(2, unpacked);
        da.SetData(3, result.PackingDensity);

        if (result.UsedBlockingGraphFallbackOrder)
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                "Blocking graph cycle force-broken via fallback ordering. Results may be suboptimal. Try increasing OrientationCount.");

        if (!result.IsInterlockFree)
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Blocking graph may still contain cycles after resolution.");

        if (result.RestoredPackAfterInterlockFailure)
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                "Interlock resolution removed all placements; restored the pre-resolution pack. Consider more orientations or disabling interlock resolution.");

        if (result.PackedIndices.Count == 0)
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                "No meshes were packed. Increase tray size or VoxelSize, add OrientationCount, or ensure meshes fit inside the tray.");
    }
}
