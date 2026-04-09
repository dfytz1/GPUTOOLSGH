using System.Drawing;
using Grasshopper;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;
using SpectralPacking.Core.Disassembly;
using SpectralPacking.Core.Geometry;
using SpectralPacking.Core.Voxelization;

namespace SpectralPacking.GH.Components.DebugOnly;

public sealed class GH_DisassemblyCheck : GH_Component
{
    public GH_DisassemblyCheck()
        : base("Disassembly Check 3D GPU", "Disasm3DGPU",
            "Directional blocking graph and interlocking test on voxelized packed meshes.",
            "GPUTools", "Spectral Pack")
    {
    }

    public override Guid ComponentGuid => new("c4e8f2a1-9b3d-4e7c-8f1a-2d6b5e7c9a03");
    protected override Bitmap Icon => null!;

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddMeshParameter("PackedMeshes", "PM", "World-space packed meshes", GH_ParamAccess.tree);
        pManager.AddPlaneParameter("Placements", "Pl", "Unused if meshes are already transformed (reserved)", GH_ParamAccess.tree);
        pManager.AddNumberParameter("VoxelSize", "Vx", "Voxel size for DBG build", GH_ParamAccess.item, 0.01);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddBooleanParameter("IsInterlockFree", "OK", "True if no multi-node SCC in blocking graph", GH_ParamAccess.item);
        pManager.AddIntegerParameter("InterlockedPairs", "IP", "Sample pairs from cyclic SCCs (branch per SCC)", GH_ParamAccess.tree);
        pManager.AddIntegerParameter("DisassemblyOrder", "Ord", "Topological removal order if acyclic", GH_ParamAccess.tree);
    }

    protected override void SolveInstance(IGH_DataAccess da)
    {
        da.SetData(0, false);
        da.SetDataTree(1, new DataTree<int>());
        da.SetDataTree(2, new DataTree<int>());

        if (!da.GetDataTree(0, out GH_Structure<GH_Mesh>? meshStruct) || meshStruct == null)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "PackedMeshes tree missing.");
            return;
        }

        double voxelSize = 0.01;
        da.GetData(2, ref voxelSize);
        if (voxelSize <= 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "VoxelSize must be positive.");
            return;
        }

        var meshes = new List<Mesh>();
        foreach (var path in meshStruct.Paths)
        {
            foreach (var goo in meshStruct.get_Branch(path))
            {
                if (goo is GH_Mesh gm && gm.Value != null && gm.Value.IsValid)
                    meshes.Add(gm.Value);
            }
        }

        if (meshes.Count == 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "No meshes.");
            da.SetData(0, true);
            return;
        }

        var union = BoundingBox.Unset;
        foreach (var m in meshes)
            union.Union(m.GetBoundingBox(true));
        if (!union.IsValid)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Invalid mesh bounds.");
            return;
        }

        union.Inflate(voxelSize * 2);
        var tray = new AxisAlignedBox(union.Min.X, union.Min.Y, union.Min.Z, union.Max.X, union.Max.Y, union.Max.Z);
        double dx = voxelSize;
        int nx = Math.Max(1, (int)Math.Ceiling((tray.MaxX - tray.MinX) / dx));
        int ny = Math.Max(1, (int)Math.Ceiling((tray.MaxY - tray.MinY) / dx));
        int nz = Math.Max(1, (int)Math.Ceiling((tray.MaxZ - tray.MinZ) / dx));

        var omega = VoxelGrid.CreateZero(nx, ny, nz);
        ConservativeVoxelizer.MarkTrayWalls(omega, 1f);
        int n = omega.LinearSize;
        var owner = new int[n];
        for (int i = 0; i < n; i++)
            owner[i] = omega.Data[i] > 0.5f ? -2 : -1;

        for (int mi = 0; mi < meshes.Count; mi++)
        {
            var soup = RhinoMeshSoup.FromRhinoMesh(meshes[mi]);
            var mask = ConservativeVoxelizer.VoxelizeMesh(soup, tray, dx, fillSixWalls: false);
            for (int z = 0; z < nz; z++)
            for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++)
            {
                if (mask[x, y, z] <= 0.5f)
                    continue;
                int idx = omega.Index(x, y, z);
                if (owner[idx] == -2)
                    continue;
                omega.Data[idx] = 1f;
                owner[idx] = mi;
            }
        }

        int objectCount = meshes.Count;
        var adj = DirectionalBlockingGraph.BuildAdjacency(nx, ny, nz, owner, objectCount, tray, dx);
        var sccs = DirectionalBlockingGraph.FindSccsTarjan(objectCount, adj);
        bool ok = !sccs.Any(c => c.Count > 1);
        da.SetData(0, ok);

        var pairTree = new DataTree<int>();
        int br = 0;
        foreach (var comp in sccs.Where(c => c.Count > 1))
        {
            var path = new GH_Path(br++);
            for (int i = 0; i < comp.Count; i++)
                pairTree.Add(comp[i], path);
        }

        da.SetDataTree(1, pairTree);

        var orderTree = new DataTree<int>();
        if (FloodFillDisassembly.TryTopologicalRemovalOrder(adj, objectCount, out var ord))
        {
            for (int i = 0; i < ord.Count; i++)
                orderTree.Add(ord[i], new GH_Path(0));
        }

        da.SetDataTree(2, orderTree);
    }
}
