using System.Drawing;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;
using SpectralPacking.Core.Geometry;
using SpectralPacking.Core.Voxelization;

namespace SpectralPacking.GH.Components;

public sealed class GH_VoxelizeGeometry : GH_Component
{
    public GH_VoxelizeGeometry()
        : base("Voxelize Geometry GPU", "VoxGeomGPU",
            "Conservative mesh voxelization on a tray lattice (CPU; optional Metal column path planned).",
            "GPUTools", "Spectral Pack")
    {
    }

    public override Guid ComponentGuid => new("c4e8f2a1-9b3d-4e7c-8f1a-2d6b5e7c9a02");
    protected override Bitmap Icon => null!;

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddMeshParameter("InputMeshes", "M", "Meshes to voxelize", GH_ParamAccess.list);
        pManager.AddBoxParameter("TrayBox", "B", "Tray / lattice bounds", GH_ParamAccess.item);
        pManager.AddNumberParameter("VoxelSize", "Vx", "Cell size", GH_ParamAccess.item, 0.01);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddGenericParameter("VoxelGrids", "VG", "List of SpectralPacking.Core.Voxelization.VoxelGrid (wrapped)", GH_ParamAccess.list);
        pManager.AddBoxParameter("BoundingVolumes", "BV", "Per-mesh bounding boxes", GH_ParamAccess.list);
    }

    protected override void SolveInstance(IGH_DataAccess da)
    {
        var grids = new List<GH_ObjectWrapper>();
        var boxes = new List<Box>();

        var meshList = new List<Mesh>();
        Box trayBox = default;
        if (!da.GetDataList(0, meshList) || !da.GetData(1, ref trayBox))
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Meshes and TrayBox required.");
            da.SetDataList(0, grids);
            da.SetDataList(1, boxes);
            return;
        }
        double vx = 0.01;
        da.GetData(2, ref vx);
        if (vx <= 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "VoxelSize must be positive.");
            return;
        }

        var bb = trayBox.BoundingBox;
        var tray = new AxisAlignedBox(bb.Min.X, bb.Min.Y, bb.Min.Z, bb.Max.X, bb.Max.Y, bb.Max.Z);

        foreach (var mesh in meshList)
        {
            if (mesh == null || !mesh.IsValid)
                continue;
            var soup = RhinoMeshSoup.FromRhinoMesh(mesh);
            var grid = ConservativeVoxelizer.VoxelizeMesh(soup, tray, vx, fillSixWalls: false);
            grids.Add(new GH_ObjectWrapper(grid));
            var mb = soup.BoundingBox;
            boxes.Add(new Box(new BoundingBox(mb.MinX, mb.MinY, mb.MinZ, mb.MaxX, mb.MaxY, mb.MaxZ)));
        }

        da.SetDataList(0, grids);
        da.SetDataList(1, boxes);
    }
}
