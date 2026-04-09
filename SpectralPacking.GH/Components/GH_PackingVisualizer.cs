using System.Drawing;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;

namespace SpectralPacking.GH.Components;

public sealed class GH_PackingVisualizer : GH_Component
{
    public GH_PackingVisualizer()
        : base("Packing Visualizer 3D GPU", "PackVisGPU",
            "Duplicate packed meshes with optional per-object colors.",
            "GPUTools", "Spectral Pack")
    {
    }

    public override Guid ComponentGuid => new("c4e8f2a1-9b3d-4e7c-8f1a-2d6b5e7c9a04");
    protected override Bitmap Icon => null!;

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddMeshParameter("PackedMeshes", "PM", "Packed meshes", GH_ParamAccess.tree);
        pManager.AddPlaneParameter("Placements", "Pl", "Placement frames (optional, not used for geometry)", GH_ParamAccess.tree);
        pManager.AddBoxParameter("TrayBox", "B", "Tray for preview context (optional)", GH_ParamAccess.item);
        pManager.AddBooleanParameter("ColorByIndex", "Col", "Assign color from branch index", GH_ParamAccess.item, true);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddMeshParameter("TransformedMeshes", "M", "Mesh copies for display", GH_ParamAccess.list);
        pManager.AddColourParameter("Colors", "C", "Per-mesh color", GH_ParamAccess.list);
    }

    protected override void SolveInstance(IGH_DataAccess da)
    {
        var outMeshes = new List<Mesh>();
        var outColors = new List<Color>();

        if (!da.GetDataTree(0, out GH_Structure<GH_Mesh>? meshStruct) || meshStruct == null)
        {
            da.SetDataList(0, outMeshes);
            da.SetDataList(1, outColors);
            return;
        }

        Box trayBox = default;
        da.GetData(2, ref trayBox);
        bool colorByIndex = true;
        da.GetData(3, ref colorByIndex);

        int idx = 0;
        foreach (var path in meshStruct.Paths)
        {
            foreach (var goo in meshStruct.get_Branch(path))
            {
                if (goo is not GH_Mesh gm || gm.Value == null || !gm.Value.IsValid)
                    continue;
                outMeshes.Add(gm.Value.DuplicateMesh());
                outColors.Add(colorByIndex ? ColorFromIndex(idx) : Color.Gray);
                idx++;
            }
        }

        da.SetDataList(0, outMeshes);
        da.SetDataList(1, outColors);
    }

    private static Color ColorFromIndex(int i)
    {
        float t = (i * 0.618033988749895f) % 1f;
        int h = (int)(t * 360);
        return ColorFromHsv(h, 0.65f, 0.95f);
    }

    private static Color ColorFromHsv(double h, double s, double v)
    {
        int hi = (int)Math.Floor(h / 60.0) % 6;
        double f = h / 60.0 - Math.Floor(h / 60.0);
        double p = v * (1 - s);
        double q = v * (1 - f * s);
        double t = v * (1 - (1 - f) * s);
        double r, g, b;
        switch (hi)
        {
            case 0: r = v; g = t; b = p; break;
            case 1: r = q; g = v; b = p; break;
            case 2: r = p; g = v; b = t; break;
            case 3: r = p; g = q; b = v; break;
            case 4: r = t; g = p; b = v; break;
            default: r = v; g = p; b = q; break;
        }

        return Color.FromArgb(255,
            (int)(r * 255),
            (int)(g * 255),
            (int)(b * 255));
    }
}
