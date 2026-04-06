using System.Drawing;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;
using GHGPUPlugin.Algorithms;
using Rhino.Geometry;

namespace GHGPUPlugin.Components.Smoothing;

public class GH_MeshIsolines : GH_Component
{
    public GH_MeshIsolines()
        : base(
            "Mesh Isolines GPU",
            "MeshIsoGPU",
            "Extract isolines from a per-vertex scalar field by edge marching and segment chaining. Mesh Geodesic Distance GPU: wire its M→M and S→S.",
            "GPUTools",
            "Mesh")
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddMeshParameter("Mesh", "M", "Mesh.", GH_ParamAccess.item);
        pManager.AddNumberParameter("Scalars", "S", "One scalar per mesh vertex.", GH_ParamAccess.list);
        pManager.AddNumberParameter("IsoValues", "IV", "Isovalues to extract.", GH_ParamAccess.list);
        pManager.AddNumberParameter("MergeTol", "Mt", "Endpoint merge when chaining segments.", GH_ParamAccess.item, 0.001);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddCurveParameter("Isolines", "IL", "Isolines per isovalue (tree branch = isovalue index).", GH_ParamAccess.tree);
        pManager.AddIntegerParameter("SegCount", "SC", "Total linear segments per isovalue (sum of polyline spans; same order as IsoValues).", GH_ParamAccess.list);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
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

        var scalars = new List<double>();
        DA.GetDataList("Scalars", scalars);
        int vc = mesh.Vertices.Count;
        if (scalars.Count < vc)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"Scalars count ({scalars.Count}) must be at least mesh vertex count ({vc}).");
            return;
        }

        var s = new double[vc];
        for (int i = 0; i < vc; i++)
            s[i] = scalars[i];

        var isos = new List<double>();
        DA.GetDataList("IsoValues", isos);
        if (isos.Count == 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Provide at least one isovalue.");
            DA.SetDataTree(0, new GH_Structure<GH_Curve>());
            DA.SetDataList(1, new List<GH_Integer>());
            return;
        }

        double mergeTol = 0.001;
        DA.GetData("MergeTol", ref mergeTol);
        if (mergeTol <= 0)
            mergeTol = 1e-6;

        var tree = new GH_Structure<GH_Curve>();
        var segCounts = new List<GH_Integer>();

        for (int k = 0; k < isos.Count; k++)
        {
            double iso = isos[k];
            List<Curve> curves = MeshIsolineHelper.ExtractIsolinesForValue(mesh, s, iso, mergeTol);
            var path = new GH_Path(k);
            int segTotal = 0;
            foreach (Curve c in curves)
            {
                tree.Append(new GH_Curve(c), path);
                if (c.TryGetPolyline(out Polyline pl))
                    segTotal += Math.Max(0, pl.Count - 1);
                else
                    segTotal += 1;
            }

            segCounts.Add(new GH_Integer(segTotal));
        }

        DA.SetDataTree(0, tree);
        DA.SetDataList(1, segCounts);
    }

    protected override Bitmap Icon => null!;

    public override Guid ComponentGuid => new("f98c9660-fe08-47a9-a2f8-c036dc63fd03");
}
