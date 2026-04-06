using System.Drawing;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using GHGPUPlugin.Algorithms;
using Rhino.Geometry;

namespace GHGPUPlugin.Components.DataRelationships;

public class GH_CurveNetworkGraph : GH_Component
{
    public GH_CurveNetworkGraph()
        : base(
            "Curve Network Graph GPU",
            "CrvNetGraphGPU",
            "Builds an undirected curve network (one edge per curve, merged endpoints). No pathfinding.",
            "GPUTools",
            "Graph")
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddCurveParameter("Curves", "C", "Network curves.", GH_ParamAccess.list);
        pManager.AddNumberParameter("MergeTol", "Mt", "Endpoint merge distance.", GH_ParamAccess.item, 0.01);
        pManager.AddBooleanParameter("FastMode", "Fm", "Chord length instead of arc length for stored edge lengths.", GH_ParamAccess.item, false);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddPointParameter("GraphNodes", "GN", "Merged network nodes.", GH_ParamAccess.list);
        pManager.AddCurveParameter("GraphEdges", "GE", "One input curve per edge.", GH_ParamAccess.list);
        pManager.AddNumberParameter("EdgeLengths", "EL", "Weight per edge (same order as GraphEdges).", GH_ParamAccess.list);
        pManager.AddIntegerParameter("EdgeEndA", "EA", "Start node index per edge.", GH_ParamAccess.list);
        pManager.AddIntegerParameter("EdgeEndB", "EB", "End node index per edge.", GH_ParamAccess.list);
        pManager.AddTextParameter("GraphInfo", "GI", "Summary string.", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        var curves = new List<Curve>();
        if (!DA.GetDataList("Curves", curves))
            curves = new List<Curve>();
        int curvesIn = curves.Count;

        double mergeTol = 0.01;
        DA.GetData("MergeTol", ref mergeTol);

        bool fast = false;
        DA.GetData("FastMode", ref fast);

        CurveGraph g = CurveNetworkShortestPath.BuildGraph(curves, mergeTol, fast);

        var gn = new List<GH_Point>(g.Vertices.Count);
        foreach (Point3d p in g.Vertices)
            gn.Add(new GH_Point(p));

        var ge = new List<GH_Curve>(g.EdgeCurves.Count);
        foreach (Curve c in g.EdgeCurves)
            ge.Add(new GH_Curve(c));

        var el = new List<GH_Number>(g.EdgeLengths.Count);
        foreach (double w in g.EdgeLengths)
            el.Add(new GH_Number(w));

        var ea = new List<GH_Integer>(g.EdgePairs.Count);
        var eb = new List<GH_Integer>(g.EdgePairs.Count);
        foreach ((int a, int b) in g.EdgePairs)
        {
            ea.Add(new GH_Integer(a));
            eb.Add(new GH_Integer(b));
        }

        string info = $"Nodes: {g.Vertices.Count}  Edges: {g.EdgePairs.Count}  Curves in: {curvesIn}";
        DA.SetDataList(0, gn);
        DA.SetDataList(1, ge);
        DA.SetDataList(2, el);
        DA.SetDataList(3, ea);
        DA.SetDataList(4, eb);
        DA.SetData(5, info);

        if (curvesIn == 0)
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "No curves provided.");
        else if (mergeTol <= 0)
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "MergeTol must be positive.");
    }

    protected override Bitmap Icon => null!;

    public override Guid ComponentGuid => new("b711343a-1d35-49ad-a8cd-96e6f694eccb");
}
