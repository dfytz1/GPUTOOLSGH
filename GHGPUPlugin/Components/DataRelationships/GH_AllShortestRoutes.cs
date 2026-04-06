using System.Drawing;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using GHGPUPlugin.Algorithms;
using Rhino.Geometry;

namespace GHGPUPlugin.Components.DataRelationships;

public class GH_AllShortestRoutes : GH_Component
{
    public GH_AllShortestRoutes()
        : base(
            "All Shortest Routes GPU",
            "AllRoutesGPU",
            "Single-source Dijkstra to every node from a snapped source point. Outputs distances and a shortest-path tree as line segments.",
            "GPUTools",
            "Routing")
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddCurveParameter("Curves", "C", "Network curves.", GH_ParamAccess.list);
        pManager.AddPointParameter("Source", "S", "Source point (snapped to nearest node).", GH_ParamAccess.item);
        pManager.AddNumberParameter("MergeTol", "Mt", "Endpoint merge distance.", GH_ParamAccess.item, 0.01);
        pManager.AddNumberParameter("SnapTol", "St", "Source snap distance.", GH_ParamAccess.item, 1.0);
        pManager.AddBooleanParameter("FastMode", "Fm", "Chord length for edge weights.", GH_ParamAccess.item, false);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddNumberParameter("Distances", "D", "Distance from source to each node (unreachable → -1).", GH_ParamAccess.list);
        pManager.AddPointParameter("GraphNodes", "GN", "Merged nodes.", GH_ParamAccess.list);
        pManager.AddCurveParameter("SpanningTree", "ST", "Line from each reached node to its predecessor.", GH_ParamAccess.list);
        pManager.AddTextParameter("GraphInfo", "GI", "Summary.", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        var curves = new List<Curve>();
        if (!DA.GetDataList("Curves", curves))
            curves = new List<Curve>();

        Point3d src = Point3d.Unset;
        bool haveSrc = DA.GetData("Source", ref src) && src.IsValid;

        double mergeTol = 0.01;
        DA.GetData("MergeTol", ref mergeTol);

        double snapTol = 1.0;
        DA.GetData("SnapTol", ref snapTol);

        bool fast = false;
        DA.GetData("FastMode", ref fast);

        CurveGraph g = CurveNetworkShortestPath.BuildGraph(curves, mergeTol, fast);

        var gn = new List<GH_Point>(g.Vertices.Count);
        foreach (Point3d p in g.Vertices)
            gn.Add(new GH_Point(p));
        DA.SetDataList(1, gn);

        if (curves.Count == 0 || !haveSrc || mergeTol <= 0 || snapTol <= 0 || g.Vertices.Count == 0)
        {
            DA.SetDataList(0, new List<GH_Number>());
            DA.SetDataList(2, new List<GH_Curve>());
            DA.SetData(3, "Nodes: 0  Edges: 0  Source node: -1");
            if (curves.Count == 0)
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Provide at least one curve.");
            else if (!haveSrc)
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Source point is required.");
            else if (mergeTol <= 0)
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "MergeTol must be positive.");
            else if (snapTol <= 0)
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "SnapTol must be positive.");
            return;
        }

        if (!CurveNetworkShortestPath.TrySingleSourceAll(g, src, snapTol, out int sourceIdx, out double[]? dist, out int[]? prev, out string? err)
            || dist == null || prev == null)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, err ?? "Dijkstra failed.");
            DA.SetDataList(0, new List<GH_Number>());
            DA.SetDataList(2, new List<GH_Curve>());
            DA.SetData(3, $"Nodes: {g.Vertices.Count}  Edges: {g.EdgePairs.Count}  Source node: -1");
            return;
        }

        var dOut = new List<GH_Number>(dist.Length);
        for (int i = 0; i < dist.Length; i++)
        {
            double v = dist[i];
            dOut.Add(new GH_Number(double.IsPositiveInfinity(v) ? -1 : v));
        }

        DA.SetDataList(0, dOut);

        var st = new List<GH_Curve>();
        for (int i = 0; i < prev.Length; i++)
        {
            int p = prev[i];
            if (p < 0 || i == sourceIdx)
                continue;
            var ln = new Line(g.Vertices[i], g.Vertices[p]);
            st.Add(new GH_Curve(new LineCurve(ln.From, ln.To)));
        }

        DA.SetDataList(2, st);
        DA.SetData(3, $"Nodes: {g.Vertices.Count}  Edges: {g.EdgePairs.Count}  Source node: {sourceIdx}");
    }

    protected override Bitmap Icon => null!;

    public override Guid ComponentGuid => new("b2c3d4e5-f6a7-8901-bcde-f12345678901");
}
