using System.Drawing;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using GHGPUPlugin.Algorithms;
using Rhino.Geometry;

namespace GHGPUPlugin.Components.DataRelationships;

/// <summary>Shortest path on a curve network (one edge per input curve).</summary>
public class GH_ShortestRouteCurves : GH_Component
{
    public GH_ShortestRouteCurves()
        : base(
            "Shortest Route Curve GPU",
            "RouteCrvGPU",
            "Each curve is one edge between its endpoints; endpoints merge within MergeTol. Weights use arc length unless Fast mode uses chord length. Start/end snap to the nearest graph node within SnapTol (linear scan).",
            "GPUTools",
            "Curve")
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddCurveParameter("Curves", "C", "Network curves: each contributes one undirected edge between its endpoints.", GH_ParamAccess.list);
        pManager.AddPointParameter("Start", "S", "Start point (snapped to nearest network node).", GH_ParamAccess.item);
        pManager.AddPointParameter("End", "E", "End point (snapped to nearest network node).", GH_ParamAccess.item);
        pManager.AddNumberParameter("MergeTol", "Mt", "Distance within which curve endpoints are treated as the same vertex.", GH_ParamAccess.item, 0.01);
        pManager.AddNumberParameter("SnapTol", "St", "Maximum distance from start/end to the network for snapping.", GH_ParamAccess.item, 1.0);
        pManager.AddBooleanParameter("FastMode", "Fm", "Use chord length instead of arc length for edge weights.", GH_ParamAccess.item, false);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddCurveParameter("RoutePath", "P", "Polyline along the shortest route.", GH_ParamAccess.item);
        pManager.AddNumberParameter("RouteLength", "L", "Total weighted path length.", GH_ParamAccess.item);
        pManager.AddIntegerParameter("NodeCount", "N", "Intermediate nodes on the path (max(0, path vertex count − 2)).", GH_ParamAccess.item);
        pManager.AddPointParameter("GraphNodes", "GN", "All merged network nodes.", GH_ParamAccess.list);
        pManager.AddCurveParameter("GraphEdges", "GE", "All edge curves (one per input edge kept in the graph).", GH_ParamAccess.list);
        pManager.AddTextParameter("GraphInfo", "GI", "Summary: node count, edge count, input curve count.", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        var curves = new List<Curve>();
        if (!DA.GetDataList("Curves", curves))
            curves = new List<Curve>();

        int curvesIn = curves.Count;

        Point3d startPt = Point3d.Unset;
        bool haveStart = DA.GetData("Start", ref startPt) && startPt.IsValid;

        Point3d endPt = Point3d.Unset;
        bool haveEnd = DA.GetData("End", ref endPt) && endPt.IsValid;

        double mergeTol = 0.01;
        DA.GetData("MergeTol", ref mergeTol);

        double snapTol = 1.0;
        DA.GetData("SnapTol", ref snapTol);

        bool fastMode = false;
        DA.GetData("FastMode", ref fastMode);

        CurveGraph graph = CurveNetworkShortestPath.BuildGraph(curves, mergeTol, fastMode);

        var ghNodes = new List<GH_Point>(graph.Vertices.Count);
        foreach (Point3d p in graph.Vertices)
            ghNodes.Add(new GH_Point(p));

        var ghEdges = new List<GH_Curve>(graph.EdgeCurves.Count);
        foreach (Curve c in graph.EdgeCurves)
            ghEdges.Add(new GH_Curve(c));

        string graphInfo = $"Nodes: {graph.Vertices.Count}  Edges: {graph.EdgePairs.Count}  Curves in: {curvesIn}";
        DA.SetDataList(3, ghNodes);
        DA.SetDataList(4, ghEdges);
        DA.SetData(5, graphInfo);

        if (curvesIn == 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Provide at least one curve.");
            ClearRouteOutputs(DA);
            return;
        }

        if (mergeTol <= 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "MergeTol must be positive.");
            ClearRouteOutputs(DA);
            return;
        }

        if (!haveStart)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Start point is required.");
            ClearRouteOutputs(DA);
            return;
        }

        if (!haveEnd)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "End point is required.");
            ClearRouteOutputs(DA);
            return;
        }

        if (graph.Vertices.Count == 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "No graph nodes (check curves and MergeTol).");
            ClearRouteOutputs(DA);
            return;
        }

        if (!CurveNetworkShortestPath.TryFindPath(graph, startPt, endPt, snapTol, out List<int>? pathIdx, out double length, out string? err)
            || pathIdx == null)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, err ?? "Shortest path failed.");
            ClearRouteOutputs(DA);
            return;
        }

        var pathPts = new List<Point3d>(pathIdx.Count);
        foreach (int vi in pathIdx)
            pathPts.Add(graph.Vertices[vi]);

        Polyline pl = pathPts.Count == 1
            ? new Polyline(new[] { pathPts[0], pathPts[0] })
            : new Polyline(pathPts);

        int nodeCount = Math.Max(0, pathIdx.Count - 2);
        DA.SetData(0, pl);
        DA.SetData(1, new GH_Number(length));
        DA.SetData(2, nodeCount);
    }

    private static void ClearRouteOutputs(IGH_DataAccess DA)
    {
        DA.SetData(0, null);
        DA.SetData(1, null);
        DA.SetData(2, 0);
    }

    protected override Bitmap Icon => null!;

    public override Guid ComponentGuid => new("c4e2b8a1-2f1d-4e6a-9c0b-5d3e7f1a2b4c");
}
