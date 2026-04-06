using System.Drawing;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using GHGPUPlugin.Algorithms;
using Rhino.Geometry;

namespace GHGPUPlugin.Components.DataRelationships;

/// <summary>Shortest path along a network of curves (Dijkstra on sampled segments); CPU-only.</summary>
public class GH_ShortestRouteCurves : GH_Component
{
    public GH_ShortestRouteCurves()
        : base(
            "Shortest Route Curves",
            "RouteCrv",
            "Shortest path along polylines/lines/curves treated as an undirected network. Curves are sampled to segments no longer than Max edge; nearby endpoints are merged within Merge tol; start/end snap to the nearest vertex within Snap tol.",
            "GPUTools",
            "Routing")
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddCurveParameter("Curves", "C", "Network curves (polylines, lines, or general curves — sampled).", GH_ParamAccess.list);
        pManager.AddPointParameter("Start", "S", "Start point (snapped to nearest network vertex).", GH_ParamAccess.item);
        pManager.AddPointParameter("End", "E", "End point (snapped to nearest network vertex).", GH_ParamAccess.item);
        pManager.AddNumberParameter("Merge tol", "Mt", "Distance within which sample points are treated as the same vertex.", GH_ParamAccess.item, 1e-6);
        pManager.AddNumberParameter("Max edge", "Me", "Maximum segment length when sampling non-polyline curves.", GH_ParamAccess.item, 1.0);
        pManager.AddNumberParameter("Snap tol", "St", "Maximum distance from start/end to the network for snapping.", GH_ParamAccess.item, 0.01);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddCurveParameter("Polyline", "Pl", "Polyline along the shortest route.", GH_ParamAccess.item);
        pManager.AddNumberParameter("Length", "L", "Total path length.", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        var curves = new List<Curve>();
        if (!DA.GetDataList("Curves", curves) || curves.Count == 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Provide at least one curve.");
            return;
        }

        Point3d startPt = Point3d.Unset;
        if (!DA.GetData("Start", ref startPt) || !startPt.IsValid)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Start point is required.");
            return;
        }

        Point3d endPt = Point3d.Unset;
        if (!DA.GetData("End", ref endPt) || !endPt.IsValid)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "End point is required.");
            return;
        }

        double mergeTol = 1e-6;
        DA.GetData("Merge tol", ref mergeTol);

        double maxEdge = 1.0;
        DA.GetData("Max edge", ref maxEdge);

        double snapTol = 0.01;
        DA.GetData("Snap tol", ref snapTol);

        if (!CurveNetworkShortestPath.TryFindPath(curves, startPt, endPt, mergeTol, maxEdge, snapTol, out List<Point3d>? path, out double length, out string? err)
            || path == null)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, err ?? "Shortest path failed.");
            return;
        }

        Polyline pl = path.Count == 1 ? new Polyline(new[] { path[0], path[0] }) : new Polyline(path);
        DA.SetData(0, pl);
        DA.SetData(1, new GH_Number(length));
    }

    protected override Bitmap Icon => null!;

    public override Guid ComponentGuid => new("c4e2b8a1-2f1d-4e6a-9c0b-5d3e7f1a2b4c");
}
