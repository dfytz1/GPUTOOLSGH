using System.Drawing;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using GHGPUPlugin.Algorithms;
using Rhino.Geometry;

namespace GHGPUPlugin.Components.DataRelationships;

/// <summary>Shortest path along a curve network (one edge per curve); CPU-only.</summary>
public class GH_ShortestRouteCurves : GH_Component
{
    public GH_ShortestRouteCurves()
        : base(
            "Shortest Route Curves",
            "RouteCrv",
            "Each curve is one edge (StartPoint→EndPoint) weighted by arc length. No sampling. Endpoints closer than Merge tol share a node; start/end snap to the nearest node within Snap tol.",
            "GPUTools",
            "Curve")
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddCurveParameter("Curves", "C", "Network curves: each contributes one undirected edge between its endpoints.", GH_ParamAccess.list);
        pManager.AddPointParameter("Start", "S", "Start point (snapped to nearest network vertex).", GH_ParamAccess.item);
        pManager.AddPointParameter("End", "E", "End point (snapped to nearest network vertex).", GH_ParamAccess.item);
        pManager.AddNumberParameter("Merge tol", "Mt", "Distance within which curve endpoints are treated as the same vertex.", GH_ParamAccess.item, 1e-3);
        pManager.AddNumberParameter("Snap tol", "St", "Maximum distance from start/end to the network for snapping.", GH_ParamAccess.item, 0.1);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddCurveParameter("Polyline", "Pl", "Polyline along the shortest route (through merged nodes).", GH_ParamAccess.item);
        pManager.AddNumberParameter("Length", "L", "Total path length.", GH_ParamAccess.item);
        pManager.AddIntegerParameter("NodeCount", "N", "Number of intermediate nodes on the path (path vertex count minus two), minimum zero.", GH_ParamAccess.item);
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

        double mergeTol = 1e-3;
        DA.GetData("Merge tol", ref mergeTol);

        double snapTol = 0.1;
        DA.GetData("Snap tol", ref snapTol);

        if (!CurveNetworkShortestPath.TryFindPath(curves, startPt, endPt, mergeTol, snapTol, out List<Point3d>? path, out double length, out string? err)
            || path == null)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, err ?? "Shortest path failed.");
            return;
        }

        Polyline pl = path.Count == 1 ? new Polyline(new[] { path[0], path[0] }) : new Polyline(path);
        int nodeCount = Math.Max(0, path.Count - 2);
        DA.SetData(0, pl);
        DA.SetData(1, new GH_Number(length));
        DA.SetData(2, nodeCount);
    }

    protected override Bitmap Icon => null!;

    public override Guid ComponentGuid => new("c4e2b8a1-2f1d-4e6a-9c0b-5d3e7f1a2b4c");
}
