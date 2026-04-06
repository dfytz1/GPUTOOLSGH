using System.Drawing;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using GHGPUPlugin.Algorithms;
using GHGPUPlugin.NativeInterop;
using Rhino.Geometry;

namespace GHGPUPlugin.Components.DataRelationships;

/// <summary>Shortest path along mesh topology (Dijkstra); edge weights are 3D edge lengths, optionally filled on GPU.</summary>
public class GH_ShortestRouteGPU : GH_Component
{
    public GH_ShortestRouteGPU()
        : base(
            "Shortest Route Mesh GPU",
            "RouteMeshGPU",
            "Shortest path on mesh topology (Euclidean edge weights). Start/end points are snapped to the nearest mesh corner (topology vertex) via closest face. Dijkstra on CPU; optional Metal to build weighted edge list.",
            "GPUTools",
            "Graph")
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddMeshParameter("Mesh", "M", "Mesh whose topology edges define the graph.", GH_ParamAccess.item);
        pManager.AddPointParameter("Start", "S", "Start point (resolved to nearest topology vertex on closest face).", GH_ParamAccess.item);
        pManager.AddPointParameter("End", "E", "End point (resolved to nearest topology vertex on closest face).", GH_ParamAccess.item);
        pManager.AddBooleanParameter("UseGPU", "UseGPU", "Use Metal to compute edge lengths when available.", GH_ParamAccess.item, true);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddCurveParameter("Polyline", "Pl", "Polyline along the shortest route.", GH_ParamAccess.item);
        pManager.AddNumberParameter("Length", "L", "Total path length.", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        NativeLoader.EnsureLoaded();

        Mesh? mesh = null;
        if (!DA.GetData("Mesh", ref mesh) || mesh == null)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "No mesh provided.");
            return;
        }

        if (!mesh.IsValid)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Mesh is not valid.");
            return;
        }

        Point3d startPt = Point3d.Unset;
        if (!DA.GetData("Start", ref startPt))
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Start point is required.");
            return;
        }

        Point3d endPt = Point3d.Unset;
        if (!DA.GetData("End", ref endPt))
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "End point is required.");
            return;
        }

        if (!TryClosestTopologyVertex(mesh, startPt, out int start, out string? snapErr0))
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, snapErr0 ?? "Could not resolve start point on mesh.");
            return;
        }

        if (!TryClosestTopologyVertex(mesh, endPt, out int end, out string? snapErr1))
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, snapErr1 ?? "Could not resolve end point on mesh.");
            return;
        }

        bool useGpu = true;
        DA.GetData("UseGPU", ref useGpu);

        if (useGpu && !MetalGuard.EnsureReady(this))
            return;

        if (!MeshShortestPath.TryDijkstraTopology(mesh, start, end, useGpu, out List<Point3d>? path, out double length, out string? err)
            || path == null)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, err ?? "Shortest path failed.");
            return;
        }

        Polyline pl = path.Count == 1 ? new Polyline(new[] { path[0], path[0] }) : new Polyline(path);
        DA.SetData(0, pl);
        DA.SetData(1, new GH_Number(length));
    }

    /// <summary>Maps a 3D point to the topology vertex index of the mesh corner closest on the closest face.</summary>
    private static bool TryClosestTopologyVertex(Mesh mesh, Point3d p, out int topologyVertexIndex, out string? error)
    {
        topologyVertexIndex = -1;
        error = null;
        var mp = mesh.ClosestMeshPoint(p, double.MaxValue);
        if (mp.FaceIndex < 0 || mp.FaceIndex >= mesh.Faces.Count)
        {
            error = "Could not project point onto the mesh.";
            return false;
        }

        MeshFace f = mesh.Faces[mp.FaceIndex];
        int bestMv = f.A;
        double bestD = p.DistanceToSquared(mesh.Vertices[f.A]);

        void Consider(int meshVertexIndex)
        {
            double d = p.DistanceToSquared(mesh.Vertices[meshVertexIndex]);
            if (d < bestD)
            {
                bestD = d;
                bestMv = meshVertexIndex;
            }
        }

        Consider(f.B);
        Consider(f.C);
        if (f.IsQuad)
            Consider(f.D);

        topologyVertexIndex = mesh.TopologyVertices.TopologyVertexIndex(bestMv);
        return true;
    }

    protected override Bitmap Icon => null!;

    public override Guid ComponentGuid => new("ad1469e9-f5a6-4805-bb28-ec71028c54f7");
}
