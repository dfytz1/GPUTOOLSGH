using System.Drawing;
using GHGPUPlugin.Algorithms;
using GHGPUPlugin.NativeInterop;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;

namespace GHGPUPlugin.Components.DataRelationships;

/// <summary>GPU Delaunay-related edge extraction via Jump Flooding Voronoi on Metal (approximate; resolution-dependent).</summary>
public class GH_JFADelaunay2D : GH_Component
{
    public GH_JFADelaunay2D()
        : base(
            "JFA Delaunay 2D GPU",
            "JFADelGPU",
            "GPU Jump Flooding Voronoi on a grid, then dual edges (approximate Delaunay edges). Projects points to a plane; requires Metal.",
            "GPUTools",
            "Graph")
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager pm)
    {
        pm.AddPointParameter("Points", "P", "Input points (projected to the plane).", GH_ParamAccess.list);
        pm.AddPlaneParameter("Plane", "Pl", "Projection plane.", GH_ParamAccess.item, Plane.WorldXY);
        pm.AddIntegerParameter("GridResolution", "GridRes", "Minimum JFA grid size; snapped up to next power of two (e.g. 512).", GH_ParamAccess.item, 512);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pm)
    {
        pm.AddLineParameter("DelaunayEdges", "E", "Extracted dual edges between sites.", GH_ParamAccess.list);
        pm.AddTextParameter("Info", "I", "Diagnostic text.", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        NativeLoader.EnsureLoaded();
        if (!MetalGuard.EnsureReady(this))
            return;

        var points = new List<Point3d>();
        if (!DA.GetDataList("Points", points) || points.Count < 3)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Need at least three points.");
            return;
        }

        var plane = Plane.WorldXY;
        DA.GetData("Plane", ref plane);

        int gridRes = 512;
        DA.GetData("GridResolution", ref gridRes);

        var uv2 = new Vector2d[points.Count];
        for (int i = 0; i < points.Count; i++)
        {
            plane.ClosestParameter(points[i], out double u, out double v);
            uv2[i] = new Vector2d(u, v);
        }

        if (!JfaDelaunay2DPlanar.TryJfaNormalizedCoords(uv2, out float[] uv, out float[] vv))
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "All points are coincident or collinear in the plane.");
            return;
        }

        int maxEdges = points.Count * 12;
        var outA = new int[maxEdges];
        var outB = new int[maxEdges];

        if (!MetalSharedContext.TryGetContext(out IntPtr ctx))
            return;

        int code = MetalBridge.JfaDelaunay2D(ctx, uv, vv, points.Count, outA, outB, out int edgeCount, maxEdges, gridRes);
        if (code != 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"mb_jfa_delaunay_2d returned error {code}.");
            return;
        }

        var edges = new List<GH_Line>(edgeCount);
        for (int i = 0; i < edgeCount; i++)
            edges.Add(new GH_Line(new Line(points[outA[i]], points[outB[i]])));

        DA.SetDataList(0, edges);
        int snapped = 64;
        while (snapped < gridRes)
            snapped *= 2;
        DA.SetData(1, $"JFA Delaunay: {points.Count} pts → {edgeCount} edges (grid {snapped}×{snapped})");
    }

    protected override Bitmap Icon => null!;

    public override Guid ComponentGuid => new("222f4947-3b6f-4a92-9977-5bdbeb1be4ef");
}
