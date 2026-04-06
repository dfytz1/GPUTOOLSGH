using System.Drawing;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using GHGPUPlugin.NativeInterop;
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

        var projected = new double[points.Count * 2];
        double minU = double.MaxValue, maxU = double.MinValue;
        double minV = double.MaxValue, maxV = double.MinValue;
        for (int i = 0; i < points.Count; i++)
        {
            plane.ClosestParameter(points[i], out double u, out double v);
            projected[i * 2] = u;
            projected[i * 2 + 1] = v;
            if (u < minU)
                minU = u;
            if (u > maxU)
                maxU = u;
            if (v < minV)
                minV = v;
            if (v > maxV)
                maxV = v;
        }

        double rangeU = maxU - minU;
        double rangeV = maxV - minV;
        double range = Math.Max(rangeU, rangeV);
        if (range < 1e-10)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "All points are coincident or collinear in the plane.");
            return;
        }

        var uv = new float[points.Count];
        var vv = new float[points.Count];
        for (int i = 0; i < points.Count; i++)
        {
            uv[i] = (float)(0.05 + 0.9 * (projected[i * 2] - minU) / range);
            vv[i] = (float)(0.05 + 0.9 * (projected[i * 2 + 1] - minV) / range);
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
