using System.Drawing;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using GHGPUPlugin.Algorithms;
using GHGPUPlugin.NativeInterop;
using Rhino.Geometry;

namespace GHGPUPlugin.Components.DataRelationships;

/// <summary>2D Delaunay in a plane (Bowyer–Watson); optional Metal for circumcircle tests per insert.</summary>
public class GH_DelaunayMeshGPU : GH_Component
{
    public GH_DelaunayMeshGPU()
        : base(
            "Delaunay Mesh GPU",
            "DelaunayGPU",
            "Planar Delaunay triangulation of 3D points projected to a plane. Optional GPU circumcircle marking (often faster than CPU for many circumcircle tests). Mesh or unique edge lines when OutMesh is false.",
            "GPUTools",
            "Mesh")
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddPointParameter("Points", "P", "Points to triangulate (projected to Plane).", GH_ParamAccess.list);
        pManager.AddPlaneParameter("Plane", "Pl", "Plane for 2D parameterization.", GH_ParamAccess.item, Plane.WorldXY);
        pManager.AddBooleanParameter("OutputMesh", "OutMesh", "If true, output a mesh; if false, output unique triangle edges as lines.", GH_ParamAccess.item, true);
        pManager.AddBooleanParameter("UseGPU", "UseGPU", "Use Metal for circumcircle tests when available.", GH_ParamAccess.item, true);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddMeshParameter("Mesh", "M", "Delaunay mesh (null if OutputMesh is false).", GH_ParamAccess.item);
        pManager.AddIntegerParameter("TriangleCorners", "Tri", "Flattened triangle corner indices (A,B,C per face) into Points list.", GH_ParamAccess.list);
        pManager.AddCurveParameter("Lines", "Ln", "Unique triangle edges as line segments when OutputMesh is false; empty otherwise.", GH_ParamAccess.list);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        NativeLoader.EnsureLoaded();

        var pts3d = new List<Point3d>();
        if (!DA.GetDataList("Points", pts3d) || pts3d.Count < 3)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Need at least three points.");
            return;
        }

        Plane plane = Plane.WorldXY;
        DA.GetData("Plane", ref plane);

        bool outputMesh = true;
        DA.GetData("OutputMesh", ref outputMesh);

        bool useGpu = true;
        DA.GetData("UseGPU", ref useGpu);

        var pts2 = new List<Point2d>(pts3d.Count);
        foreach (Point3d p in pts3d)
        {
            plane.ClosestParameter(p, out double u, out double v);
            pts2.Add(new Point2d(u, v));
        }

        if (!Delaunay2D.TryTriangulate(pts2, useGpu, out List<(int A, int B, int C)>? tris, out string? err) || tris == null)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, err ?? "Delaunay failed.");
            return;
        }

        var flat = new List<GH_Integer>(tris.Count * 3);
        foreach (var t in tris)
        {
            flat.Add(new GH_Integer(t.A));
            flat.Add(new GH_Integer(t.B));
            flat.Add(new GH_Integer(t.C));
        }

        DA.SetDataList(1, flat);

        if (!outputMesh)
        {
            DA.SetData(0, null);
            var seen = new HashSet<(int U, int V)>();
            var lines = new List<Curve>();
            void AddEdge(int a, int b)
            {
                if (a == b)
                    return;
                int u = Math.Min(a, b);
                int v = Math.Max(a, b);
                if (!seen.Add((u, v)))
                    return;
                lines.Add(new LineCurve(new Line(pts3d[u], pts3d[v])));
            }

            foreach (var t in tris)
            {
                AddEdge(t.A, t.B);
                AddEdge(t.B, t.C);
                AddEdge(t.C, t.A);
            }

            DA.SetDataList(2, lines);
            return;
        }

        DA.SetDataList(2, new List<Curve>());

        var mesh = new Mesh();
        foreach (Point3d p in pts3d)
            mesh.Vertices.Add(p);
        foreach (var t in tris)
            mesh.Faces.AddFace(t.A, t.B, t.C);
        mesh.Normals.ComputeNormals();
        DA.SetData(0, mesh);
    }

    protected override Bitmap Icon => null!;

    public override Guid ComponentGuid => new("62ffb586-9401-42ab-b81d-fed036827972");
}
