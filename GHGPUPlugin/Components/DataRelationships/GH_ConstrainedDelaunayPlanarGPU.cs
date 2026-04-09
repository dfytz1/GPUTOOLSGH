using System.Drawing;
using GHGPUPlugin.Algorithms;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;

namespace GHGPUPlugin.Components.DataRelationships;

/// <summary>Planar constrained Delaunay triangulation (Triangle.NET): closed boundary and hole loops become hard edges; optional max triangle area refinement.</summary>
public class GH_ConstrainedDelaunayPlanarGPU : GH_Component
{
    public GH_ConstrainedDelaunayPlanarGPU()
        : base(
            "Constrained Delaunay Planar GPU",
            "CDtPlanGPU",
            "Meshes a planar region from a closed boundary curve and optional closed hole curves. All polyline edges are preserved (constrained Delaunay). Set Max area greater than zero to subdivide with Triangle.NET quality refinement (Steiner points). Uses Unofficial.Triangle.NET.",
            "GPUTools",
            "Mesh")
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager pm)
    {
        pm.AddCurveParameter("Boundary", "B", "Closed outer boundary (planar).", GH_ParamAccess.item);
        pm.AddCurveParameter("Holes", "H", "Closed hole curves (planar). Empty list for no holes.", GH_ParamAccess.list);
        pm.AddNumberParameter("MaxArea", "A", "Maximum triangle area for refinement; ≤0 skips refinement (CDT only, no Steiner points beyond Triangle defaults).", GH_ParamAccess.item, 0.0);
        pm.AddNumberParameter("MaxEdge", "E", "Target max spacing along curves when sampling polylines (model units).", GH_ParamAccess.item, 1.0);
        pm.AddPlaneParameter("Plane", "Pl", "Plane for projection; curve geometry is projected to this plane before meshing.", GH_ParamAccess.item, Plane.WorldXY);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pm)
    {
        pm.AddMeshParameter("Mesh", "M", "Triangulated mesh in the plane.", GH_ParamAccess.item);
        pm.AddTextParameter("Info", "I", "Vertex and triangle counts.", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        Curve? boundary = null;
        if (!DA.GetData("Boundary", ref boundary) || boundary == null)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Boundary curve is missing.");
            return;
        }

        var holesRaw = new List<Curve>();
        DA.GetDataList("Holes", holesRaw);
        List<Curve> holes = holesRaw.Where(c => c != null && c.IsValid).ToList();

        double maxArea = 0.0;
        DA.GetData("MaxArea", ref maxArea);

        double maxEdge = 1.0;
        DA.GetData("MaxEdge", ref maxEdge);

        var plane = Plane.WorldXY;
        DA.GetData("Plane", ref plane);

        if (!boundary.IsClosed)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Boundary must be a closed curve.");
            return;
        }

        for (int i = 0; i < holes.Count; i++)
        {
            if (holes[i] != null && !holes[i].IsClosed)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, $"Hole curve at index {i} must be closed.");
                return;
            }
        }

        if (!PlanarCdtFromCurves.TryTriangulate(
                boundary,
                holes,
                plane,
                maxEdge,
                maxArea,
                out List<Point3d> verts,
                out List<int> tris,
                out string detail))
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, detail);
            return;
        }

        var mesh = new Mesh();
        for (int i = 0; i < verts.Count; i++)
            mesh.Vertices.Add(verts[i]);

        int nTri = tris.Count / 3;
        for (int t = 0; t < nTri; t++)
            mesh.Faces.AddFace(tris[t * 3], tris[t * 3 + 1], tris[t * 3 + 2]);

        mesh.Normals.ComputeNormals();
        mesh.Compact();

        DA.SetData(0, mesh);
        DA.SetData(1, $"{detail} | MaxArea={(maxArea > 0 ? maxArea.ToString() : "off")}");
    }

    protected override Bitmap Icon => null!;

    public override Guid ComponentGuid => new("c4d8e2f1-6a3b-4c9d-8e7f-0a1b2c3d4e5f");
}
