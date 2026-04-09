using System.Drawing;
using GHGPUPlugin.Algorithms;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;

namespace GHGPUPlugin.Components.DataRelationships;

/// <summary>Planar constrained Delaunay triangulation (Triangle.NET, CPU): boundary and holes as hard edges.</summary>
public class GH_BoundaryMesh : GH_Component
{
    public GH_BoundaryMesh()
        : base(
            "Boundary Mesh",
            "BndMesh",
            "Meshes a planar region from a closed boundary curve and optional closed hole curves (floor-plan style). Edges are preserved (constrained Delaunay). Max area greater than zero enables Triangle.NET refinement (Steiner points). CPU only (Unofficial.Triangle.NET).",
            "GPUTools",
            "Mesh")
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager pm)
    {
        pm.AddCurveParameter("Boundary", "B", "Closed outer boundary (planar).", GH_ParamAccess.item);
        pm.AddCurveParameter("Holes", "H", "Closed hole curves (planar). Empty list for no holes.", GH_ParamAccess.list);
        pm.AddNumberParameter("MaxArea", "A", "Maximum triangle area for refinement; zero or negative skips refinement (CDT only).", GH_ParamAccess.item, 0.0);
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
        if (!DA.GetData(0, ref boundary) || boundary == null)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Boundary curve is missing.");
            return;
        }

        var holesRaw = new List<Curve>();
        DA.GetDataList(1, holesRaw);
        List<Curve> holes = holesRaw.Where(c => c != null && c.IsValid).ToList();

        double maxArea = 0.0;
        DA.GetData(2, ref maxArea);

        double maxEdge = 1.0;
        DA.GetData(3, ref maxEdge);

        var plane = Plane.WorldXY;
        DA.GetData(4, ref plane);

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

    public override Guid ComponentGuid => new("d4e8f2a1-7b4c-5d0e-9f8a-2b3c4d5e6f70");
}
