using Rhino.Geometry;
using SpectralPacking.Core.Geometry;

namespace SpectralPacking.GH.Components.DebugOnly;

public static class RhinoMeshSoup
{
    public static MeshTriangleSoup FromRhinoMesh(Mesh mesh)
    {
        var m = mesh.DuplicateMesh();
        m.Faces.ConvertQuadsToTriangles();
        int vc = m.Vertices.Count;
        var vx = new double[vc];
        var vy = new double[vc];
        var vz = new double[vc];
        for (int i = 0; i < vc; i++)
        {
            var p = m.Vertices[i];
            vx[i] = p.X;
            vy[i] = p.Y;
            vz[i] = p.Z;
        }

        int fc = m.Faces.Count;
        var tri = new int[fc * 3];
        for (int i = 0; i < fc; i++)
        {
            var f = m.Faces[i];
            tri[i * 3] = f.A;
            tri[i * 3 + 1] = f.B;
            tri[i * 3 + 2] = f.C;
        }

        return new MeshTriangleSoup(vx, vy, vz, tri);
    }

    public static Mesh TransformMesh(Mesh source, Transform xform)
    {
        var m = source.DuplicateMesh();
        m.Transform(xform);
        return m;
    }
}
