using System.Numerics;
using Rhino.Geometry;
using SpectralPacking.Core.Geometry;

namespace SpectralPacking.GH.Components.DebugOnly;

public static class PackedMeshBuilder
{
    public static Mesh Build(Mesh original, Matrix4x4 R, Vector3 t)
    {
        var soup = RhinoMeshSoup.FromRhinoMesh(original);
        var r = soup.RotatedAboutCentroid(
            R.M11, R.M12, R.M13,
            R.M21, R.M22, R.M23,
            R.M31, R.M32, R.M33);

        var m = new Mesh();
        for (int i = 0; i < r.VertexCount; i++)
        {
            m.Vertices.Add(
                r.Vx[i] + t.X,
                r.Vy[i] + t.Y,
                r.Vz[i] + t.Z);
        }

        int fc = r.TriangleCount;
        for (int i = 0; i < fc; i++)
        {
            int a = r.TriangleIndices[i * 3];
            int b = r.TriangleIndices[i * 3 + 1];
            int c = r.TriangleIndices[i * 3 + 2];
            m.Faces.AddFace(a, b, c);
        }

        m.Normals.ComputeNormals();
        m.Compact();
        return m;
    }

    public static Rhino.Geometry.Plane ToPlacementPlane(Mesh original, Matrix4x4 R, Vector3 t)
    {
        var soup = RhinoMeshSoup.FromRhinoMesh(original);
        var (cx, cy, cz) = soup.Centroid();
        var origin = new Point3d(cx + t.X, cy + t.Y, cz + t.Z);
        var xAxis = new Vector3d(R.M11, R.M21, R.M31);
        var yAxis = new Vector3d(R.M12, R.M22, R.M32);
        var zAxis = new Vector3d(R.M13, R.M23, R.M33);
        xAxis.Unitize();
        yAxis.Unitize();
        zAxis.Unitize();
        return new Rhino.Geometry.Plane(origin, xAxis, yAxis);
    }
}
