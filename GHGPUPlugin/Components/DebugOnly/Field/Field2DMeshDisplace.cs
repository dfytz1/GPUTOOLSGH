using System;
using Rhino.Geometry;

namespace GHGPUPlugin.Components.Field;

internal static class Field2DMeshDisplace
{
    internal static Mesh Build(
        Mesh input,
        float[,] field,
        int nx,
        int ny,
        Plane pl,
        double sx,
        double sy,
        double amplitude,
        bool normalize,
        bool useMeshNormals)
    {
        var m = input.DuplicateMesh();
        m.FaceNormals.ComputeFaceNormals();
        m.Normals.ComputeNormals();

        float min = float.MaxValue, max = float.MinValue;
        if (normalize)
        {
            for (int ix = 0; ix < nx; ix++)
            {
                for (int iy = 0; iy < ny; iy++)
                {
                    float v = field[ix, iy];
                    if (v < min) min = v;
                    if (v > max) max = v;
                }
            }
        }

        float denom = normalize && max > min ? max - min : 1f;
        Vector3d planeN = pl.ZAxis;
        planeN.Unitize();

        for (int i = 0; i < m.Vertices.Count; i++)
        {
            Point3d p = m.Vertices[i];
            float s = Field2DPlaneSampling.SampleAtWorld(field, nx, ny, pl, sx, sy, p);
            float t = normalize ? (s - min) / denom : s;
            t = Math.Clamp(t, 0f, 1f);
            Vector3d n = useMeshNormals ? m.Normals[i] : planeN;
            if (!n.Unitize())
                n = planeN;
            p += n * (t * amplitude);
            m.Vertices.SetVertex(i, p);
        }

        m.Normals.ComputeNormals();
        m.Compact();
        return m;
    }
}
