using System;
using System.Drawing;
using Rhino.Geometry;

namespace GHGPUPlugin.Components.Field;

internal static class Field2DMeshVertexPaint
{
    /// <summary>Colors quad grid mesh from field extrema (full grid).</summary>
    internal static void ApplyScalarField(Mesh mesh, float[,] field, int nx, int ny, Plane pl, double sx, double sy, bool normalizeColors)
    {
        float min = float.MaxValue, max = float.MinValue;
        if (normalizeColors)
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

        float denom = normalizeColors && max > min ? max - min : 1f;

        mesh.VertexColors.Clear();
        for (int i = 0; i < mesh.Vertices.Count; i++)
        {
            Point3d p = mesh.Vertices.Point3dAt(i);
            float s = Field2DPlaneSampling.SampleAtWorld(field, nx, ny, pl, sx, sy, p);
            float t = normalizeColors ? (max > min ? (s - min) / denom : 0.5f) : s;
            t = Math.Clamp(t, 0f, 1f);
            Color c = Field2DColormap.TurboColor(t);
            mesh.VertexColors.Add(c.R, c.G, c.B);
        }
    }

    /// <summary>Colors arbitrary mesh; when normalizing, uses min–max of <em>sampled</em> values at vertices (better contrast).</summary>
    internal static void ApplyScalarFieldVertexSampledRange(Mesh mesh, float[,] field, int nx, int ny, Plane pl, double sx, double sy, bool normalizeColors)
    {
        int vc = mesh.Vertices.Count;
        var sampled = new float[vc];
        for (int i = 0; i < vc; i++)
            sampled[i] = Field2DPlaneSampling.SampleAtWorld(field, nx, ny, pl, sx, sy, mesh.Vertices.Point3dAt(i));

        float min = float.MaxValue, max = float.MinValue;
        if (normalizeColors)
        {
            for (int i = 0; i < vc; i++)
            {
                float v = sampled[i];
                if (v < min) min = v;
                if (v > max) max = v;
            }
        }

        float denom = normalizeColors && max > min ? max - min : 1f;

        mesh.VertexColors.Clear();
        for (int i = 0; i < vc; i++)
        {
            float s = sampled[i];
            float t = normalizeColors ? (max > min ? (s - min) / denom : 0.5f) : s;
            t = Math.Clamp(t, 0f, 1f);
            Color c = Field2DColormap.TurboColor(t);
            mesh.VertexColors.Add(c.R, c.G, c.B);
        }
    }
}
