using System;
using System.Drawing;
using Rhino.Geometry;

namespace GHGPUPlugin.Components.Field;

internal static class Field2DPreviewMesh
{
    internal static Mesh BuildMesh(float[,] field, Plane plane, double sizeX, double sizeY, bool normalizeColors)
    {
        int nx = field.GetLength(0);
        int ny = field.GetLength(1);
        var ax = plane.XAxis;
        ax.Unitize();
        var ay = plane.YAxis;
        ay.Unitize();

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

        var mesh = new Mesh();
        for (int ix = 0; ix < nx; ix++)
        {
            for (int iy = 0; iy < ny; iy++)
            {
                var pt = plane.Origin + ax * (ix / (double)(nx - 1) * sizeX) + ay * (iy / (double)(ny - 1) * sizeY);
                mesh.Vertices.Add(pt);
                float t = normalizeColors ? (field[ix, iy] - min) / denom : field[ix, iy];
                t = Math.Clamp(t, 0f, 1f);
                Color c = Field2DColormap.TurboColor(t);
                mesh.VertexColors.Add(c.R, c.G, c.B);
            }
        }

        for (int ix = 0; ix < nx - 1; ix++)
        {
            for (int iy = 0; iy < ny - 1; iy++)
            {
                int i00 = ix * ny + iy;
                int i10 = (ix + 1) * ny + iy;
                int i11 = (ix + 1) * ny + (iy + 1);
                int i01 = ix * ny + (iy + 1);
                mesh.Faces.AddFace(i00, i10, i11, i01);
            }
        }

        mesh.Normals.ComputeNormals();
        mesh.Compact();
        return mesh;
    }
}

internal static class Field2DColormap
{
    internal static Color TurboColor(float t)
    {
        t = Math.Clamp(t, 0f, 1f);
        float r = Math.Clamp(1.5f - Math.Abs(4f * t - 3f), 0f, 1f);
        float g = Math.Clamp(1.5f - Math.Abs(4f * t - 2f), 0f, 1f);
        float b = Math.Clamp(1.5f - Math.Abs(4f * t - 1f), 0f, 1f);
        return Color.FromArgb(255, (int)(r * 255), (int)(g * 255), (int)(b * 255));
    }
}
