using System;
using Rhino.Geometry;

namespace GHGPUPlugin.Components.Field;

internal static class Field2DPlaneSampling
{
    internal static void PlaneAxes(Plane pl, out Vector3d ax, out Vector3d ay)
    {
        ax = pl.XAxis;
        ax.Unitize();
        ay = pl.YAxis;
        ay.Unitize();
    }

    /// <summary>Domain fraction along plane axes: (0,0) at origin, (1,1) at origin+sx·X+sy·Y.</summary>
    internal static void WorldToFraction(Plane pl, Vector3d ax, Vector3d ay, double sx, double sy, Point3d p, out double fu, out double fv)
    {
        Vector3d w = p - pl.Origin;
        fu = (w * ax) / sx;
        fv = (w * ay) / sy;
    }

    internal static Point3d CellCenterWorld(Plane pl, Vector3d ax, Vector3d ay, double sx, double sy, int nx, int ny, int ix, int iy) =>
        pl.Origin + ax * ((ix + 0.5) / nx * sx) + ay * ((iy + 0.5) / ny * sy);

    /// <summary>Cell-centered <c>float[nx,ny]</c>: samples using the same layout as the reaction-diffusion grid.</summary>
    internal static float SampleBilinear(float[,] data, int nx, int ny, double fu, double fv)
    {
        fu = Math.Clamp(fu, 0.0, 1.0);
        fv = Math.Clamp(fv, 0.0, 1.0);
        double fx = fu * nx - 0.5;
        double fy = fv * ny - 0.5;
        int x0 = (int)Math.Floor(fx);
        int y0 = (int)Math.Floor(fy);
        x0 = Math.Clamp(x0, 0, nx - 2);
        y0 = Math.Clamp(y0, 0, ny - 2);
        double tx = fx - x0;
        double ty = fy - y0;
        float v00 = data[x0, y0];
        float v10 = data[x0 + 1, y0];
        float v01 = data[x0, y0 + 1];
        float v11 = data[x0 + 1, y0 + 1];
        float a0 = (float)((1 - tx) * v00 + tx * v10);
        float a1 = (float)((1 - tx) * v01 + tx * v11);
        return (float)((1 - ty) * a0 + ty * a1);
    }

    internal static float SampleAtWorld(float[,] data, int nx, int ny, Plane pl, double sx, double sy, Point3d p)
    {
        PlaneAxes(pl, out Vector3d ax, out Vector3d ay);
        WorldToFraction(pl, ax, ay, sx, sy, p, out double fu, out double fv);
        return SampleBilinear(data, nx, ny, fu, fv);
    }
}
