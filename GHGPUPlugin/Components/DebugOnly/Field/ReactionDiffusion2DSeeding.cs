using System;
using System.Collections.Generic;
using Rhino.Geometry;

namespace GHGPUPlugin.Components.Field;

internal static class ReactionDiffusion2DSeeding
{
    internal static void SplatAll(
        float[,] b,
        int nx,
        int ny,
        Plane pl,
        double sx,
        double sy,
        IReadOnlyList<Point3d> seedPoints,
        IReadOnlyList<Curve> seedCurves,
        double radiusWorld,
        bool useDefaultCenter,
        Mesh? meshVerticesToSplat = null)
    {
        Field2DPlaneSampling.PlaneAxes(pl, out Vector3d ax, out Vector3d ay);

        var pts = new List<Point3d>(seedPoints);
        if (useDefaultCenter && pts.Count == 0 && (seedCurves == null || seedCurves.Count == 0) && meshVerticesToSplat == null)
        {
            pts.Add(pl.Origin + ax * (0.5 * sx) + ay * (0.5 * sy));
        }

        double r2 = radiusWorld * radiusWorld;

        if (meshVerticesToSplat != null)
            SplatMeshVertices(b, nx, ny, pl, sx, sy, ax, ay, meshVerticesToSplat);

        if (pts.Count > 0)
        {
            for (int ix = 0; ix < nx; ix++)
            {
                for (int iy = 0; iy < ny; iy++)
                {
                    var cell = Field2DPlaneSampling.CellCenterWorld(pl, ax, ay, sx, sy, nx, ny, ix, iy);
                    foreach (Point3d p in pts)
                    {
                        if (cell.DistanceToSquared(p) <= r2)
                        {
                            b[ix, iy] = 1f;
                            break;
                        }
                    }
                }
            }
        }

        if (seedCurves != null && seedCurves.Count > 0 && radiusWorld > 0)
        {
            for (int ix = 0; ix < nx; ix++)
            {
                for (int iy = 0; iy < ny; iy++)
                {
                    if (b[ix, iy] >= 1f)
                        continue;
                    var cell = Field2DPlaneSampling.CellCenterWorld(pl, ax, ay, sx, sy, nx, ny, ix, iy);
                    foreach (Curve? c in seedCurves)
                    {
                        if (c == null || !c.IsValid)
                            continue;
                        if (!c.ClosestPoint(cell, out double t))
                            continue;
                        Point3d onC = c.PointAt(t);
                        if (cell.DistanceTo(onC) <= radiusWorld)
                        {
                            b[ix, iy] = 1f;
                            break;
                        }
                    }
                }
            }
        }
    }

    private static void SplatMeshVertices(float[,] b, int nx, int ny, Plane pl, double sx, double sy, Vector3d ax, Vector3d ay, Mesh mesh)
    {
        for (int vi = 0; vi < mesh.Vertices.Count; vi++)
        {
            Point3d p = mesh.Vertices[vi];
            Field2DPlaneSampling.WorldToFraction(pl, ax, ay, sx, sy, p, out double fu, out double fv);
            if (fu < 0 || fu > 1 || fv < 0 || fv > 1)
                continue;
            int ix = (int)Math.Clamp(Math.Round(fu * nx - 0.5), 0, nx - 1);
            int iy = (int)Math.Clamp(Math.Round(fv * ny - 0.5), 0, ny - 1);
            b[ix, iy] = 1f;
        }
    }
}
