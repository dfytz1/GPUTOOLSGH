using System;
using Rhino.Geometry;

namespace GHGPUPlugin.Components.Field;

internal static class MeshFieldDomainFit
{
    /// <summary>Fits a plane to vertices; on failure uses WorldXY through mesh bounding-box center.</summary>
    internal static bool TryReferencePlane(Mesh mesh, Plane? userPlane, out Plane plane, out string message)
    {
        message = string.Empty;
        if (userPlane.HasValue && userPlane.Value.IsValid)
        {
            plane = userPlane.Value;
            plane.XAxis.Unitize();
            plane.YAxis.Unitize();
            plane.ZAxis.Unitize();
            return true;
        }

        int n = mesh.Vertices.Count;
        if (n < 3)
        {
            message = "Mesh needs at least 3 vertices to fit a plane.";
            plane = Plane.Unset;
            return false;
        }

        var pts = new Point3d[n];
        for (int i = 0; i < n; i++)
            pts[i] = mesh.Vertices.Point3dAt(i);

        PlaneFitResult fit = Plane.FitPlaneToPoints(pts, out plane);
        if ((fit == PlaneFitResult.Success || fit == PlaneFitResult.Inconclusive) && plane.IsValid)
        {
            plane.XAxis.Unitize();
            plane.YAxis.Unitize();
            plane.ZAxis.Unitize();
            return true;
        }

        BoundingBox bb = mesh.GetBoundingBox(false);
        if (!bb.IsValid)
        {
            message = "Could not fit a plane or bounding box.";
            plane = Plane.Unset;
            return false;
        }

        plane = Plane.WorldXY;
        plane.Origin = bb.Center;
        message = "FitPlane failed; using WorldXY through bounding-box center.";
        return true;
    }

    /// <summary>Axis-aligned rectangle on <paramref name="referencePlane"/> that contains all vertex projections; origin at min corner.</summary>
    internal static bool TryDomainFromMesh(
        Mesh mesh,
        Plane referencePlane,
        double domainScale,
        out Plane domainPlane,
        out double sx,
        out double sy)
    {
        domainScale = Math.Max(domainScale, 1e-6);
        Field2DPlaneSampling.PlaneAxes(referencePlane, out Vector3d ax, out Vector3d ay);

        double uMin = double.MaxValue, uMax = double.MinValue;
        double vMin = double.MaxValue, vMax = double.MinValue;

        for (int i = 0; i < mesh.Vertices.Count; i++)
        {
            Point3d p = mesh.Vertices.Point3dAt(i);
            Vector3d w = p - referencePlane.Origin;
            double u = w * ax;
            double v = w * ay;
            if (u < uMin) uMin = u;
            if (u > uMax) uMax = u;
            if (v < vMin) vMin = v;
            if (v > vMax) vMax = v;
        }

        double du = uMax - uMin;
        double dv = vMax - vMin;
        const double Eps = 1e-9;
        if (du < Eps) du = Math.Max(referencePlane.Origin.DistanceTo(mesh.Vertices.Point3dAt(0)) * 0.01, Eps);
        if (dv < Eps) dv = du;

        double cu = (uMin + uMax) * 0.5;
        double cv = (vMin + vMax) * 0.5;
        double halfX = du * 0.5 * domainScale;
        double halfY = dv * 0.5 * domainScale;

        Point3d corner = referencePlane.Origin + ax * (cu - halfX) + ay * (cv - halfY);
        domainPlane = new Plane(corner, ax, ay);
        sx = 2.0 * halfX;
        sy = 2.0 * halfY;
        return sx > Eps && sy > Eps;
    }

    /// <summary>Cell counts with ~square world cells; <paramref name="resolution"/> is the count along the shorter domain side (after mul).</summary>
    internal static void ResolutionForDomain(double sx, double sy, int resolution, double resMul, out int nx, out int ny)
    {
        resolution = Math.Max(8, resolution);
        resMul = Math.Max(0.25, resMul);
        int nShort = Math.Max(8, (int)Math.Round(resolution * resMul));
        if (sx >= sy)
        {
            ny = nShort;
            nx = Math.Max(8, (int)Math.Round(nShort * (sx / Math.Max(sy, 1e-12))));
        }
        else
        {
            nx = nShort;
            ny = Math.Max(8, (int)Math.Round(nShort * (sy / Math.Max(sx, 1e-12))));
        }

        long cells = (long)nx * ny;
        if (cells > GrayScottField2DSolver.MaxCells)
        {
            double s = Math.Sqrt((double)GrayScottField2DSolver.MaxCells / (nx * (double)ny));
            nx = Math.Max(8, (int)(nx * s));
            ny = Math.Max(8, (int)(ny * s));
        }
    }

    internal static double AutoSeedRadius(double sx, double sy, double radiusMul) =>
        Math.Max(Math.Max(sx, sy) * 0.03 * Math.Max(radiusMul, 1e-6), 1e-9);
}
