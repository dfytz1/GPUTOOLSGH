using Rhino.Geometry;
using TriangleNet.Geometry;
using TriangleNet.Meshing;

namespace GHGPUPlugin.Algorithms;

/// <summary>Build a Triangle.NET polygon (boundary + holes) from planar Rhino curves, then triangulate.</summary>
public static class PlanarCdtFromCurves
{
    /// <summary>
    /// <paramref name="vertices3d"/> matches Triangle.NET vertex order for building a Rhino mesh (includes Steiner points).
    /// </summary>
    public static bool TryTriangulate(
        Curve boundary,
        IReadOnlyList<Curve> holes,
        Plane plane,
        double maxEdgeLength,
        double maxTriangleArea,
        out List<Point3d> vertices3d,
        out List<int> triangleIndices,
        out string detail)
    {
        vertices3d = new List<Point3d>();
        triangleIndices = new List<int>();
        detail = string.Empty;

        if (boundary == null || !boundary.IsValid)
        {
            detail = "Boundary curve is invalid.";
            return false;
        }

        maxEdgeLength = Math.Max(maxEdgeLength, 1e-9);

        if (!TryDiscretizeClosed(boundary, plane, maxEdgeLength, out List<Vector2d> outer2d, out List<Point3d> outer3d, out string e0))
        {
            detail = e0;
            return false;
        }

        if (outer2d.Count < 3)
        {
            detail = "Boundary has fewer than three vertices after discretization.";
            return false;
        }

        var polygon = new Polygon();
        var shellVerts = outer2d.Select(p => new Vertex(p.X, p.Y)).ToList();
        polygon.Add(new Contour(shellVerts));

        for (int hi = 0; hi < holes.Count; hi++)
        {
            Curve h = holes[hi];
            if (h == null || !h.IsValid)
                continue;

            if (!TryDiscretizeClosed(h, plane, maxEdgeLength, out List<Vector2d> hole2d, out _, out string eh))
            {
                detail = $"Hole {hi}: {eh}";
                return false;
            }

            if (hole2d.Count < 3)
            {
                detail = $"Hole {hi} has fewer than three vertices.";
                return false;
            }

            var hv = hole2d.Select(p => new Vertex(p.X, p.Y)).ToList();
            var holeContour = new Contour(hv);
            TriangleNet.Geometry.Point holePt;
            try
            {
                holePt = holeContour.FindInteriorPoint();
            }
            catch
            {
                detail = $"Could not find an interior point for hole {hi} (self-intersection or degenerate?).";
                return false;
            }

            polygon.Add(holeContour, holePt);
        }

        var constraint = new ConstraintOptions
        {
            Convex = false,
            SegmentSplitting = 0
        };

        try
        {
            TriangleNet.Meshing.IMesh mesh;
            if (maxTriangleArea > 0.0 && double.IsFinite(maxTriangleArea))
            {
                var quality = new QualityOptions
                {
                    MaximumArea = maxTriangleArea
                };
                mesh = polygon.Triangulate(constraint, quality);
            }
            else
            {
                mesh = polygon.Triangulate(constraint);
            }

            var idToIndex = new Dictionary<int, int>();
            foreach (var v in mesh.Vertices)
            {
                idToIndex[v.ID] = vertices3d.Count;
                vertices3d.Add(plane.PointAt(v.X, v.Y));
            }

            foreach (var t in mesh.Triangles)
            {
                int a = t.GetVertexID(0);
                int b = t.GetVertexID(1);
                int c = t.GetVertexID(2);
                if (!idToIndex.TryGetValue(a, out int ia) || !idToIndex.TryGetValue(b, out int ib) || !idToIndex.TryGetValue(c, out int ic))
                {
                    detail = "Unexpected vertex id in Triangle.NET output.";
                    return false;
                }

                triangleIndices.Add(ia);
                triangleIndices.Add(ib);
                triangleIndices.Add(ic);
            }

            detail = $"{vertices3d.Count} verts, {triangleIndices.Count / 3} tris";
            return triangleIndices.Count >= 3;
        }
        catch (Exception ex)
        {
            detail = ex.Message;
            vertices3d.Clear();
            triangleIndices.Clear();
            return false;
        }
    }

    /// <summary>Discretize a closed curve into plane UV and 3D points on <paramref name="plane"/>.</summary>
    private static bool TryDiscretizeClosed(
        Curve curve,
        Plane plane,
        double maxEdgeLength,
        out List<Vector2d> uv,
        out List<Point3d> lifted,
        out string error)
    {
        uv = new List<Vector2d>();
        lifted = new List<Point3d>();
        error = string.Empty;

        double len = curve.GetLength();
        if (len < 1e-12)
        {
            error = "Curve length is zero.";
            return false;
        }

        int nSeg = Math.Max(3, (int)Math.Ceiling(len / maxEdgeLength));
        nSeg = Math.Min(nSeg, 50000);

        if (!curve.IsClosed)
        {
            error = "Curve must be closed.";
            return false;
        }

        double[]? tDiv = curve.DivideByCount(nSeg, true);
        Point3d[]? pts = tDiv != null && tDiv.Length >= 3
            ? tDiv.Select(t => curve.PointAt(t)).ToArray()
            : null;
        if (pts == null || pts.Length < 3)
        {
            double[]? tParams = curve.DivideByLength(maxEdgeLength, true);
            if (tParams == null || tParams.Length < 3)
            {
                error = "Could not divide curve (try a larger MaxEdge length).";
                return false;
            }

            pts = tParams.Select(t => curve.PointAt(t)).ToArray();
        }

        for (int i = 0; i < pts.Length; i++)
        {
            plane.ClosestParameter(pts[i], out double uu, out double vv);
            var pOn = plane.PointAt(uu, vv);
            lifted.Add(pOn);
            uv.Add(new Vector2d(uu, vv));
        }

        RemoveClosingDuplicate(uv, lifted);
        if (uv.Count < 3)
        {
            error = "Too few unique points after discretization.";
            return false;
        }

        return true;
    }

    private static void RemoveClosingDuplicate(List<Vector2d> uv, List<Point3d> lifted)
    {
        if (uv.Count < 2)
            return;
        int last = uv.Count - 1;
        var du = uv[0] - uv[last];
        if (du.X * du.X + du.Y * du.Y < 1e-20 && lifted[0].DistanceToSquared(lifted[last]) < 1e-20)
        {
            uv.RemoveAt(last);
            lifted.RemoveAt(last);
        }
    }
}
