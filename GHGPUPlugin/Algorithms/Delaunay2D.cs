using GHGPUPlugin.NativeInterop;
using Rhino.Geometry;

namespace GHGPUPlugin.Algorithms;

/// <summary>Planar Delaunay (Bowyer–Watson). Optional GPU pass marks bad triangles per inserted point.</summary>
public static class Delaunay2D
{
    public static bool TryTriangulate(
        IReadOnlyList<Point2d> userPts,
        bool useGpu,
        out List<(int A, int B, int C)> trianglesOut,
        out string? error)
    {
        trianglesOut = new List<(int, int, int)>();
        error = null;
        int n = userPts.Count;
        if (n < 3)
        {
            error = "Need at least three coplanar distinct points.";
            return false;
        }

        double minX = userPts[0].X, maxX = minX, minY = userPts[0].Y, maxY = minY;
        for (int i = 1; i < n; i++)
        {
            var p = userPts[i];
            minX = Math.Min(minX, p.X);
            maxX = Math.Max(maxX, p.X);
            minY = Math.Min(minY, p.Y);
            maxY = Math.Max(maxY, p.Y);
        }

        double dx = maxX - minX;
        double dy = maxY - minY;
        double dmax = Math.Max(dx, dy);
        if (dmax < 1e-30)
        {
            error = "Points are degenerate (collapsed).";
            return false;
        }

        double margin = dmax * 10 + 1;
        double cx = (minX + maxX) * 0.5;
        double cy = (minY + maxY) * 0.5;
        var super0 = new Point2d(cx - 2 * margin, cy - margin);
        var super1 = new Point2d(cx, cy + 2 * margin);
        var super2 = new Point2d(cx + 2 * margin, cy - margin);

        var pts = new List<Point2d>(userPts) { super0, super1, super2 };
        int sn = pts.Count;
        int s0 = n, s1 = n + 1, s2 = n + 2;

        var tris = new List<(int A, int B, int C)> { (s0, s1, s2) };

        var px = new float[sn];
        var py = new float[sn];
        for (int i = 0; i < sn; i++)
        {
            px[i] = (float)pts[i].X;
            py[i] = (float)pts[i].Y;
        }

        IntPtr ctx = IntPtr.Zero;
        bool haveCtx = useGpu && NativeLoader.IsMetalAvailable && MetalSharedContext.TryGetContext(out ctx);

        for (int pi = 0; pi < n; pi++)
        {
            float qx = (float)userPts[pi].X;
            float qy = (float)userPts[pi].Y;

            var bad = new List<int>();
            int tc = tris.Count;
            if (haveCtx && tc > 0)
            {
                var triFlat = new int[tc * 3];
                for (int t = 0; t < tc; t++)
                {
                    triFlat[3 * t] = tris[t].A;
                    triFlat[3 * t + 1] = tris[t].B;
                    triFlat[3 * t + 2] = tris[t].C;
                }

                var mask = new int[tc];
                int code = MetalBridge.DelaunayMarkBadTriangles(ctx, px, py, sn, triFlat, tc, qx, qy, mask);
                if (code != 0)
                {
                    haveCtx = false;
                    bad.Clear();
                    for (int t = 0; t < tc; t++)
                    {
                        if (InCircumcircle(pts[tris[t].A], pts[tris[t].B], pts[tris[t].C], userPts[pi]))
                            bad.Add(t);
                    }
                }
                else
                {
                    for (int t = 0; t < tc; t++)
                    {
                        if (mask[t] != 0)
                            bad.Add(t);
                    }
                }
            }
            else
            {
                for (int t = 0; t < tc; t++)
                {
                    if (InCircumcircle(pts[tris[t].A], pts[tris[t].B], pts[tris[t].C], userPts[pi]))
                        bad.Add(t);
                }
            }

            if (bad.Count == 0)
                continue;

            var edgeCount = new Dictionary<(int, int), int>();
            void AddEdge(int a, int b)
            {
                int u = Math.Min(a, b);
                int v = Math.Max(a, b);
                var k = (u, v);
                edgeCount[k] = edgeCount.GetValueOrDefault(k) + 1;
            }

            foreach (int bi in bad)
            {
                var t = tris[bi];
                AddEdge(t.A, t.B);
                AddEdge(t.B, t.C);
                AddEdge(t.C, t.A);
            }

            var boundary = new List<(int, int)>();
            foreach (var kv in edgeCount)
            {
                if (kv.Value == 1)
                    boundary.Add(kv.Key);
            }

            bad.Sort();
            for (int i = bad.Count - 1; i >= 0; i--)
                tris.RemoveAt(bad[i]);

            int pIdx = pi;
            foreach (var e in boundary)
            {
                int u = e.Item1;
                int v = e.Item2;
                if (Orient2d(pts[u], pts[v], pts[pIdx]) > 0)
                    tris.Add((u, v, pIdx));
                else
                    tris.Add((v, u, pIdx));
            }
        }

        for (int i = tris.Count - 1; i >= 0; i--)
        {
            var t = tris[i];
            if (t.A >= n || t.B >= n || t.C >= n)
                tris.RemoveAt(i);
        }

        if (tris.Count == 0)
        {
            error = "Delaunay produced no triangles (check collinearity).";
            return false;
        }

        trianglesOut = tris;
        return true;
    }

    private static double Orient2d(Point2d a, Point2d b, Point2d c) =>
        (b.X - a.X) * (c.Y - a.Y) - (b.Y - a.Y) * (c.X - a.X);

    private static bool InCircumcircle(Point2d a, Point2d b, Point2d c, Point2d d)
    {
        double ax = a.X - d.X, ay = a.Y - d.Y;
        double bx = b.X - d.X, by = b.Y - d.Y;
        double cx = c.X - d.X, cy = c.Y - d.Y;
        double det = (ax * ax + ay * ay) * (bx * cy - by * cx) - (bx * bx + by * by) * (ax * cy - ay * cx)
            + (cx * cx + cy * cy) * (ax * by - ay * bx);
        double orient = Orient2d(a, b, c);
        if (Math.Abs(orient) < 1e-20)
            return false;
        return orient > 0 ? det > 0 : det < 0;
    }
}
