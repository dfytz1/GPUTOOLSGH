using Rhino.Geometry;

namespace GHGPUPlugin.Algorithms;

/// <summary>2D Delaunay (Bowyer–Watson) in a plane parameterization; triangle indices reference the input point list.</summary>
public static class PlanarDelaunay2D
{
    private readonly struct Tri
    {
        public readonly int A;
        public readonly int B;
        public readonly int C;

        public Tri(int a, int b, int c)
        {
            A = a;
            B = b;
            C = c;
        }
    }

    /// <summary>Returns triangles as triples of indices into <paramref name="uv"/> (length ≥ 3).</summary>
    public static List<int> BowyerWatson(IReadOnlyList<Vector2d> uv)
    {
        int n = uv.Count;
        if (n < 3)
            return new List<int>();

        double minX = double.MaxValue, minY = double.MaxValue, maxX = double.MinValue, maxY = double.MinValue;
        for (int i = 0; i < n; i++)
        {
            var p = uv[i];
            if (p.X < minX)
                minX = p.X;
            if (p.X > maxX)
                maxX = p.X;
            if (p.Y < minY)
                minY = p.Y;
            if (p.Y > maxY)
                maxY = p.Y;
        }

        double dx = maxX - minX;
        double dy = maxY - minY;
        double d = Math.Max(dx, dy);
        if (d < 1e-12)
            d = 1.0;
        double midX = 0.5 * (minX + maxX);
        double midY = 0.5 * (minY + maxY);

        var pts = new List<Vector2d>(n + 3);
        for (int i = 0; i < n; i++)
            pts.Add(uv[i]);
        double big = d * 10.0;
        pts.Add(new Vector2d(midX - 2 * big, midY - big));
        pts.Add(new Vector2d(midX + big, midY - big));
        pts.Add(new Vector2d(midX, midY + 2 * big));

        var tris = new List<Tri> { new(n, n + 1, n + 2) };

        for (int pi = 0; pi < n; pi++)
        {
            var p = pts[pi];
            var isBad = new bool[tris.Count];
            int nBad = 0;
            for (int t = 0; t < tris.Count; t++)
            {
                Tri tri = tris[t];
                if (InCircumcircle(pts[tri.A], pts[tri.B], pts[tri.C], p))
                {
                    isBad[t] = true;
                    nBad++;
                }
            }

            if (nBad == 0)
                continue;

            var edgeCount = new Dictionary<(int A, int B), int>();
            for (int t = 0; t < tris.Count; t++)
            {
                if (!isBad[t])
                    continue;
                Tri tri = tris[t];
                AddEdge(edgeCount, tri.A, tri.B);
                AddEdge(edgeCount, tri.B, tri.C);
                AddEdge(edgeCount, tri.C, tri.A);
            }

            var next = new List<Tri>();
            for (int t = 0; t < tris.Count; t++)
            {
                if (!isBad[t])
                    next.Add(tris[t]);
            }

            foreach (KeyValuePair<(int A, int B), int> kv in edgeCount)
            {
                if (kv.Value != 1)
                    continue;
                next.Add(new Tri(kv.Key.A, kv.Key.B, pi));
            }

            tris = next;
        }

        var outTris = new List<int>();
        for (int t = 0; t < tris.Count; t++)
        {
            Tri tri = tris[t];
            if (tri.A >= n || tri.B >= n || tri.C >= n)
                continue;
            outTris.Add(tri.A);
            outTris.Add(tri.B);
            outTris.Add(tri.C);
        }

        return outTris;
    }

    /// <summary>Circumradius of triangle in the UV plane.</summary>
    public static double CircumradiusUv(Vector2d a, Vector2d b, Vector2d c)
    {
        double lab = (b - a).Length;
        double lbc = (c - b).Length;
        double lca = (a - c).Length;
        double cross = (b.X - a.X) * (c.Y - a.Y) - (b.Y - a.Y) * (c.X - a.X);
        double area2 = Math.Abs(cross);
        if (area2 < 1e-30)
            return double.PositiveInfinity;
        return lab * lbc * lca / area2;
    }

    private static void AddEdge(Dictionary<(int A, int B), int> edgeCount, int i, int j)
    {
        int lo = i < j ? i : j;
        int hi = i < j ? j : i;
        var k = (lo, hi);
        edgeCount.TryGetValue(k, out int c);
        edgeCount[k] = c + 1;
    }

    private static bool InCircumcircle(Vector2d a, Vector2d b, Vector2d c, Vector2d p)
    {
        double ax = a.X - p.X, ay = a.Y - p.Y;
        double bx = b.X - p.X, by = b.Y - p.Y;
        double cx = c.X - p.X, cy = c.Y - p.Y;
        double a2 = ax * ax + ay * ay;
        double b2 = bx * bx + by * by;
        double c2 = cx * cx + cy * cy;
        double det = ax * (by * c2 - cy * b2) - ay * (bx * c2 - cx * b2) + a2 * (bx * cy - cx * by);
        double orient = (b.X - a.X) * (c.Y - a.Y) - (b.Y - a.Y) * (c.X - a.X);
        return orient > 0 ? det > 0 : det < 0;
    }
}
