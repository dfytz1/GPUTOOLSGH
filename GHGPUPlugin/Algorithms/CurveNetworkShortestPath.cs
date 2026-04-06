using System.Linq;
using Rhino;
using Rhino.Geometry;

namespace GHGPUPlugin.Algorithms;

/// <summary>Shortest path along a network of curves (Dijkstra); undirected edges weighted by segment length.</summary>
public static class CurveNetworkShortestPath
{
    public static bool TryFindPath(
        IEnumerable<Curve> curves,
        Point3d startPt,
        Point3d endPt,
        double mergeTolerance,
        double maxEdgeLength,
        double snapTolerance,
        out List<Point3d> path,
        out double length,
        out string? error)
    {
        path = new List<Point3d>();
        length = 0;
        error = null;

        if (mergeTolerance <= 0)
        {
            error = "Merge tolerance must be positive.";
            return false;
        }

        if (maxEdgeLength <= 0)
        {
            error = "Max edge length must be positive.";
            return false;
        }

        if (snapTolerance <= 0)
        {
            error = "Snap tolerance must be positive.";
            return false;
        }

        var verts = new List<Point3d>();
        var edges = new List<(int A, int B, double W)>();

        foreach (Curve c in curves)
        {
            if (c == null || !c.IsValid)
                continue;

            if (!TrySampleCurve(c, maxEdgeLength, out List<Point3d>? pts) || pts == null || pts.Count < 2)
                continue;

            for (int i = 0; i < pts.Count - 1; i++)
            {
                int a = AddOrGetVertex(verts, pts[i], mergeTolerance);
                int b = AddOrGetVertex(verts, pts[i + 1], mergeTolerance);
                if (a == b)
                    continue;
                double w = pts[i].DistanceTo(pts[i + 1]);
                edges.Add((a, b, w));
                edges.Add((b, a, w));
            }
        }

        if (verts.Count == 0)
        {
            error = "No usable curve segments (need at least one curve with two or more sample points).";
            return false;
        }

        int startV = NearestVertexWithin(verts, startPt, snapTolerance, out _);
        int endV = NearestVertexWithin(verts, endPt, snapTolerance, out _);
        if (startV < 0)
        {
            error = "Start point is too far from the curve network (increase Snap tolerance or move the point).";
            return false;
        }

        if (endV < 0)
        {
            error = "End point is too far from the curve network (increase Snap tolerance or move the point).";
            return false;
        }

        if (startV == endV)
        {
            path.Add(verts[startV]);
            length = 0;
            return true;
        }

        int n = verts.Count;
        var adj = new List<(int v, float w)>[n];
        for (int i = 0; i < n; i++)
            adj[i] = new List<(int, float)>();
        foreach (var (a, b, w) in edges)
            adj[a].Add((b, (float)w));

        var dist = new double[n];
        var prev = new int[n];
        for (int i = 0; i < n; i++)
        {
            dist[i] = double.PositiveInfinity;
            prev[i] = -1;
        }

        dist[startV] = 0;
        var pq = new PriorityQueue<int, double>();
        pq.Enqueue(startV, 0);

        while (pq.TryDequeue(out int u, out double du))
        {
            if (du > dist[u])
                continue;
            if (u == endV)
                break;

            foreach (var (v, w) in adj[u])
            {
                double nd = du + w;
                if (nd < dist[v])
                {
                    dist[v] = nd;
                    prev[v] = u;
                    pq.Enqueue(v, nd);
                }
            }
        }

        if (double.IsPositiveInfinity(dist[endV]))
        {
            error = "No path between start and end through the given curves (network may be disconnected).";
            return false;
        }

        length = dist[endV];
        var order = new List<int>();
        for (int at = endV; ; at = prev[at])
        {
            order.Add(at);
            if (at == startV)
                break;
            if (prev[at] == -1)
            {
                error = "Path reconstruction failed.";
                return false;
            }
        }

        order.Reverse();
        path = new List<Point3d>(order.Count);
        foreach (int vi in order)
            path.Add(verts[vi]);
        return true;
    }

    private static int AddOrGetVertex(List<Point3d> verts, Point3d p, double tol)
    {
        double tolSq = tol * tol;
        for (int i = 0; i < verts.Count; i++)
        {
            if (verts[i].DistanceToSquared(p) <= tolSq)
                return i;
        }

        verts.Add(p);
        return verts.Count - 1;
    }

    private static int NearestVertexWithin(List<Point3d> verts, Point3d p, double maxDist, out double bestD)
    {
        int best = -1;
        bestD = double.MaxValue;
        double maxSq = maxDist * maxDist;
        for (int i = 0; i < verts.Count; i++)
        {
            double d2 = verts[i].DistanceToSquared(p);
            if (d2 > maxSq)
                continue;
            double d = Math.Sqrt(d2);
            if (best < 0 || d < bestD)
            {
                bestD = d;
                best = i;
            }
        }

        return best;
    }

    private static bool TrySampleCurve(Curve crv, double maxEdgeLength, out List<Point3d>? points)
    {
        points = null;
        if (crv.TryGetPolyline(out Polyline pl))
        {
            points = new List<Point3d>(pl.Count);
            for (int i = 0; i < pl.Count; i++)
                points.Add(pl[i]);
            return points.Count >= 2;
        }

        if (crv.IsLinear(RhinoMath.ZeroTolerance))
        {
            points = new List<Point3d> { crv.PointAtStart, crv.PointAtEnd };
            return true;
        }

        double len = crv.GetLength();
        if (len < 1e-12)
            return false;

        int segCount = Math.Max(1, (int)Math.Ceiling(len / maxEdgeLength));
        double[]? ts = crv.DivideByCount(segCount, true);
        if (ts == null || ts.Length < 2)
            return false;

        points = ts.Select(t => crv.PointAt(t)).ToList();
        return points.Count >= 2;
    }
}
