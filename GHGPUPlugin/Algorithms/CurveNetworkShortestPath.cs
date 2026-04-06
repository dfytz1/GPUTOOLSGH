using Rhino.Geometry;

namespace GHGPUPlugin.Algorithms;

/// <summary>Shortest path along a curve network (Dijkstra). Each curve is one undirected edge Start→End weighted by arc length.</summary>
public static class CurveNetworkShortestPath
{
    public static bool TryFindPath(
        IEnumerable<Curve> curves,
        Point3d startPt,
        Point3d endPt,
        double mergeTolerance,
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

        if (snapTolerance <= 0)
        {
            error = "Snap tolerance must be positive.";
            return false;
        }

        var verts = new List<Point3d>();
        var grid = new Dictionary<(long X, long Y, long Z), List<int>>();
        double cell = mergeTolerance;
        double mergeSq = mergeTolerance * mergeTolerance;

        var edges = new List<(int A, int B, double W)>();

        foreach (Curve c in curves)
        {
            if (c == null || !c.IsValid)
                continue;

            Point3d pa = c.PointAtStart;
            Point3d pb = c.PointAtEnd;
            int a = AddOrGetVertex(verts, grid, cell, mergeSq, pa);
            int b = AddOrGetVertex(verts, grid, cell, mergeSq, pb);
            if (a == b)
                continue;

            double w = c.GetLength();
            edges.Add((a, b, w));
            edges.Add((b, a, w));
        }

        if (verts.Count == 0)
        {
            error = "No usable curves (need at least one valid curve with distinct merged endpoints).";
            return false;
        }

        int snapRadiusCells = (int)Math.Ceiling(snapTolerance / mergeTolerance);
        int startV = NearestVertexWithinGrid(verts, grid, cell, startPt, snapTolerance, snapRadiusCells, out _);
        int endV = NearestVertexWithinGrid(verts, grid, cell, endPt, snapTolerance, snapRadiusCells, out _);
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

    private static (long X, long Y, long Z) CellOf(Point3d p, double cell)
    {
        return (
            (long)Math.Floor(p.X / cell),
            (long)Math.Floor(p.Y / cell),
            (long)Math.Floor(p.Z / cell));
    }

    /// <summary>Merges with an existing vertex within merge tolerance using a 3×3×3 cell neighborhood.</summary>
    private static int AddOrGetVertex(
        List<Point3d> verts,
        Dictionary<(long X, long Y, long Z), List<int>> grid,
        double cell,
        double mergeSq,
        Point3d p)
    {
        (long cx, long cy, long cz) = CellOf(p, cell);

        for (long dx = -1; dx <= 1; dx++)
        for (long dy = -1; dy <= 1; dy++)
        for (long dz = -1; dz <= 1; dz++)
        {
            var key = (cx + dx, cy + dy, cz + dz);
            if (!grid.TryGetValue(key, out List<int>? bucket))
                continue;
            for (int i = 0; i < bucket.Count; i++)
            {
                int vi = bucket[i];
                if (verts[vi].DistanceToSquared(p) <= mergeSq)
                    return vi;
            }
        }

        int idx = verts.Count;
        verts.Add(p);
        var home = (cx, cy, cz);
        if (!grid.TryGetValue(home, out List<int>? list))
        {
            list = new List<int>();
            grid[home] = list;
        }

        list.Add(idx);
        return idx;
    }

    /// <summary>Nearest vertex within <paramref name="maxDist"/> by scanning grid cells in a cube of half-size <paramref name="radiusCells"/> (in cell units).</summary>
    private static int NearestVertexWithinGrid(
        List<Point3d> verts,
        Dictionary<(long X, long Y, long Z), List<int>> grid,
        double cell,
        Point3d p,
        double maxDist,
        int radiusCells,
        out double bestD)
    {
        int best = -1;
        bestD = double.MaxValue;
        double maxSq = maxDist * maxDist;
        (long cx, long cy, long cz) = CellOf(p, cell);

        long r = radiusCells;
        for (long dx = -r; dx <= r; dx++)
        for (long dy = -r; dy <= r; dy++)
        for (long dz = -r; dz <= r; dz++)
        {
            var key = (cx + dx, cy + dy, cz + dz);
            if (!grid.TryGetValue(key, out List<int>? bucket))
                continue;
            for (int i = 0; i < bucket.Count; i++)
            {
                int vi = bucket[i];
                double d2 = verts[vi].DistanceToSquared(p);
                if (d2 > maxSq)
                    continue;
                double d = Math.Sqrt(d2);
                if (best < 0 || d < bestD)
                {
                    bestD = d;
                    best = vi;
                }
            }
        }

        return best;
    }
}
