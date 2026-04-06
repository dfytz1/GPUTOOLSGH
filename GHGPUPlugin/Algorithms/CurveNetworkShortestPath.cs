using Rhino.Geometry;

namespace GHGPUPlugin.Algorithms;

/// <summary>Graph built from curves (one undirected edge per curve between merged endpoints).</summary>
public sealed class CurveGraph
{
    public List<Point3d> Vertices { get; }
    public List<(int A, int B)> EdgePairs { get; }
    public List<Curve> EdgeCurves { get; }
    public List<double> EdgeLengths { get; }
    internal List<(int V, double W)>[] Adj { get; }

    internal CurveGraph(
        List<Point3d> vertices,
        List<(int A, int B)> edgePairs,
        List<Curve> edgeCurves,
        List<double> edgeLengths,
        List<(int V, double W)>[] adj)
    {
        Vertices = vertices;
        EdgePairs = edgePairs;
        EdgeCurves = edgeCurves;
        EdgeLengths = edgeLengths;
        Adj = adj;
    }
}

/// <summary>Shortest path on a <see cref="CurveGraph"/> (Dijkstra).</summary>
public static class CurveNetworkShortestPath
{
    public static CurveGraph BuildGraph(IEnumerable<Curve> curves, double mergeTolerance, bool useChordLength)
    {
        var edgePairs = new List<(int A, int B)>();
        var edgeCurves = new List<Curve>();
        var edgeLengths = new List<double>();

        if (mergeTolerance <= 0)
            return new CurveGraph(new List<Point3d>(), edgePairs, edgeCurves, edgeLengths, Array.Empty<List<(int V, double W)>>());

        var grid = new GridIndex(mergeTolerance);

        foreach (Curve c in curves)
        {
            if (c == null || !c.IsValid)
                continue;

            Point3d pa = c.PointAtStart;
            Point3d pb = c.PointAtEnd;
            int a = grid.AddOrGet(pa);
            int b = grid.AddOrGet(pb);
            if (a == b)
                continue;

            double w = useChordLength ? pa.DistanceTo(pb) : c.GetLength();
            edgePairs.Add((a, b));
            edgeCurves.Add(c);
            edgeLengths.Add(w);
        }

        List<Point3d> verts = grid.Vertices;
        int n = verts.Count;
        var adj = new List<(int V, double W)>[n];
        for (int i = 0; i < n; i++)
            adj[i] = new List<(int V, double W)>();

        for (int i = 0; i < edgePairs.Count; i++)
        {
            (int a, int b) = edgePairs[i];
            double w = edgeLengths[i];
            adj[a].Add((b, w));
            adj[b].Add((a, w));
        }

        return new CurveGraph(verts, edgePairs, edgeCurves, edgeLengths, adj);
    }

    public static bool TryFindPath(
        CurveGraph g,
        Point3d startPt,
        Point3d endPt,
        double snapTolerance,
        out List<int> pathIndices,
        out double length,
        out string? error)
    {
        pathIndices = new List<int>();
        length = 0;
        error = null;

        if (snapTolerance <= 0)
        {
            error = "Snap tolerance must be positive.";
            return false;
        }

        if (g.Vertices.Count == 0)
        {
            error = "Graph has no vertices.";
            return false;
        }

        int startV = NearestVertexLinear(g.Vertices, startPt, snapTolerance, out _);
        int endV = NearestVertexLinear(g.Vertices, endPt, snapTolerance, out _);
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
            pathIndices.Add(startV);
            length = 0;
            return true;
        }

        int n = g.Vertices.Count;
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

            foreach (var (v, w) in g.Adj[u])
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
        pathIndices = order;
        return true;
    }

    /// <summary>Single-source Dijkstra to all nodes; runs until the priority queue is empty.</summary>
    public static bool TrySingleSourceAll(
        CurveGraph g,
        Point3d sourcePt,
        double snapTolerance,
        out int sourceIdx,
        out double[]? dist,
        out int[]? prev,
        out string? error)
    {
        sourceIdx = -1;
        dist = null;
        prev = null;
        error = null;

        if (snapTolerance <= 0)
        {
            error = "Snap tolerance must be positive.";
            return false;
        }

        if (g.Vertices.Count == 0)
        {
            error = "Graph has no vertices.";
            return false;
        }

        int startV = NearestVertexLinear(g.Vertices, sourcePt, snapTolerance, out _);
        if (startV < 0)
        {
            error = "Source point is too far from the curve network (increase Snap tolerance or move the point).";
            return false;
        }

        sourceIdx = startV;
        int n = g.Vertices.Count;
        var distArr = new double[n];
        var prevArr = new int[n];
        for (int i = 0; i < n; i++)
        {
            distArr[i] = double.PositiveInfinity;
            prevArr[i] = -1;
        }

        distArr[startV] = 0;
        var pq = new PriorityQueue<int, double>();
        pq.Enqueue(startV, 0);

        while (pq.TryDequeue(out int u, out double du))
        {
            if (du > distArr[u])
                continue;

            foreach (var (v, w) in g.Adj[u])
            {
                double nd = du + w;
                if (nd < distArr[v])
                {
                    distArr[v] = nd;
                    prevArr[v] = u;
                    pq.Enqueue(v, nd);
                }
            }
        }

        dist = distArr;
        prev = prevArr;
        return true;
    }

    private static int NearestVertexLinear(List<Point3d> verts, Point3d p, double maxDist, out double bestD)
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

    /// <summary>Spatial hash for merging endpoints during graph build only.</summary>
    internal sealed class GridIndex
    {
        private readonly List<Point3d> _vertices = new();
        private readonly Dictionary<(long X, long Y, long Z), List<int>> _cells = new();
        private readonly double _cell;
        private readonly double _mergeSq;

        public GridIndex(double mergeTolerance)
        {
            _cell = mergeTolerance;
            _mergeSq = mergeTolerance * mergeTolerance;
        }

        public List<Point3d> Vertices => _vertices;

        public int AddOrGet(Point3d p)
        {
            (long cx, long cy, long cz) = CellOf(p, _cell);

            for (long dx = -1; dx <= 1; dx++)
            for (long dy = -1; dy <= 1; dy++)
            for (long dz = -1; dz <= 1; dz++)
            {
                var key = (cx + dx, cy + dy, cz + dz);
                if (!_cells.TryGetValue(key, out List<int>? bucket))
                    continue;
                for (int i = 0; i < bucket.Count; i++)
                {
                    int vi = bucket[i];
                    if (_vertices[vi].DistanceToSquared(p) <= _mergeSq)
                        return vi;
                }
            }

            int idx = _vertices.Count;
            _vertices.Add(p);
            var home = (cx, cy, cz);
            if (!_cells.TryGetValue(home, out List<int>? list))
            {
                list = new List<int>();
                _cells[home] = list;
            }

            list.Add(idx);
            return idx;
        }

        private static (long X, long Y, long Z) CellOf(Point3d p, double cell)
        {
            return (
                (long)Math.Floor(p.X / cell),
                (long)Math.Floor(p.Y / cell),
                (long)Math.Floor(p.Z / cell));
        }
    }
}
