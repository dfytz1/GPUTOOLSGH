using Rhino.Geometry;

namespace GHGPUPlugin.Algorithms;

/// <summary>March triangle edges for isolines and chain segments with endpoint merging.</summary>
public static class MeshIsolineHelper
{
    public static List<Curve> ExtractIsolinesForValue(Mesh mesh, double[] scalarPerMeshVertex, double iso, double mergeTol)
    {
        var segments = new List<Line>();

        for (int fi = 0; fi < mesh.Faces.Count; fi++)
        {
            MeshFace f = mesh.Faces[fi];
            if (f.IsTriangle)
                MarchTriangle(mesh, f.A, f.B, f.C, scalarPerMeshVertex, iso, segments);
            else if (f.IsQuad)
            {
                MarchTriangle(mesh, f.A, f.B, f.C, scalarPerMeshVertex, iso, segments);
                MarchTriangle(mesh, f.A, f.C, f.D, scalarPerMeshVertex, iso, segments);
            }
        }

        return ChainSegments(segments, mergeTol, mergeTol * 2 + 1e-7);
    }

    private static void MarchTriangle(
        Mesh mesh,
        int ia,
        int ib,
        int ic,
        double[] s,
        double iso,
        List<Line> segments)
    {
        Point3d pa = mesh.Vertices[ia];
        Point3d pb = mesh.Vertices[ib];
        Point3d pc = mesh.Vertices[ic];
        double sa = s[ia], sb = s[ib], sc = s[ic];

        var hits = new List<Point3d>(3);
        TryEdge(iso, pa, pb, sa, sb, hits);
        TryEdge(iso, pb, pc, sb, sc, hits);
        TryEdge(iso, pc, pa, sc, sa, hits);

        if (hits.Count >= 2)
            segments.Add(new Line(hits[0], hits[1]));
    }

    private static void TryEdge(double iso, Point3d p0, Point3d p1, double s0, double s1, List<Point3d> hits)
    {
        if (Math.Abs(s1 - s0) < 1e-30)
            return;
        bool c0 = s0 <= iso && iso < s1;
        bool c1 = s1 <= iso && iso < s0;
        if (!c0 && !c1)
            return;
        double t = (iso - s0) / (s1 - s0);
        t = Math.Clamp(t, 0, 1);
        hits.Add(p0 + (p1 - p0) * t);
    }

    private static List<Curve> ChainSegments(List<Line> segments, double mergeTol, double tipMatchEps)
    {
        if (segments.Count == 0)
            return new List<Curve>();

        var merger = new EndpointMerger(mergeTol);
        var v0 = new List<int>();
        var v1 = new List<int>();
        var p0 = new List<Point3d>();
        var p1 = new List<Point3d>();

        foreach (Line ln in segments)
        {
            Point3d a = ln.From, b = ln.To;
            int ida = merger.AddOrGet(a);
            int idb = merger.AddOrGet(b);
            if (ida == idb)
                continue;
            v0.Add(ida);
            v1.Add(idb);
            p0.Add(a);
            p1.Add(b);
        }

        int m = v0.Count;
        if (m == 0)
            return new List<Curve>();

        int maxC = 0;
        for (int i = 0; i < m; i++)
        {
            maxC = Math.Max(maxC, v0[i]);
            maxC = Math.Max(maxC, v1[i]);
        }

        var adj = new List<int>[maxC + 1];
        for (int c = 0; c <= maxC; c++)
            adj[c] = new List<int>();
        for (int i = 0; i < m; i++)
        {
            adj[v0[i]].Add(i);
            adj[v1[i]].Add(i);
        }

        var used = new bool[m];
        var curves = new List<Curve>();

        for (int si = 0; si < m; si++)
        {
            if (used[si])
                continue;
            used[si] = true;
            var pts = new List<Point3d> { p0[si], p1[si] };
            int tipC = v1[si];
            Point3d tipP = p1[si];

            while (TryExtend(ref tipC, ref tipP, pts, append: true, v0, v1, p0, p1, adj, used, tipMatchEps))
            {
            }

            tipC = v0[si];
            tipP = p0[si];
            pts.Reverse();
            while (TryExtend(ref tipC, ref tipP, pts, append: false, v0, v1, p0, p1, adj, used, tipMatchEps))
            {
            }

            if (pts.Count >= 2)
                curves.Add(new PolylineCurve(pts));
        }

        return curves;
    }

    private static bool TryExtend(
        ref int tipC,
        ref Point3d tipP,
        List<Point3d> pts,
        bool append,
        List<int> v0,
        List<int> v1,
        List<Point3d> p0,
        List<Point3d> p1,
        List<int>[] adj,
        bool[] used,
        double tipMatchEps)
    {
        double eps = tipMatchEps;
        foreach (int j in adj[tipC])
        {
            if (used[j])
                continue;
            int a = v0[j], b = v1[j];
            Point3d pa = p0[j], pb = p1[j];
            Point3d otherP;
            int otherC;
            if (a == tipC && pa.DistanceTo(tipP) <= eps)
            {
                otherP = pb;
                otherC = b;
            }
            else if (b == tipC && pb.DistanceTo(tipP) <= eps)
            {
                otherP = pa;
                otherC = a;
            }
            else
                continue;

            used[j] = true;
            if (append)
                pts.Add(otherP);
            else
                pts.Insert(0, otherP);
            tipC = otherC;
            tipP = otherP;
            return true;
        }

        return false;
    }

    private sealed class EndpointMerger
    {
        private readonly List<Point3d> _pts = new();
        private readonly Dictionary<(long X, long Y, long Z), List<int>> _cells = new();
        private readonly double _cell;
        private readonly double _tolSq;

        public EndpointMerger(double tol)
        {
            _cell = tol > 0 ? tol : 1e-6;
            _tolSq = _cell * _cell;
        }

        public int AddOrGet(Point3d p)
        {
            (long cx, long cy, long cz) = CellOf(p);
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
                    if (_pts[vi].DistanceToSquared(p) <= _tolSq)
                        return vi;
                }
            }

            int idx = _pts.Count;
            _pts.Add(p);
            var home = (cx, cy, cz);
            if (!_cells.TryGetValue(home, out List<int>? list))
            {
                list = new List<int>();
                _cells[home] = list;
            }

            list.Add(idx);
            return idx;
        }

        private (long X, long Y, long Z) CellOf(Point3d p)
        {
            return (
                (long)Math.Floor(p.X / _cell),
                (long)Math.Floor(p.Y / _cell),
                (long)Math.Floor(p.Z / _cell));
        }
    }
}
