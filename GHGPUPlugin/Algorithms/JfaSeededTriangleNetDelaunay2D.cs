using GHGPUPlugin.NativeInterop;
using Rhino.Geometry;
using TriangleNet.Geometry;
using TriangleNet.Meshing;

namespace GHGPUPlugin.Algorithms;

/// <summary>
/// GPU JFA approximate Delaunay edges, then Triangle.NET constrained triangulation to recover a triangle mesh.
/// Falls back to <see cref="AnisoCvtDelaunay2D.BowyerWatson"/> when Metal or Triangle.NET fails or adds Steiner points.
/// </summary>
public static class JfaSeededTriangleNetDelaunay2D
{
    /// <summary>
    /// <paramref name="triangles"/> are triples of indices into <paramref name="uv"/>.
    /// </summary>
    public static bool TryTriangulate(
        IReadOnlyList<Vector2d> uv,
        IntPtr metalCtx,
        int gridResolution,
        out List<int> triangles,
        out string detail)
    {
        triangles = new List<int>();
        detail = string.Empty;
        int n = uv.Count;
        if (n < 3 || metalCtx == IntPtr.Zero)
        {
            detail = "need ≥3 points and Metal context";
            return false;
        }

        if (!JfaDelaunay2DPlanar.TryJfaNormalizedCoords(uv, out float[] jfaPx, out float[] jfaPy))
        {
            detail = "degenerate UV span";
            return false;
        }

        int maxEdges = n * 12;
        var outA = new int[maxEdges];
        var outB = new int[maxEdges];
        int code = MetalBridge.JfaDelaunay2D(metalCtx, jfaPx, jfaPy, n, outA, outB, out int edgeCount, maxEdges, gridResolution);
        if (code != 0)
        {
            detail = $"JFA error {code}";
            return false;
        }

        if (edgeCount < 1)
        {
            detail = "JFA returned no edges";
            return false;
        }

        var polygon = new Polygon();
        var verts = new Vertex[n];
        for (int i = 0; i < n; i++)
        {
            verts[i] = new Vertex(uv[i].X, uv[i].Y);
            polygon.Add(verts[i]);
        }

        var seen = new HashSet<(int Lo, int Hi)>();
        for (int e = 0; e < edgeCount; e++)
        {
            int a = outA[e];
            int b = outB[e];
            if (a < 0 || b < 0 || a == b || a >= n || b >= n)
                continue;
            int lo = a < b ? a : b;
            int hi = a < b ? b : a;
            if (!seen.Add((lo, hi)))
                continue;
            polygon.Add(new Segment(verts[lo], verts[hi]), false);
        }

        try
        {
            var opts = new ConstraintOptions
            {
                Convex = true,
                ConformingDelaunay = false,
                SegmentSplitting = 2
            };

            TriangleNet.Meshing.IMesh mesh = polygon.Triangulate(opts);

            if (mesh.Vertices.Count != n)
            {
                detail = $"Triangle.NET added vertices ({mesh.Vertices.Count} vs {n})";
                return false;
            }

            foreach (var t in mesh.Triangles)
            {
                int id0 = t.GetVertexID(0);
                int id1 = t.GetVertexID(1);
                int id2 = t.GetVertexID(2);
                if ((uint)id0 >= (uint)n || (uint)id1 >= (uint)n || (uint)id2 >= (uint)n)
                {
                    detail = "triangle references unexpected vertex id";
                    return false;
                }

                triangles.Add(id0);
                triangles.Add(id1);
                triangles.Add(id2);
            }

            if (triangles.Count < 3)
            {
                detail = "too few triangles from Triangle.NET";
                triangles.Clear();
                return false;
            }

            detail = $"JFA {edgeCount} edges → Triangle.NET {triangles.Count / 3} tris";
            return true;
        }
        catch (Exception ex)
        {
            detail = ex.Message;
            triangles.Clear();
            return false;
        }
    }
}
