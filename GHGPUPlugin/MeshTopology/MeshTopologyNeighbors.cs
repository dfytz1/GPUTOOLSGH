using Rhino;
using Rhino.Geometry;

namespace GHGPUPlugin.MeshTopology;

/// <summary>Topology neighbor lists (same idea as Chromodoris VertexSmooth: one array per topology vertex).</summary>
public static class MeshTopologyNeighbors
{
    /// <summary>One pass over topology edges (fast). Prefer over <see cref="FromMesh"/> which calls Rhino once per vertex.</summary>
    public static int[][] NeighborsFromEdges(Mesh mesh)
    {
        int n = mesh.TopologyVertices.Count;
        var lists = new List<int>[n];
        for (int i = 0; i < n; i++)
            lists[i] = new List<int>(8);

        int edgeCount = mesh.TopologyEdges.Count;
        for (int ei = 0; ei < edgeCount; ei++)
        {
            IndexPair ends = mesh.TopologyEdges.GetTopologyVertices(ei);
            int a = ends.I;
            int b = ends.J;
            lists[a].Add(b);
            lists[b].Add(a);
        }

        var neighbors = new int[n][];
        for (int i = 0; i < n; i++)
        {
            List<int> L = lists[i];
            if (L.Count == 0)
            {
                neighbors[i] = Array.Empty<int>();
                continue;
            }

            L.Sort();
            int w = 0;
            for (int k = 0; k < L.Count; k++)
            {
                if (k == 0 || L[k] != L[k - 1])
                    L[w++] = L[k];
            }

            if (w < L.Count)
                L.RemoveRange(w, L.Count - w);
            neighbors[i] = L.ToArray();
        }

        return neighbors;
    }

    /// <summary>Calls <see cref="MeshTopologyVertexList.ConnectedTopologyVertices"/> per vertex — can be very slow on large meshes.</summary>
    public static int[][] FromMesh(Mesh mesh)
    {
        int n = mesh.TopologyVertices.Count;
        var neighbors = new int[n][];
        for (int i = 0; i < n; i++)
            neighbors[i] = mesh.TopologyVertices.ConnectedTopologyVertices(i);
        return neighbors;
    }

    /// <summary>Packs jagged neighbor lists into CSR for Metal kernels.</summary>
    public static void ToCsr(int[][] neighbors, out int[] adjFlat, out int[] rowOffsets)
    {
        int n = neighbors.Length;
        int nnz = 0;
        for (int i = 0; i < n; i++)
            nnz += neighbors[i].Length;

        adjFlat = new int[nnz];
        rowOffsets = new int[n + 1];
        int o = 0;
        for (int i = 0; i < n; i++)
        {
            rowOffsets[i] = o;
            int[] row = neighbors[i];
            for (int k = 0; k < row.Length; k++)
                adjFlat[o++] = row[k];
        }

        rowOffsets[n] = o;
    }

    public static void TopologyPositionsToArray(Mesh mesh, Point3f[] positions)
    {
        int n = mesh.TopologyVertices.Count;
        for (int i = 0; i < n; i++)
            positions[i] = mesh.TopologyVertices[i];
    }

    public static void ApplyTopologyPositions(Mesh mesh, Point3f[] topoPositions)
    {
        var tv = mesh.TopologyVertices;
        int n = tv.Count;
        for (int ti = 0; ti < n; ti++)
        {
            Point3f p = topoPositions[ti];
            int[] mv = tv.MeshVertexIndices(ti);
            for (int k = 0; k < mv.Length; k++)
                mesh.Vertices.SetVertex(mv[k], p);
        }

        mesh.Normals.ComputeNormals();
    }

    /// <summary>Chromodoris-style output: new mesh, bulk vertices, copy faces — avoids <see cref="Mesh.DuplicateMesh"/> and per-vertex SetVertex.</summary>
    public static Mesh SmoothedMeshFromTopology(Mesh source, Point3f[] topoPositions)
    {
        int vc = source.Vertices.Count;
        var mVerts = new Point3f[vc];
        var tv = source.TopologyVertices;
        int nTopo = tv.Count;
        for (int ti = 0; ti < nTopo; ti++)
        {
            Point3f loc = topoPositions[ti];
            int[] mv = tv.MeshVertexIndices(ti);
            for (int k = 0; k < mv.Length; k++)
                mVerts[mv[k]] = loc;
        }

        Mesh newMesh = new Mesh();
        newMesh.Vertices.AddVertices(mVerts);
        newMesh.Faces.AddFaces(source.Faces);
        newMesh.Normals.ComputeNormals();
        return newMesh;
    }
}
