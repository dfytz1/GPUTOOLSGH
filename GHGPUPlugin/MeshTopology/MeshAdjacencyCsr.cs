using Rhino;
using Rhino.Geometry;

namespace GHGPUPlugin.MeshTopology;

/// <summary>Topology-vertex adjacency in CSR form (undirected 1-ring from mesh edges).</summary>
public static class MeshAdjacencyCsr
{
    public static void Build(Mesh mesh, out int[] adjFlat, out int[] rowOffsets, out int topologyVertexCount)
    {
        topologyVertexCount = mesh.TopologyVertices.Count;
        var neighbors = new List<int>[topologyVertexCount];
        for (int i = 0; i < topologyVertexCount; i++)
            neighbors[i] = new List<int>();

        int edgeCount = mesh.TopologyEdges.Count;
        for (int ei = 0; ei < edgeCount; ei++)
        {
            IndexPair ends = mesh.TopologyEdges.GetTopologyVertices(ei);
            int a = ends.I;
            int b = ends.J;
            neighbors[a].Add(b);
            neighbors[b].Add(a);
        }

        for (int i = 0; i < topologyVertexCount; i++)
        {
            if (neighbors[i].Count == 0)
                continue;
            neighbors[i].Sort();
            int w = 0;
            for (int k = 0; k < neighbors[i].Count; k++)
            {
                if (k == 0 || neighbors[i][k] != neighbors[i][k - 1])
                    neighbors[i][w++] = neighbors[i][k];
            }

            if (w < neighbors[i].Count)
                neighbors[i].RemoveRange(w, neighbors[i].Count - w);
        }

        int nnz = 0;
        for (int i = 0; i < topologyVertexCount; i++)
            nnz += neighbors[i].Count;

        adjFlat = new int[nnz];
        rowOffsets = new int[topologyVertexCount + 1];
        int off = 0;
        for (int i = 0; i < topologyVertexCount; i++)
        {
            rowOffsets[i] = off;
            foreach (int j in neighbors[i])
                adjFlat[off++] = j;
        }

        rowOffsets[topologyVertexCount] = off;
    }

    public static void TopologyPositionsToSoa(Mesh mesh, float[] posX, float[] posY, float[] posZ)
    {
        int n = mesh.TopologyVertices.Count;
        for (int i = 0; i < n; i++)
        {
            Point3f p = mesh.TopologyVertices[i];
            posX[i] = p.X;
            posY[i] = p.Y;
            posZ[i] = p.Z;
        }
    }

    public static void ApplyTopologyPositions(Mesh mesh, float[] posX, float[] posY, float[] posZ)
    {
        var tv = mesh.TopologyVertices;
        int n = tv.Count;
        for (int ti = 0; ti < n; ti++)
        {
            var p = new Point3f(posX[ti], posY[ti], posZ[ti]);
            int[] mv = tv.MeshVertexIndices(ti);
            for (int k = 0; k < mv.Length; k++)
                mesh.Vertices.SetVertex(mv[k], p);
        }

        mesh.Normals.ComputeNormals();
    }
}
