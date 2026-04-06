using GHGPUPlugin.MeshTopology;
using GHGPUPlugin.NativeInterop;
using Rhino.Geometry;

namespace GHGPUPlugin.Algorithms;

/// <summary>Dijkstra shortest path on mesh topology; edge weights from 3D Euclidean length (GPU fills CSR edge list when requested).</summary>
public static class MeshShortestPath
{
    public static bool TryDijkstraTopology(
        Mesh mesh,
        int startTopo,
        int endTopo,
        bool useGpu,
        out List<Point3d> path,
        out double length,
        out string? error)
    {
        path = new List<Point3d>();
        length = 0;
        error = null;

        int n = mesh.TopologyVertices.Count;
        if (startTopo < 0 || startTopo >= n || endTopo < 0 || endTopo >= n)
        {
            error = "Start or end topology index is out of range.";
            return false;
        }

        if (startTopo == endTopo)
        {
            path.Add(mesh.TopologyVertices[startTopo]);
            length = 0;
            return true;
        }

        int[][] neighbors = MeshTopologyNeighbors.NeighborsFromEdges(mesh);
        MeshTopologyNeighbors.ToCsr(neighbors, out int[] adjFlat, out int[] rowOffsets);
        int nnz = rowOffsets[n];

        if (nnz == 0)
        {
            error = "Mesh has no topology edges.";
            return false;
        }

        var vx = new float[n];
        var vy = new float[n];
        var vz = new float[n];
        for (int i = 0; i < n; i++)
        {
            Point3d p = mesh.TopologyVertices[i];
            vx[i] = (float)p.X;
            vy[i] = (float)p.Y;
            vz[i] = (float)p.Z;
        }

        var edgeU = new int[nnz];
        var edgeV = new int[nnz];
        var edgeW = new float[nnz];

        bool weightsFromGpu = false;
        if (useGpu && NativeLoader.IsMetalAvailable && MetalSharedContext.TryGetContext(out IntPtr ctx))
        {
            int code = MetalBridge.BuildWeightedEdgesCsr(
                ctx,
                vx,
                vy,
                vz,
                n,
                rowOffsets,
                adjFlat,
                rowOffsets,
                nnz,
                edgeU,
                edgeV,
                edgeW);
            if (code == 0)
                weightsFromGpu = true;
        }

        if (!weightsFromGpu)
        {
            int e = 0;
            for (int v = 0; v < n; v++)
            {
                Point3d pv = mesh.TopologyVertices[v];
                for (int k = rowOffsets[v]; k < rowOffsets[v + 1]; k++)
                {
                    int u = adjFlat[k];
                    Point3d pu = mesh.TopologyVertices[u];
                    edgeU[e] = v;
                    edgeV[e] = u;
                    edgeW[e] = (float)pv.DistanceTo(pu);
                    e++;
                }
            }
        }

        var adj = new List<(int v, float w)>[n];
        for (int i = 0; i < n; i++)
            adj[i] = new List<(int, float)>();
        for (int e = 0; e < nnz; e++)
            adj[edgeU[e]].Add((edgeV[e], edgeW[e]));

        var dist = new double[n];
        var prev = new int[n];
        for (int i = 0; i < n; i++)
        {
            dist[i] = double.PositiveInfinity;
            prev[i] = -1;
        }

        dist[startTopo] = 0;
        var pq = new PriorityQueue<int, double>();
        pq.Enqueue(startTopo, 0);

        while (pq.TryDequeue(out int u, out double du))
        {
            if (du > dist[u])
                continue;
            if (u == endTopo)
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

        if (double.IsPositiveInfinity(dist[endTopo]))
        {
            error = "No path between start and end on mesh topology.";
            return false;
        }

        length = dist[endTopo];
        var order = new List<int>();
        for (int at = endTopo; ; at = prev[at])
        {
            order.Add(at);
            if (at == startTopo)
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
            path.Add(mesh.TopologyVertices[vi]);
        return true;
    }
}
