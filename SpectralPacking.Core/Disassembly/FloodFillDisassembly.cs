namespace SpectralPacking.Core.Disassembly;

public static class FloodFillDisassembly
{
    /// <summary>
    /// If the blocking graph is acyclic, returns a removal order (objects that nobody blocks first).
    /// </summary>
    public static bool TryTopologicalRemovalOrder(List<int>[] adjBlocks, int objectCount, out List<int> order)
    {
        order = new List<int>(objectCount);
        var indegree = new int[objectCount];
        for (int i = 0; i < objectCount; i++)
        {
            foreach (int v in adjBlocks[i])
                indegree[v]++;
        }

        var q = new Queue<int>();
        for (int i = 0; i < objectCount; i++)
        {
            if (indegree[i] == 0)
                q.Enqueue(i);
        }

        while (q.Count > 0)
        {
            int u = q.Dequeue();
            order.Add(u);
            foreach (int v in adjBlocks[u])
            {
                indegree[v]--;
                if (indegree[v] == 0)
                    q.Enqueue(v);
            }
        }

        return order.Count == objectCount;
    }
}
