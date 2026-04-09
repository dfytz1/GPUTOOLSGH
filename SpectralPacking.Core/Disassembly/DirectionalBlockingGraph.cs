using SpectralPacking.Core.Geometry;

namespace SpectralPacking.Core.Disassembly;

/// <summary>
/// Axis-aligned directional blocking graph (paper DBG): edge i→j means i blocks j for that axis direction.
/// Rays are cast from <b>surface</b> voxels of j (voxels with a 6-neighbor not belonging to j), not only the centroid.
/// </summary>
public static class DirectionalBlockingGraph
{
    /// <summary>owner: -2 wall, -1 empty, else object index [0, objectCount).</summary>
    public static List<int>[] BuildAdjacency(
        int nx,
        int ny,
        int nz,
        ReadOnlySpan<int> owner,
        int objectCount,
        AxisAlignedBox trayWorld,
        double voxelSize)
    {
        var adj = new List<int>[objectCount];
        for (int i = 0; i < objectCount; i++)
            adj[i] = new List<int>();

        var cnt = new int[objectCount];
        for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
        for (int x = 0; x < nx; x++)
        {
            int oi = owner[Index(x, y, z, nx, ny)];
            if (oi >= 0)
                cnt[oi]++;
        }

        ReadOnlySpan<(int dx, int dy, int dz)> dirs = stackalloc (int, int, int)[]
        {
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)
        };

        double dx = voxelSize;

        for (int j = 0; j < objectCount; j++)
        {
            if (cnt[j] == 0)
                continue;

            for (int z = 0; z < nz; z++)
            for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++)
            {
                if (owner[Index(x, y, z, nx, ny)] != j)
                    continue;
                if (!IsSurfaceVoxel(owner, nx, ny, nz, x, y, z, j))
                    continue;

                double wx = trayWorld.MinX + (x + 0.5) * dx;
                double wy = trayWorld.MinY + (y + 0.5) * dx;
                double wz = trayWorld.MinZ + (z + 0.5) * dx;

                for (int di = 0; di < 6; di++)
                {
                    int hit = RayFirstObject(owner, nx, ny, nz, wx, wy, wz, dirs[di], j, trayWorld, voxelSize);
                    if (hit >= 0 && hit != j && !adj[hit].Contains(j))
                        adj[hit].Add(j);
                }
            }
        }

        return adj;
    }

    /// <summary>
    /// When SCC-based resolution is exhausted but cycles may remain, use ascending out-degree
    /// (least-blocking nodes first) as a deterministic fallback priority order.
    /// </summary>
    public static List<int> BuildFallbackRemovalOrderByAscendingOutDegree(List<int>[] adj, int objectCount)
    {
        var ids = new int[objectCount];
        for (int i = 0; i < objectCount; i++)
            ids[i] = i;
        Array.Sort(ids, (a, b) =>
        {
            int ca = adj[a].Count, cb = adj[b].Count;
            int c = ca.CompareTo(cb);
            return c != 0 ? c : a.CompareTo(b);
        });
        return ids.ToList();
    }

    private static bool IsSurfaceVoxel(ReadOnlySpan<int> owner, int nx, int ny, int nz, int x, int y, int z, int self)
    {
        ReadOnlySpan<(int dx, int dy, int dz)> nb = stackalloc (int, int, int)[]
        {
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)
        };

        for (int k = 0; k < 6; k++)
        {
            int nx2 = x + nb[k].dx;
            int ny2 = y + nb[k].dy;
            int nz2 = z + nb[k].dz;
            if ((uint)nx2 >= (uint)nx || (uint)ny2 >= (uint)ny || (uint)nz2 >= (uint)nz)
                return true;
            int o = owner[Index(nx2, ny2, nz2, nx, ny)];
            if (o != self)
                return true;
        }

        return false;
    }

    private static int RayFirstObject(
        ReadOnlySpan<int> owner,
        int nx,
        int ny,
        int nz,
        double wx,
        double wy,
        double wz,
        (int dx, int dy, int dz) dir,
        int self,
        AxisAlignedBox trayWorld,
        double voxelSize)
    {
        double dx = voxelSize;
        int ix = (int)Math.Floor((wx - trayWorld.MinX) / dx);
        int iy = (int)Math.Floor((wy - trayWorld.MinY) / dx);
        int iz = (int)Math.Floor((wz - trayWorld.MinZ) / dx);
        ix = Math.Clamp(ix, 0, nx - 1);
        iy = Math.Clamp(iy, 0, ny - 1);
        iz = Math.Clamp(iz, 0, nz - 1);

        int x = ix, y = iy, z = iz;
        while (x >= 0 && y >= 0 && z >= 0 && x < nx && y < ny && z < nz && owner[Index(x, y, z, nx, ny)] == self)
        {
            x += dir.dx;
            y += dir.dy;
            z += dir.dz;
        }

        while (x >= 0 && y >= 0 && z >= 0 && x < nx && y < ny && z < nz)
        {
            int o = owner[Index(x, y, z, nx, ny)];
            if (o == -2)
                return -1;
            if (o >= 0 && o != self)
                return o;
            x += dir.dx;
            y += dir.dy;
            z += dir.dz;
        }

        return -1;
    }

    private static int Index(int x, int y, int z, int nx, int ny) => x + y * nx + z * nx * ny;

    public static List<List<int>> FindSccsTarjan(int n, List<int>[] adj)
    {
        var index = new int[n];
        var low = new int[n];
        var onStack = new bool[n];
        var stack = new Stack<int>();
        var sccs = new List<List<int>>();
        int current = 0;

        void StrongConnect(int v)
        {
            index[v] = current;
            low[v] = current;
            current++;
            stack.Push(v);
            onStack[v] = true;

            foreach (int w in adj[v])
            {
                if ((uint)w >= (uint)n)
                    continue;
                if (index[w] < 0)
                {
                    StrongConnect(w);
                    low[v] = Math.Min(low[v], low[w]);
                }
                else if (onStack[w])
                    low[v] = Math.Min(low[v], index[w]);
            }

            if (low[v] == index[v])
            {
                var comp = new List<int>();
                while (true)
                {
                    int w = stack.Pop();
                    onStack[w] = false;
                    comp.Add(w);
                    if (w == v)
                        break;
                }

                sccs.Add(comp);
            }
        }

        for (int i = 0; i < n; i++)
            index[i] = -1;

        for (int i = 0; i < n; i++)
        {
            if (index[i] < 0)
                StrongConnect(i);
        }

        return sccs;
    }
}
