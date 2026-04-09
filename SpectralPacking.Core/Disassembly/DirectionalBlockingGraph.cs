using SpectralPacking.Core.Geometry;

namespace SpectralPacking.Core.Disassembly;

/// <summary>
/// Axis-aligned directional blocking: edge i→j means object i blocks j for that direction sample (union over 6 axes).
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

        var sumX = new double[objectCount];
        var sumY = new double[objectCount];
        var sumZ = new double[objectCount];
        var cnt = new int[objectCount];
        double dx = voxelSize;

        for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
        for (int x = 0; x < nx; x++)
        {
            int oi = owner[Index(x, y, z, nx, ny)];
            if (oi < 0)
                continue;
            double wx = trayWorld.MinX + (x + 0.5) * dx;
            double wy = trayWorld.MinY + (y + 0.5) * dx;
            double wz = trayWorld.MinZ + (z + 0.5) * dx;
            sumX[oi] += wx;
            sumY[oi] += wy;
            sumZ[oi] += wz;
            cnt[oi]++;
        }

        var cx = new double[objectCount];
        var cy = new double[objectCount];
        var cz = new double[objectCount];
        for (int i = 0; i < objectCount; i++)
        {
            if (cnt[i] > 0)
            {
                cx[i] = sumX[i] / cnt[i];
                cy[i] = sumY[i] / cnt[i];
                cz[i] = sumZ[i] / cnt[i];
            }
        }

        ReadOnlySpan<(int dx, int dy, int dz)> dirs = stackalloc (int, int, int)[]
        {
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)
        };

        for (int j = 0; j < objectCount; j++)
        {
            if (cnt[j] == 0)
                continue;
            for (int di = 0; di < 6; di++)
            {
                int hit = RayFirstObject(owner, nx, ny, nz, cx[j], cy[j], cz[j], dirs[di], j, trayWorld, voxelSize);
                if (hit >= 0 && hit != j && !adj[hit].Contains(j))
                    adj[hit].Add(j);
            }
        }

        return adj;
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
