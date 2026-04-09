using SpectralPacking.Core.Native;
using SpectralPacking.Core.Voxelization;

namespace SpectralPacking.Core.Metrics;

/// <summary>
/// Discrete distance in empty voxels from the solid region (multi-source BFS through empty cells).
/// Solid voxels get 0; empty voxels get (steps × voxelSize) to the nearest solid cell.
/// </summary>
public static class DistanceField
{
    public static void BuildFromSolid(VoxelGrid solidBinary, VoxelGrid phiOut, float voxelSize)
    {
        int nx = solidBinary.Width, ny = solidBinary.Height, nz = solidBinary.Depth;
        if (phiOut.Width != nx || phiOut.Height != ny || phiOut.Depth != nz)
            throw new ArgumentException("phiOut dimensions must match solid grid.");

        int n = nx * ny * nz;
        var dist = new int[n];
        const int Inf = int.MaxValue / 4;
        for (int i = 0; i < n; i++)
            dist[i] = Inf;

        var q = new Queue<int>(n / 8);

        for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
        for (int x = 0; x < nx; x++)
        {
            int idx = solidBinary.Index(x, y, z);
            if (solidBinary.Data[idx] > 0.5f)
            {
                dist[idx] = 0;
                q.Enqueue(idx);
            }
        }

        ReadOnlySpan<(int dx, int dy, int dz)> nb = stackalloc (int, int, int)[]
        {
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)
        };

        while (q.Count > 0)
        {
            int cur = q.Dequeue();
            int dv = dist[cur];
            int cz = cur / (nx * ny);
            int rem = cur - cz * nx * ny;
            int cy = rem / nx;
            int cx = rem - cy * nx;

            for (int k = 0; k < 6; k++)
            {
                int nx1 = cx + nb[k].dx;
                int ny1 = cy + nb[k].dy;
                int nz1 = cz + nb[k].dz;
                if ((uint)nx1 >= (uint)nx || (uint)ny1 >= (uint)ny || (uint)nz1 >= (uint)nz)
                    continue;
                int ni = solidBinary.Index(nx1, ny1, nz1);
                if (solidBinary.Data[ni] > 0.5f)
                    continue;
                int nd = dv + 1;
                if (nd < dist[ni])
                {
                    dist[ni] = nd;
                    q.Enqueue(ni);
                }
            }
        }

        float scale = voxelSize;
        for (int i = 0; i < n; i++)
        {
            if (solidBinary.Data[i] > 0.5f)
                phiOut.Data[i] = 0f;
            else
                phiOut.Data[i] = dist[i] >= Inf ? 1e6f : dist[i] * scale;
        }
    }

    public static void BuildFromSolidGpu(
        IntPtr metalCtx,
        VoxelGrid solidBinary,
        VoxelGrid phiOut,
        float voxelSize)
    {
        if (MetalSpectralInterop.TryDistanceFieldBfs(metalCtx, solidBinary, phiOut, voxelSize) == 0)
            return;
        BuildFromSolid(solidBinary, phiOut, voxelSize);
    }
}
