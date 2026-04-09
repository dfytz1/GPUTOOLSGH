using SpectralPacking.Core.Geometry;

namespace SpectralPacking.Core.Voxelization;

/// <summary>Conservative occupancy: a voxel is solid if its axis-aligned cell intersects any mesh triangle.</summary>
public static class ConservativeVoxelizer
{
    public static VoxelGrid VoxelizeMesh(
        MeshTriangleSoup mesh,
        AxisAlignedBox trayWorld,
        double voxelSize,
        bool fillSixWalls)
    {
        if (voxelSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(voxelSize));

        double dx = voxelSize;
        int nx = Math.Max(1, (int)Math.Ceiling((trayWorld.MaxX - trayWorld.MinX) / dx));
        int ny = Math.Max(1, (int)Math.Ceiling((trayWorld.MaxY - trayWorld.MinY) / dx));
        int nz = Math.Max(1, (int)Math.Ceiling((trayWorld.MaxZ - trayWorld.MinZ) / dx));

        var grid = VoxelGrid.CreateZero(nx, ny, nz);
        if (fillSixWalls)
            MarkTrayWalls(grid, 1f);

        double ox = trayWorld.MinX;
        double oy = trayWorld.MinY;
        double oz = trayWorld.MinZ;

        int triCount = mesh.TriangleCount;
        for (int t = 0; t < triCount; t++)
        {
            int i0 = mesh.TriangleIndices[t * 3];
            int i1 = mesh.TriangleIndices[t * 3 + 1];
            int i2 = mesh.TriangleIndices[t * 3 + 2];
            double ax = mesh.Vx[i0], ay = mesh.Vy[i0], az = mesh.Vz[i0];
            double bx = mesh.Vx[i1], by = mesh.Vy[i1], bz = mesh.Vz[i1];
            double cx = mesh.Vx[i2], cy = mesh.Vy[i2], cz = mesh.Vz[i2];

            double tMinX = Math.Min(ax, Math.Min(bx, cx));
            double tMaxX = Math.Max(ax, Math.Max(bx, cx));
            double tMinY = Math.Min(ay, Math.Min(by, cy));
            double tMaxY = Math.Max(ay, Math.Max(by, cy));
            double tMinZ = Math.Min(az, Math.Min(bz, cz));
            double tMaxZ = Math.Max(az, Math.Max(bz, cz));

            int i0x = (int)Math.Floor((tMinX - ox) / dx);
            int i1x = (int)Math.Floor((tMaxX - ox) / dx);
            int j0y = (int)Math.Floor((tMinY - oy) / dx);
            int j1y = (int)Math.Floor((tMaxY - oy) / dx);
            int k0z = (int)Math.Floor((tMinZ - oz) / dx);
            int k1z = (int)Math.Floor((tMaxZ - oz) / dx);

            i0x = Math.Clamp(i0x, 0, nx - 1);
            i1x = Math.Clamp(i1x, 0, nx - 1);
            j0y = Math.Clamp(j0y, 0, ny - 1);
            j1y = Math.Clamp(j1y, 0, ny - 1);
            k0z = Math.Clamp(k0z, 0, nz - 1);
            k1z = Math.Clamp(k1z, 0, nz - 1);

            for (int iz = k0z; iz <= k1z; iz++)
            {
                double z0 = oz + iz * dx;
                double z1 = z0 + dx;
                for (int iy = j0y; iy <= j1y; iy++)
                {
                    double y0 = oy + iy * dx;
                    double y1 = y0 + dx;
                    for (int ix = i0x; ix <= i1x; ix++)
                    {
                        double x0 = ox + ix * dx;
                        double x1 = x0 + dx;
                        if (TriBoxOverlapCell(x0, y0, z0, x1, y1, z1, ax, ay, az, bx, by, bz, cx, cy, cz))
                            grid[ix, iy, iz] = 1f;
                    }
                }
            }
        }

        return grid;
    }

    public static void MarkTrayWalls(VoxelGrid grid, float value)
    {
        int nx = grid.Width, ny = grid.Height, nz = grid.Depth;
        for (int y = 0; y < ny; y++)
        for (int z = 0; z < nz; z++)
        {
            grid[0, y, z] = value;
            grid[nx - 1, y, z] = value;
        }

        for (int x = 0; x < nx; x++)
        for (int z = 0; z < nz; z++)
        {
            grid[x, 0, z] = value;
            grid[x, ny - 1, z] = value;
        }

        for (int x = 0; x < nx; x++)
        for (int y = 0; y < ny; y++)
        {
            grid[x, y, 0] = value;
            grid[x, y, nz - 1] = value;
        }
    }

    /// <summary>Rasterizes mesh into the same lattice as <paramref name="container"/> (walls unchanged).</summary>
    public static void StampMeshIntoGrid(VoxelGrid container, MeshTriangleSoup meshWorld, AxisAlignedBox trayWorld, double voxelSize)
    {
        var part = VoxelizeMesh(meshWorld, trayWorld, voxelSize, fillSixWalls: false);
        for (int i = 0; i < container.Data.Length; i++)
        {
            if (part.Data[i] > 0.5f)
                container.Data[i] = 1f;
        }
    }

    // Tomas Akenine-Möller — triBoxOverlap (Public Domain), cell [min,max], triangle in world space.
    private static bool TriBoxOverlapCell(
        double boxMinX, double boxMinY, double boxMinZ,
        double boxMaxX, double boxMaxY, double boxMaxZ,
        double ax, double ay, double az,
        double bx, double by, double bz,
        double cx, double cy, double cz)
    {
        double cxB = (boxMinX + boxMaxX) * 0.5;
        double cyB = (boxMinY + boxMaxY) * 0.5;
        double czB = (boxMinZ + boxMaxZ) * 0.5;
        double hx = (boxMaxX - boxMinX) * 0.5;
        double hy = (boxMaxY - boxMinY) * 0.5;
        double hz = (boxMaxZ - boxMinZ) * 0.5;

        double v0x = ax - cxB, v0y = ay - cyB, v0z = az - czB;
        double v1x = bx - cxB, v1y = by - cyB, v1z = bz - czB;
        double v2x = cx - cxB, v2y = cy - cyB, v2z = cz - czB;

        double minV = Math.Min(v0x, Math.Min(v1x, v2x));
        double maxV = Math.Max(v0x, Math.Max(v1x, v2x));
        if (minV > hx || maxV < -hx)
            return false;
        minV = Math.Min(v0y, Math.Min(v1y, v2y));
        maxV = Math.Max(v0y, Math.Max(v1y, v2y));
        if (minV > hy || maxV < -hy)
            return false;
        minV = Math.Min(v0z, Math.Min(v1z, v2z));
        maxV = Math.Max(v0z, Math.Max(v1z, v2z));
        if (minV > hz || maxV < -hz)
            return false;

        double e1x = v1x - v0x, e1y = v1y - v0y, e1z = v1z - v0z;
        double e2x = v2x - v1x, e2y = v2y - v1y, e2z = v2z - v1z;
        double e3x = v0x - v2x, e3y = v0y - v2y, e3z = v0z - v2z;

        if (!AxisTestX01(v0y, v0z, v1y, v1z, v2y, v2z, hy, hz, e1z, e1y))
            return false;
        if (!AxisTestX01(v0y, v0z, v1y, v1z, v2y, v2z, hy, hz, e2z, e2y))
            return false;
        if (!AxisTestX01(v0y, v0z, v1y, v1z, v2y, v2z, hy, hz, e3z, e3y))
            return false;

        if (!AxisTestY02(v0x, v0z, v1x, v1z, v2x, v2z, hx, hz, e1z, e1x))
            return false;
        if (!AxisTestY02(v0x, v0z, v1x, v1z, v2x, v2z, hx, hz, e2z, e2x))
            return false;
        if (!AxisTestY02(v0x, v0z, v1x, v1z, v2x, v2z, hx, hz, e3z, e3x))
            return false;

        if (!AxisTestZ12(v0x, v0y, v1x, v1y, v2x, v2y, hx, hy, e1y, e1x))
            return false;
        if (!AxisTestZ12(v0x, v0y, v1x, v1y, v2x, v2y, hx, hy, e2y, e2x))
            return false;
        if (!AxisTestZ12(v0x, v0y, v1x, v1y, v2x, v2y, hx, hy, e3y, e3x))
            return false;

        double nx = e1y * e2z - e1z * e2y;
        double ny = e1z * e2x - e1x * e2z;
        double nz = e1x * e2y - e1y * e2x;
        return PlaneBoxOverlap(nx, ny, nz, v0x, v0y, v0z, hx, hy, hz);
    }

    private static bool PlaneBoxOverlap(double nx, double ny, double nz, double vx, double vy, double vz, double hx, double hy, double hz)
    {
        double nxAbs = Math.Abs(nx);
        double nyAbs = Math.Abs(ny);
        double nzAbs = Math.Abs(nz);
        double maxDist = hx * nxAbs + hy * nyAbs + hz * nzAbs;
        double d = nx * vx + ny * vy + nz * vz;
        return Math.Abs(d) <= maxDist;
    }

    private static bool AxisTestX01(
        double v0y, double v0z, double v1y, double v1z, double v2y, double v2z,
        double halfY, double halfZ, double a, double b)
    {
        double p0 = a * v0y - b * v0z;
        double p1 = a * v1y - b * v1z;
        double p2 = a * v2y - b * v2z;
        double minP = Math.Min(p0, Math.Min(p1, p2));
        double maxP = Math.Max(p0, Math.Max(p1, p2));
        double rad = halfY * Math.Abs(a) + halfZ * Math.Abs(b);
        return !(minP > rad || maxP < -rad);
    }

    private static bool AxisTestY02(
        double v0x, double v0z, double v1x, double v1z, double v2x, double v2z,
        double halfX, double halfZ, double a, double b)
    {
        double p0 = -a * v0x + b * v0z;
        double p1 = -a * v1x + b * v1z;
        double p2 = -a * v2x + b * v2z;
        double minP = Math.Min(p0, Math.Min(p1, p2));
        double maxP = Math.Max(p0, Math.Max(p1, p2));
        double rad = halfX * Math.Abs(a) + halfZ * Math.Abs(b);
        return !(minP > rad || maxP < -rad);
    }

    private static bool AxisTestZ12(
        double v0x, double v0y, double v1x, double v1y, double v2x, double v2y,
        double halfX, double halfY, double a, double b)
    {
        double p0 = a * v0x - b * v0y;
        double p1 = a * v1x - b * v1y;
        double p2 = a * v2x - b * v2y;
        double minP = Math.Min(p0, Math.Min(p1, p2));
        double maxP = Math.Max(p0, Math.Max(p1, p2));
        double rad = halfX * Math.Abs(a) + halfY * Math.Abs(b);
        return !(minP > rad || maxP < -rad);
    }
}
