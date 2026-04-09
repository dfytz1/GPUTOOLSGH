namespace SpectralPacking.Core.Geometry;

public readonly struct AxisAlignedBox
{
    public AxisAlignedBox(double minX, double minY, double minZ, double maxX, double maxY, double maxZ)
    {
        MinX = minX;
        MinY = minY;
        MinZ = minZ;
        MaxX = maxX;
        MaxY = maxY;
        MaxZ = maxZ;
    }

    public double MinX { get; }
    public double MinY { get; }
    public double MinZ { get; }
    public double MaxX { get; }
    public double MaxY { get; }
    public double MaxZ { get; }

    public double Volume =>
        Math.Max(0, MaxX - MinX) * Math.Max(0, MaxY - MinY) * Math.Max(0, MaxZ - MinZ);

    public static AxisAlignedBox FromPoints(ReadOnlySpan<double> vx, ReadOnlySpan<double> vy, ReadOnlySpan<double> vz)
    {
        if (vx.Length == 0)
            return new AxisAlignedBox(0, 0, 0, 0, 0, 0);
        double minX = vx[0], maxX = vx[0];
        double minY = vy[0], maxY = vy[0];
        double minZ = vz[0], maxZ = vz[0];
        for (int i = 1; i < vx.Length; i++)
        {
            minX = Math.Min(minX, vx[i]);
            maxX = Math.Max(maxX, vx[i]);
            minY = Math.Min(minY, vy[i]);
            maxY = Math.Max(maxY, vy[i]);
            minZ = Math.Min(minZ, vz[i]);
            maxZ = Math.Max(maxZ, vz[i]);
        }

        return new AxisAlignedBox(minX, minY, minZ, maxX, maxY, maxZ);
    }

    public bool IntersectsTriangle(
        double ax, double ay, double az,
        double bx, double by, double bz,
        double cx, double cy, double cz)
    {
        double tMinX = Math.Min(ax, Math.Min(bx, cx));
        double tMaxX = Math.Max(ax, Math.Max(bx, cx));
        if (tMaxX < MinX || tMinX > MaxX)
            return false;
        double tMinY = Math.Min(ay, Math.Min(by, cy));
        double tMaxY = Math.Max(ay, Math.Max(by, cy));
        if (tMaxY < MinY || tMinY > MaxY)
            return false;
        double tMinZ = Math.Min(az, Math.Min(bz, cz));
        double tMaxZ = Math.Max(az, Math.Max(bz, cz));
        if (tMaxZ < MinZ || tMinZ > MaxZ)
            return false;
        return true;
    }
}
