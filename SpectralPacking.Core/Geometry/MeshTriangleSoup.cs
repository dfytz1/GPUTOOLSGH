namespace SpectralPacking.Core.Geometry;

/// <summary>Triangle mesh as SoA vertex coordinates and CCW triangle indices (3 per face).</summary>
public sealed class MeshTriangleSoup
{
    public MeshTriangleSoup(double[] vx, double[] vy, double[] vz, int[] triangleIndices)
    {
        Vx = vx;
        Vy = vy;
        Vz = vz;
        TriangleIndices = triangleIndices;
    }

    public double[] Vx { get; }
    public double[] Vy { get; }
    public double[] Vz { get; }
    public int[] TriangleIndices { get; }

    public int VertexCount => Vx.Length;
    public int TriangleCount => TriangleIndices.Length / 3;

    public AxisAlignedBox BoundingBox => AxisAlignedBox.FromPoints(Vx, Vy, Vz);

    public (double cx, double cy, double cz) Centroid()
    {
        int n = VertexCount;
        if (n == 0)
            return (0, 0, 0);
        double sx = 0, sy = 0, sz = 0;
        for (int i = 0; i < n; i++)
        {
            sx += Vx[i];
            sy += Vy[i];
            sz += Vz[i];
        }

        return (sx / n, sy / n, sz / n);
    }

    /// <summary>Rotate about mesh centroid: v' = R*v + (c - R*c).</summary>
    public MeshTriangleSoup RotatedAboutCentroid(float m00, float m01, float m02, float m10, float m11, float m12, float m20, float m21, float m22)
    {
        var (cx, cy, cz) = Centroid();
        double rx = m00 * cx + m01 * cy + m02 * cz;
        double ry = m10 * cx + m11 * cy + m12 * cz;
        double rz = m20 * cx + m21 * cy + m22 * cz;
        double tx = cx - rx;
        double ty = cy - ry;
        double tz = cz - rz;
        return Transformed(m00, m01, m02, m10, m11, m12, m20, m21, m22, tx, ty, tz);
    }

    public MeshTriangleSoup Transformed(
        double m00, double m01, double m02, double m10, double m11, double m12, double m20, double m21, double m22,
        double tx, double ty, double tz)
    {
        int n = VertexCount;
        var ox = new double[n];
        var oy = new double[n];
        var oz = new double[n];
        for (int i = 0; i < n; i++)
        {
            double x = Vx[i], y = Vy[i], z = Vz[i];
            ox[i] = m00 * x + m01 * y + m02 * z + tx;
            oy[i] = m10 * x + m11 * y + m12 * z + ty;
            oz[i] = m20 * x + m21 * y + m22 * z + tz;
        }

        var tri = new int[TriangleIndices.Length];
        Array.Copy(TriangleIndices, tri, tri.Length);
        return new MeshTriangleSoup(ox, oy, oz, tri);
    }
}
