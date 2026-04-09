using System.Numerics;

namespace SpectralPacking.Core.Placement;

public enum OrientationSamplingMode
{
    UniformEuler,
    Icosphere
}

public static class OrientationSampler
{
    /// <summary>Right-handed rotation matrices (rows map object frame to world), Z then Y then X Euler in degrees.</summary>
    public static List<Matrix4x4> Sample(int targetCount, OrientationSamplingMode mode)
    {
        return mode == OrientationSamplingMode.Icosphere
            ? SampleIcosphereBased(targetCount)
            : SampleUniformEuler(targetCount);
    }

    private static List<Matrix4x4> SampleUniformEuler(int targetCount)
    {
        int n = Math.Max(2, (int)Math.Round(Math.Pow(Math.Max(8, targetCount), 1.0 / 3.0)));
        double step = 360.0 / n;
        var list = new List<Matrix4x4>(n * n * n);
        for (int iz = 0; iz < n; iz++)
        for (int iy = 0; iy < n; iy++)
        for (int ix = 0; ix < n; ix++)
        {
            float rz = (float)(iz * step * (Math.PI / 180.0));
            float ry = (float)(iy * step * (Math.PI / 180.0));
            float rx = (float)(ix * step * (Math.PI / 180.0));
            list.Add(MatrixFromEulerZYX(rz, ry, rx));
            if (list.Count >= targetCount)
                return list;
        }

        return list;
    }

    private static List<Matrix4x4> SampleIcosphereBased(int targetCount)
    {
        // Subdivision-0 icosahedron vertices (12), then subdivide once if needed.
        var verts = IcosahedronVertices();
        if (targetCount > 24)
        {
            var more = new List<Vector3>();
            foreach (var a in verts)
            foreach (var b in verts)
            {
                if (Vector3.Distance(a, b) < 1e-4f)
                    continue;
                var m = Vector3.Normalize((a + b) * 0.5f);
                if (!more.Any(p => Vector3.Distance(p, m) < 0.08f))
                    more.Add(m);
            }

            verts = verts.Concat(more).ToList();
        }

        var mats = new List<Matrix4x4>(targetCount);
        int vi = 0;
        while (mats.Count < targetCount && vi < verts.Count)
        {
            mats.Add(MatrixFromForwardUp(verts[vi], Vector3.UnitZ));
            vi++;
        }

        while (mats.Count < targetCount)
            mats.AddRange(SampleUniformEuler(targetCount - mats.Count));
        return mats;
    }

    private static List<Vector3> IcosahedronVertices()
    {
        float t = (1f + MathF.Sqrt(5f)) * 0.5f;
        var v = new List<Vector3>
        {
            Vector3.Normalize(new Vector3(-1, t, 0)),
            Vector3.Normalize(new Vector3(1, t, 0)),
            Vector3.Normalize(new Vector3(-1, -t, 0)),
            Vector3.Normalize(new Vector3(1, -t, 0)),
            Vector3.Normalize(new Vector3(0, -1, t)),
            Vector3.Normalize(new Vector3(0, 1, t)),
            Vector3.Normalize(new Vector3(0, -1, -t)),
            Vector3.Normalize(new Vector3(0, 1, -t)),
            Vector3.Normalize(new Vector3(t, 0, -1)),
            Vector3.Normalize(new Vector3(t, 0, 1)),
            Vector3.Normalize(new Vector3(-t, 0, -1)),
            Vector3.Normalize(new Vector3(-t, 0, 1))
        };
        return v;
    }

    /// <summary>World +Z up: build orthonormal basis with local +Z aligned to <paramref name="forward"/>.</summary>
    public static Matrix4x4 MatrixFromForwardUp(Vector3 forward, Vector3 worldUp)
    {
        forward = Vector3.Normalize(forward);
        var right = Vector3.Normalize(Vector3.Cross(worldUp, forward));
        if (right.LengthSquared() < 1e-6f)
            right = Vector3.Normalize(Vector3.Cross(Vector3.UnitX, forward));
        var up = Vector3.Normalize(Vector3.Cross(forward, right));
        // Rows: local X,Y,Z expressed in world (column vectors are axes).
        return new Matrix4x4(
            right.X, right.Y, right.Z, 0,
            up.X, up.Y, up.Z, 0,
            forward.X, forward.Y, forward.Z, 0,
            0, 0, 0, 1);
    }

    public static Matrix4x4 MatrixFromEulerZYX(float rz, float ry, float rx)
    {
        var Rz = Matrix4x4.CreateRotationZ(rz);
        var Ry = Matrix4x4.CreateRotationY(ry);
        var Rx = Matrix4x4.CreateRotationX(rx);
        return Rz * Ry * Rx;
    }
}
