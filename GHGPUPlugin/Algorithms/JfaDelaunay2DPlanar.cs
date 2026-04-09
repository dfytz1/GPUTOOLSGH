using Rhino.Geometry;

namespace GHGPUPlugin.Algorithms;

/// <summary>Planar UV normalization for <c>mb_jfa_delaunay_2d</c> (same mapping as <c>GH_JFADelaunay2D</c>).</summary>
public static class JfaDelaunay2DPlanar
{
    /// <summary>Maps UV bounds to [0.05, 0.95]² for JFA. Returns false if degenerate.</summary>
    public static bool TryJfaNormalizedCoords(IReadOnlyList<Vector2d> uv, out float[] px, out float[] py)
    {
        px = Array.Empty<float>();
        py = Array.Empty<float>();
        int n = uv.Count;
        if (n < 1)
            return false;

        double minU = double.MaxValue, maxU = double.MinValue;
        double minV = double.MaxValue, maxV = double.MinValue;
        for (int i = 0; i < n; i++)
        {
            var p = uv[i];
            if (p.X < minU)
                minU = p.X;
            if (p.X > maxU)
                maxU = p.X;
            if (p.Y < minV)
                minV = p.Y;
            if (p.Y > maxV)
                maxV = p.Y;
        }

        double rangeU = maxU - minU;
        double rangeV = maxV - minV;
        double range = Math.Max(rangeU, rangeV);
        if (range < 1e-10)
            return false;

        px = new float[n];
        py = new float[n];
        for (int i = 0; i < n; i++)
        {
            px[i] = (float)(0.05 + 0.9 * (uv[i].X - minU) / range);
            py[i] = (float)(0.05 + 0.9 * (uv[i].Y - minV) / range);
        }

        return true;
    }
}
