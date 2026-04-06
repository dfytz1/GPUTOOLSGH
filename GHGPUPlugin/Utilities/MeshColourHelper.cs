using System.Drawing;
using Rhino.Geometry;

namespace GHGPUPlugin.Utilities;

/// <summary>Per-mesh-vertex false-colouring from scalars (blue→red in HSL).</summary>
public static class MeshColourHelper
{
    /// <summary>Duplicates <paramref name="mesh"/> and assigns vertex colours from <paramref name="values"/> (length = mesh.Vertices.Count).</summary>
    public static Mesh ColourByScalar(Mesh mesh, double[] values, bool normaliseMinMax)
    {
        int vc = mesh.Vertices.Count;
        var colours = new Color[vc];
        if (values.Length < vc)
            throw new ArgumentException("values length must be at least mesh.Vertices.Count.", nameof(values));

        double lo = double.MaxValue, hi = double.MinValue;
        for (int i = 0; i < vc; i++)
        {
            double v = values[i];
            if (double.IsNaN(v) || double.IsInfinity(v))
                continue;
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }

        if (hi <= lo + 1e-30)
        {
            Color mid = HslToRgb(120, 1.0, 0.5);
            for (int i = 0; i < vc; i++)
                colours[i] = mid;
        }
        else
        {
            for (int i = 0; i < vc; i++)
            {
                double v = values[i];
                if (double.IsNaN(v) || double.IsInfinity(v))
                {
                    colours[i] = Color.Gray;
                    continue;
                }

                double t = normaliseMinMax ? (v - lo) / (hi - lo) : v;
                if (normaliseMinMax)
                    t = Math.Clamp(t, 0, 1);
                else
                    t = Math.Clamp(t, 0, 1);

                double hue = 240.0 * (1.0 - t);
                colours[i] = HslToRgb(hue, 1.0, 0.5);
            }
        }

        Mesh m = mesh.DuplicateMesh();
        m.VertexColors.SetColors(colours);
        return m;
    }

    /// <summary>Hue in [0,360), S and L in [0,1].</summary>
    public static Color HslToRgb(double h, double s, double l)
    {
        h = ((h % 360.0) + 360.0) % 360.0;
        double c = (1.0 - Math.Abs(2.0 * l - 1.0)) * s;
        double x = c * (1.0 - Math.Abs((h / 60.0) % 2.0 - 1.0));
        double m = l - c / 2.0;
        double rp = 0, gp = 0, bp = 0;
        if (h < 60) { rp = c; gp = x; }
        else if (h < 120) { rp = x; gp = c; }
        else if (h < 180) { gp = c; bp = x; }
        else if (h < 240) { gp = x; bp = c; }
        else if (h < 300) { rp = x; bp = c; }
        else { rp = c; bp = x; }

        int R = (int)Math.Round((rp + m) * 255.0);
        int G = (int)Math.Round((gp + m) * 255.0);
        int B = (int)Math.Round((bp + m) * 255.0);
        R = Math.Clamp(R, 0, 255);
        G = Math.Clamp(G, 0, 255);
        B = Math.Clamp(B, 0, 255);
        return Color.FromArgb(R, G, B);
    }
}
