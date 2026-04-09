using System.Drawing;
using System.Drawing.Drawing2D;

namespace GHGPUPlugin.Utilities;

/// <summary>24×24 Grasshopper component icons (avoids null icon derefs).</summary>
internal static class ComponentIcons24
{
    private static Bitmap? _meshCollision;
    private static Bitmap? _greedyPairs;

    public static Bitmap MeshCollision => _meshCollision ??= DrawMeshCollision();

    public static Bitmap GreedyPointPairs => _greedyPairs ??= DrawGreedyPairs();

    private static Bitmap DrawMeshCollision()
    {
        var bmp = new Bitmap(24, 24);
        using (var g = Graphics.FromImage(bmp))
        {
            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.Clear(Color.Transparent);
            using var penA = new Pen(Color.FromArgb(80, 120, 200), 1.6f);
            using var penB = new Pen(Color.FromArgb(200, 100, 80), 1.6f);
            g.DrawPolygon(penA, new[] { new PointF(4, 18), new PointF(10, 4), new PointF(16, 16) });
            g.DrawPolygon(penB, new[] { new PointF(8, 20), new PointF(14, 6), new PointF(20, 18) });
        }

        return bmp;
    }

    private static Bitmap DrawGreedyPairs()
    {
        var bmp = new Bitmap(24, 24);
        using (var g = Graphics.FromImage(bmp))
        {
            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.Clear(Color.Transparent);
            using var pen = new Pen(Color.FromArgb(60, 140, 100), 1.8f);
            g.DrawLine(pen, 5, 8, 19, 16);
            g.FillEllipse(Brushes.DimGray, 3, 6, 4, 4);
            g.FillEllipse(Brushes.DimGray, 17, 14, 4, 4);
        }

        return bmp;
    }
}
