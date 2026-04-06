using System.Threading.Tasks;
using Rhino.Geometry;

namespace GHGPUPlugin.Algorithms;

/// <summary>Discrete mean and Gaussian curvature per topology vertex (cotangent Laplacian).</summary>
public static class MeshCurvatureDiscrete
{
    public static void Compute(Mesh mesh, bool useParallel, out double[] meanH, out double[] gaussianK, out double[] kMin, out double[] kMax)
    {
        int n = mesh.TopologyVertices.Count;
        meanH = new double[n];
        gaussianK = new double[n];
        kMin = new double[n];
        kMax = new double[n];

        var tv = mesh.TopologyVertices;
        var cotSum = new Dictionary<long, double>();
        var angleSum = new double[n];
        var mixedArea = new double[n];

        for (int fi = 0; fi < mesh.Faces.Count; fi++)
        {
            MeshFace f = mesh.Faces[fi];
            if (f.IsTriangle)
                ProcessTriangle(mesh, f.A, f.B, f.C, cotSum, angleSum, mixedArea);
            else if (f.IsQuad)
            {
                ProcessTriangle(mesh, f.A, f.B, f.C, cotSum, angleSum, mixedArea);
                ProcessTriangle(mesh, f.A, f.C, f.D, cotSum, angleSum, mixedArea);
            }
        }

        var opts = new ParallelOptions { MaxDegreeOfParallelism = useParallel ? Environment.ProcessorCount : 1 };
        var meanTmp = new double[n];
        var gaussTmp = new double[n];
        var kMinTmp = new double[n];
        var kMaxTmp = new double[n];

        Parallel.For(0, n, opts, i =>
        {
            Point3d pi = tv[i];
            Vector3d lap = Vector3d.Zero;
            int[] nb = tv.ConnectedTopologyVertices(i);
            for (int k = 0; k < nb.Length; k++)
            {
                int j = nb[k];
                int a = i < j ? i : j;
                int b = i < j ? j : i;
                long key = ((long)a << 32) | (uint)b;
                double cot = cotSum.TryGetValue(key, out double c) ? c : 0;
                Point3d pj = tv[j];
                lap += cot * (pj - pi);
            }

            double A = mixedArea[i];
            if (A < 1e-30)
                A = 1e-30;

            double meanMag = lap.Length / (2.0 * A);
            meanTmp[i] = meanMag;

            double sumAng = angleSum[i];
            bool boundary = IsBoundaryTopoVertex(mesh, i);
            double defect = boundary ? Math.PI - sumAng : 2.0 * Math.PI - sumAng;
            gaussTmp[i] = defect / A;

            double disc = meanMag * meanMag - gaussTmp[i];
            if (disc < 0)
                disc = 0;
            double s = Math.Sqrt(disc);
            kMinTmp[i] = meanMag - s;
            kMaxTmp[i] = meanMag + s;
        });

        Array.Copy(meanTmp, meanH, n);
        Array.Copy(gaussTmp, gaussianK, n);
        Array.Copy(kMinTmp, kMin, n);
        Array.Copy(kMaxTmp, kMax, n);
    }

    private static bool IsBoundaryTopoVertex(Mesh mesh, int topologyVertexIndex)
    {
        int[] edges = mesh.TopologyVertices.ConnectedEdges(topologyVertexIndex);
        for (int k = 0; k < edges.Length; k++)
        {
            int[] faces = mesh.TopologyEdges.GetConnectedFaces(edges[k]);
            if (faces.Length < 2)
                return true;
        }

        return false;
    }

    private static void ProcessTriangle(
        Mesh mesh,
        int ma,
        int mb,
        int mc,
        Dictionary<long, double> cotSum,
        double[] angleSum,
        double[] mixedArea)
    {
        Point3d pa = mesh.Vertices[ma];
        Point3d pb = mesh.Vertices[mb];
        Point3d pc = mesh.Vertices[mc];
        int ta = mesh.TopologyVertices.TopologyVertexIndex(ma);
        int tb = mesh.TopologyVertices.TopologyVertexIndex(mb);
        int tc = mesh.TopologyVertices.TopologyVertexIndex(mc);

        double A = 0.5 * Vector3d.CrossProduct(pb - pa, pc - pa).Length;
        if (A < 1e-30)
            return;

        AddCotEdge(ta, tb, pa, pb, pc, cotSum);
        AddCotEdge(tb, tc, pb, pc, pa, cotSum);
        AddCotEdge(tc, ta, pc, pa, pb, cotSum);

        angleSum[ta] += AngleAt(pc, pb, pa);
        angleSum[tb] += AngleAt(pa, pc, pb);
        angleSum[tc] += AngleAt(pb, pa, pc);

        double third = A / 3.0;
        mixedArea[ta] += third;
        mixedArea[tb] += third;
        mixedArea[tc] += third;
    }

    private static void AddCotEdge(int ta, int tb, Point3d pa, Point3d pb, Point3d pc, Dictionary<long, double> cotSum)
    {
        Vector3d u = pa - pc;
        Vector3d v = pb - pc;
        double lu = u.Length, lv = v.Length;
        if (lu < 1e-30 || lv < 1e-30)
            return;
        double sin = Vector3d.CrossProduct(u, v).Length / (lu * lv);
        if (sin < 1e-10)
            return;
        double cos = (u * v) / (lu * lv);
        double cot = cos / sin;
        int i = ta < tb ? ta : tb;
        int j = ta < tb ? tb : ta;
        long key = ((long)i << 32) | (uint)j;
        if (cotSum.TryGetValue(key, out double old))
            cotSum[key] = old + cot;
        else
            cotSum[key] = cot;
    }

    private static double AngleAt(Point3d apex, Point3d p0, Point3d p1)
    {
        Vector3d u = p0 - apex;
        Vector3d v = p1 - apex;
        if (u.IsTiny() || v.IsTiny())
            return 0;
        u.Unitize();
        v.Unitize();
        double c = Math.Clamp(u * v, -1, 1);
        return Math.Acos(c);
    }
}
