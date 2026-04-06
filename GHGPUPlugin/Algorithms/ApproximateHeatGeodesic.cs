using System.Collections.Generic;
using GHGPUPlugin.MeshTopology;
using GHGPUPlugin.NativeInterop;
using Rhino.Geometry;

namespace GHGPUPlugin.Algorithms;

/// <summary>Heat-method geodesic: diffuse u, unit-gradient divergence, then Jacobi/SOR on ∇²φ = div (approximate Poisson).</summary>
public static class ApproximateHeatGeodesic
{
    private const int HeatIterations = 60;
    private const float HeatStrength = 0.5f;

    public static bool TryCompute(
        Mesh mesh,
        IReadOnlyList<int> seedMeshVertexIndices,
        int iterations,
        double strength,
        bool useGpu,
        out double[] distancePerMeshVertex,
        out string? error)
    {
        distancePerMeshVertex = Array.Empty<double>();
        error = null;

        int[][] neighbors = MeshTopologyNeighbors.NeighborsFromEdges(mesh);
        int nTopo = neighbors.Length;
        if (nTopo == 0)
        {
            error = "Mesh has no topology vertices.";
            return false;
        }

        MeshTopologyNeighbors.ToCsr(neighbors, out int[] adjFlat, out int[] rowOffsets);

        var tv = mesh.TopologyVertices;
        int vc = mesh.Vertices.Count;
        var heatTopo = new float[nTopo];
        var seedTopo = new bool[nTopo];
        foreach (int mvi in seedMeshVertexIndices)
        {
            if (mvi < 0 || mvi >= vc)
                continue;
            int ti = tv.TopologyVertexIndex(mvi);
            heatTopo[ti] = 1f;
            seedTopo[ti] = true;
        }

        var x = new float[nTopo];
        var y = new float[nTopo];
        var z = new float[nTopo];

        IntPtr heatCtx = IntPtr.Zero;
        bool heatGpu = useGpu && NativeLoader.IsMetalAvailable && MetalSharedContext.TryGetContext(out heatCtx);
        var heatOpts = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };
        double heatStrengthD = HeatStrength;

        for (int it = 0; it < HeatIterations; it++)
        {
            bool iterGpu = false;
            if (heatGpu)
            {
                for (int i = 0; i < nTopo; i++)
                    x[i] = y[i] = z[i] = heatTopo[i];

                int code = MetalBridge.RunLaplacianIterations(
                    heatCtx,
                    x,
                    y,
                    z,
                    adjFlat,
                    rowOffsets,
                    nTopo,
                    HeatStrength,
                    1);
                iterGpu = code == 0;
            }

            if (!iterGpu)
                UmbrellaScalar1(heatTopo, neighbors, heatStrengthD, heatOpts);
            else
            {
                for (int i = 0; i < nTopo; i++)
                    heatTopo[i] = x[i];
            }

            for (int si = 0; si < nTopo; si++)
            {
                if (seedTopo[si])
                    heatTopo[si] = 1f;
            }
        }

        var uMesh = new double[vc];
        for (int mv = 0; mv < vc; mv++)
        {
            int ti = tv.TopologyVertexIndex(mv);
            uMesh[mv] = heatTopo[ti];
        }

        var divTopo = new double[nTopo];
        AccumulateDivergence(mesh, uMesh, divTopo);

        if (iterations < 1)
            iterations = 1;

        var phi = new double[nTopo];
        var phiNew = new double[nTopo];
        var jacobiOpts = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };
        for (int it = 0; it < iterations; it++)
        {
            JacobiStep(phi, phiNew, divTopo, neighbors, strength, jacobiOpts);
            (phi, phiNew) = (phiNew, phi);
        }

        double seedMean = 0;
        int seedCount = 0;
        for (int i = 0; i < nTopo; i++)
        {
            if (!seedTopo[i])
                continue;
            seedMean += phi[i];
            seedCount++;
        }

        if (seedCount > 0)
            seedMean /= seedCount;

        for (int i = 0; i < nTopo; i++)
        {
            phi[i] -= seedMean;
            if (phi[i] < 0)
                phi[i] = 0;
        }

        distancePerMeshVertex = new double[vc];
        for (int mv = 0; mv < vc; mv++)
        {
            int ti = tv.TopologyVertexIndex(mv);
            distancePerMeshVertex[mv] = phi[ti];
        }

        return true;
    }

    private static void UmbrellaScalar1(float[] u, int[][] neighbors, double strength, ParallelOptions opts)
    {
        int n = u.Length;
        var nu = new float[n];
        Parallel.For(0, n, opts, i =>
        {
            int[] nb = neighbors[i];
            if (nb.Length == 0)
            {
                nu[i] = u[i];
                return;
            }

            double s = 0;
            for (int k = 0; k < nb.Length; k++)
                s += u[nb[k]];
            s /= nb.Length;
            nu[i] = (float)(u[i] + strength * (s - u[i]));
        });
        Array.Copy(nu, u, n);
    }

    private static void JacobiStep(
        double[] phi,
        double[] phiNew,
        double[] rhs,
        int[][] neighbors,
        double strength,
        ParallelOptions opts)
    {
        int n = phi.Length;
        Parallel.For(0, n, opts, i =>
        {
            int[] nb = neighbors[i];
            if (nb.Length == 0)
            {
                phiNew[i] = phi[i];
                return;
            }

            double sum = 0;
            for (int k = 0; k < nb.Length; k++)
                sum += phi[nb[k]];
            double target = (sum - rhs[i]) / nb.Length;
            phiNew[i] = phi[i] + strength * (target - phi[i]);
        });
    }

    private static void UmbrellaScalar3(float[] x, float[] y, float[] z, int[][] neighbors, double strength, ParallelOptions opts)
    {
        int n = x.Length;
        var nx = new float[n];
        var ny = new float[n];
        var nz = new float[n];
        Parallel.For(0, n, opts, i =>
        {
            int[] nb = neighbors[i];
            if (nb.Length == 0)
            {
                nx[i] = x[i];
                ny[i] = y[i];
                nz[i] = z[i];
                return;
            }

            double sx = 0, sy = 0, sz = 0;
            for (int k = 0; k < nb.Length; k++)
            {
                int j = nb[k];
                sx += x[j];
                sy += y[j];
                sz += z[j];
            }

            double inv = 1.0 / nb.Length;
            double mx = sx * inv, my = sy * inv, mz = sz * inv;
            nx[i] = (float)(x[i] + strength * (mx - x[i]));
            ny[i] = (float)(y[i] + strength * (my - y[i]));
            nz[i] = (float)(z[i] + strength * (mz - z[i]));
        });
        Array.Copy(nx, x, n);
        Array.Copy(ny, y, n);
        Array.Copy(nz, z, n);
    }

    private static void AccumulateDivergence(Mesh mesh, double[] uMesh, double[] divTopo)
    {
        var tv = mesh.TopologyVertices;
        int nTopo = tv.Count;
        for (int i = 0; i < nTopo; i++)
            divTopo[i] = 0;

        for (int fi = 0; fi < mesh.Faces.Count; fi++)
        {
            MeshFace f = mesh.Faces[fi];
            if (f.IsTriangle)
                AddFaceDiv(mesh, f.A, f.B, f.C, uMesh, divTopo);
            else if (f.IsQuad)
            {
                AddFaceDiv(mesh, f.A, f.B, f.C, uMesh, divTopo);
                AddFaceDiv(mesh, f.A, f.C, f.D, uMesh, divTopo);
            }
        }
    }

    private static void AddFaceDiv(Mesh mesh, int ma, int mb, int mc, double[] uMesh, double[] divTopo)
    {
        Point3d p0 = mesh.Vertices[ma];
        Point3d p1 = mesh.Vertices[mb];
        Point3d p2 = mesh.Vertices[mc];
        double u0 = uMesh[ma], u1 = uMesh[mb], u2 = uMesh[mc];

        Vector3d gu = GradientOnTriangle(p0, p1, p2, u0, u1, u2);
        double gl = gu.Length;
        if (gl < 1e-20)
            return;
        Vector3d X = -gu / gl;

        Vector3d e1 = p1 - p0;
        Vector3d e2 = p2 - p0;
        Vector3d Nu = Vector3d.CrossProduct(e1, e2);
        double a2 = Nu.Length;
        double A = 0.5 * a2;
        if (A < 1e-30)
            return;
        Nu.Unitize();

        Vector3d g0 = Vector3d.CrossProduct(Nu, p2 - p1) / (2.0 * A);
        Vector3d g1 = Vector3d.CrossProduct(Nu, p0 - p2) / (2.0 * A);
        Vector3d g2 = Vector3d.CrossProduct(Nu, p1 - p0) / (2.0 * A);

        int t0 = mesh.TopologyVertices.TopologyVertexIndex(ma);
        int t1 = mesh.TopologyVertices.TopologyVertexIndex(mb);
        int t2 = mesh.TopologyVertices.TopologyVertexIndex(mc);

        divTopo[t0] -= A * (X * g0);
        divTopo[t1] -= A * (X * g1);
        divTopo[t2] -= A * (X * g2);
    }

    private static Vector3d GradientOnTriangle(Point3d p0, Point3d p1, Point3d p2, double u0, double u1, double u2)
    {
        Vector3d e1 = p1 - p0;
        Vector3d e2 = p2 - p0;
        double du1 = u1 - u0;
        double du2 = u2 - u0;
        double g11 = e1 * e1;
        double g12 = e1 * e2;
        double g22 = e2 * e2;
        double det = g11 * g22 - g12 * g12;
        if (Math.Abs(det) < 1e-30)
            return Vector3d.Zero;
        double a = (du1 * g22 - du2 * g12) / det;
        double b = (du2 * g11 - du1 * g12) / det;
        return e1 * a + e2 * b;
    }
}
