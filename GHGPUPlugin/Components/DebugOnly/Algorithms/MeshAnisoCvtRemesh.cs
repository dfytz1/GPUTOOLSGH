using System.Diagnostics;
using GHGPUPlugin.MeshTopology;
using GHGPUPlugin.Utilities;
using GHGPUPlugin.NativeInterop;
using Grasshopper.Kernel;
using Rhino.Collections;
using Rhino.Geometry;

namespace GHGPUPlugin.Algorithms;

/// <summary>Anisotropic CVT-style particle relaxation on a mesh and Delaunay remeshing.</summary>
public static class MeshAnisoCvtRemesh
{
    public static bool TrySolve(
        GH_Component owner,
        Mesh inputMesh,
        int particleCount,
        int iterations,
        double anisotropyStrength,
        double repulsionStrength,
        bool boundaryFixed,
        bool useGpu,
        double circumradiusFactor,
        out Mesh? remeshedMesh,
        out List<Point3d>? particlePositions,
        out double solveTimeMs,
        out string? error)
    {
        remeshedMesh = null;
        particlePositions = null;
        solveTimeMs = 0;
        error = null;

        NativeLoader.EnsureLoaded();

        if (!MeshTriangleUtils.TryGetTriangleMeshForClosest(inputMesh, out Mesh? work) || work == null)
        {
            error = "Mesh needs triangle faces (quads are triangulated once).";
            return false;
        }

        if (!work.IsValid || work.Vertices.Count == 0 || work.Faces.Count == 0)
        {
            error = "Mesh is not usable.";
            return false;
        }

        if (particleCount < 3)
        {
            error = "ParticleCount must be at least 3.";
            return false;
        }

        if (iterations < 1)
        {
            error = "Iterations must be at least 1.";
            return false;
        }

        int vc = work.Vertices.Count;
        if (particleCount > vc * 4)
        {
            owner.AddRuntimeMessage(
                GH_RuntimeMessageLevel.Warning,
                $"ParticleCount ({particleCount}) is greater than mesh vertex count × 4 ({vc * 4}); quality may suffer.");
        }

        if (circumradiusFactor < 0)
            circumradiusFactor = 0;
        if (circumradiusFactor > 0 && circumradiusFactor < 2.0)
        {
            owner.AddRuntimeMessage(
                GH_RuntimeMessageLevel.Remark,
                "CircumradiusFactor under 2× target spacing often strips boundary Delaunay triangles; the component falls back to full Delaunay if too few faces remain. Prefer ~3.5–6 for closed-looking meshes.");
        }

        anisotropyStrength = Math.Clamp(anisotropyStrength, 0.0, 2.0);
        repulsionStrength = Math.Clamp(repulsionStrength, 0.0, 1.0);

        double totalArea = 0.0;
        int triCount = work.Faces.Count;
        var triAreas = new double[triCount];
        for (int fi = 0; fi < triCount; fi++)
        {
            MeshFace f = work.Faces[fi];
            if (!f.IsTriangle)
            {
                error = "Triangle mesh expected.";
                return false;
            }

            Point3d a = work.Vertices[f.A];
            Point3d b = work.Vertices[f.B];
            Point3d c = work.Vertices[f.C];
            double aTri = 0.5 * Vector3d.CrossProduct(b - a, c - a).Length;
            triAreas[fi] = aTri;
            totalArea += aTri;
        }

        if (totalArea < 1e-20)
        {
            error = "Mesh surface area is degenerate.";
            return false;
        }

        double targetSpacing = Math.Sqrt(totalArea / particleCount) * 1.1;
        float cellSize = (float)(targetSpacing * 1.5);
        float invCell = 1.0f / cellSize;

        var sw = Stopwatch.StartNew();

        if (!BuildTopoCurvatureData(work, out float[] px, out float[] py, out float[] pz, out float[] nx, out float[] ny,
                out float[] nz, out int[] adjFlat, out float[] cotW, out int[] rowOff, out float[] mixedArea,
                out float[] angleSum, out byte[] topoBoundary, out int nTopo))
        {
            error = "Failed to build topology cotan data.";
            return false;
        }

        var metricTopo = new float[Math.Max(1, nTopo * 9)];

        IntPtr ctx = IntPtr.Zero;
        bool gpuMetric = useGpu && MetalGuard.EnsureReady(owner) && MetalSharedContext.TryGetContext(out ctx);
        if (gpuMetric)
        {
            int code = MetalBridge.AnisoCvtComputeMetricTopo(
                ctx,
                px,
                py,
                pz,
                nx,
                ny,
                nz,
                adjFlat,
                cotW,
                rowOff,
                mixedArea,
                angleSum,
                topoBoundary,
                nTopo,
                (float)anisotropyStrength,
                metricTopo);
            if (code != 0)
            {
                owner.AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"GPU metric tensor failed ({code}); using isotropic CPU metric.");
                gpuMetric = false;
            }
        }

        if (!gpuMetric)
            FillIsotropicMetric(metricTopo, nTopo);

        var vx = new float[vc];
        var vy = new float[vc];
        var vz = new float[vc];
        for (int i = 0; i < vc; i++)
        {
            Point3f p = work.Vertices[i];
            vx[i] = p.X;
            vy[i] = p.Y;
            vz[i] = p.Z;
        }

        var triIdx = new int[triCount * 3];
        for (int fi = 0; fi < triCount; fi++)
        {
            MeshFace f = work.Faces[fi];
            triIdx[3 * fi] = f.A;
            triIdx[3 * fi + 1] = f.B;
            triIdx[3 * fi + 2] = f.C;
        }

        if (!TryBuildNakedSegments(work, out float[] segAx, out float[] segAy, out float[] segAz, out float[] segBx,
                out float[] segBy, out float[] segBz, out int nSeg))
        {
            segAx = Array.Empty<float>();
            segAy = Array.Empty<float>();
            segAz = Array.Empty<float>();
            segBx = Array.Empty<float>();
            segBy = Array.Empty<float>();
            segBz = Array.Empty<float>();
            nSeg = 0;
        }

        var rnd = new Random(17);
        var posX = new float[particleCount];
        var posY = new float[particleCount];
        var posZ = new float[particleCount];
        var fixedMask = new byte[particleCount];
        var boundaryParticle = new byte[particleCount];
        var metricP = new float[particleCount * 9];

        AreaWeightedSeeding(
            work,
            triAreas,
            totalArea,
            particleCount,
            rnd,
            boundaryFixed,
            posX,
            posY,
            posZ,
            fixedMask,
            boundaryParticle,
            metricTopo,
            nTopo,
            metricP);

        BoundingBox bb = work.GetBoundingBox(true);
        bb.Inflate(cellSize * 2.0);
        float bbMinX = (float)bb.Min.X;
        float bbMinY = (float)bb.Min.Y;
        float bbMinZ = (float)bb.Min.Z;
        int dimX = Math.Max(1, (int)Math.Ceiling((bb.Max.X - bb.Min.X) / cellSize));
        int dimY = Math.Max(1, (int)Math.Ceiling((bb.Max.Y - bb.Min.Y) / cellSize));
        int dimZ = Math.Max(1, (int)Math.Ceiling((bb.Max.Z - bb.Min.Z) / cellSize));

        var outCx = new float[particleCount];
        var outCy = new float[particleCount];
        var outCz = new float[particleCount];
        var outD2 = new float[particleCount];
        var outTi = new int[particleCount];

        bool gpuActive = useGpu && MetalGuard.EnsureReady(owner) && MetalSharedContext.TryGetContext(out ctx);
        int gpuItersDone = 0;
        if (gpuActive)
        {
            for (int it = 0; it < iterations; it++)
            {
                int g = MetalBridge.AnisoCvtParticleGpuPreProject(
                    ctx,
                    posX,
                    posY,
                    posZ,
                    metricP,
                    fixedMask,
                    particleCount,
                    bbMinX,
                    bbMinY,
                    bbMinZ,
                    dimX,
                    dimY,
                    dimZ,
                    cellSize,
                    invCell,
                    (float)targetSpacing,
                    (float)repulsionStrength,
                    0.3f);
                if (g != 0)
                {
                    owner.AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"GPU particle step failed ({g}); continuing on CPU.");
                    break;
                }

                int c = MetalBridge.ClosestPointsMesh(
                    ctx,
                    posX,
                    posY,
                    posZ,
                    particleCount,
                    vx,
                    vy,
                    vz,
                    vc,
                    triIdx,
                    triCount,
                    outCx,
                    outCy,
                    outCz,
                    outD2,
                    outTi);
                if (c != 0)
                {
                    owner.AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"GPU closest-point failed ({c}); continuing on CPU.");
                    break;
                }

                Array.Copy(outCx, posX, particleCount);
                Array.Copy(outCy, posY, particleCount);
                Array.Copy(outCz, posZ, particleCount);

                if (boundaryFixed && nSeg > 0)
                {
                    int b = MetalBridge.AnisoCvtProjectBoundarySegments(
                        ctx,
                        posX,
                        posY,
                        posZ,
                        boundaryParticle,
                        particleCount,
                        segAx,
                        segAy,
                        segAz,
                        segBx,
                        segBy,
                        segBz,
                        nSeg);
                    if (b != 0)
                    {
                        owner.AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"GPU boundary projection failed ({b}).");
                    }
                }

                gpuItersDone++;
            }
        }

        int remaining = iterations - gpuItersDone;
        if (remaining > 0)
        {
            CpuParticleRelaxation(
                work,
                posX,
                posY,
                posZ,
                fixedMask,
                particleCount,
                targetSpacing,
                remaining,
                bbMinX,
                bbMinY,
                bbMinZ,
                dimX,
                dimY,
                dimZ,
                cellSize,
                invCell);
        }

        sw.Stop();
        solveTimeMs = sw.Elapsed.TotalMilliseconds;

        var pts = new List<Point3d>(particleCount);
        for (int i = 0; i < particleCount; i++)
            pts.Add(new Point3d(posX[i], posY[i], posZ[i]));

        particlePositions = pts;

        if (!TryDelaunayRemesh(owner, pts, targetSpacing, circumradiusFactor, out Mesh? rm))
        {
            error = "Delaunay remeshing failed (degenerate point set?).";
            return false;
        }

        remeshedMesh = rm;
        return true;
    }

    private static void FillIsotropicMetric(float[] metricTopo, int nTopo)
    {
        for (int i = 0; i < nTopo; i++)
        {
            int o = i * 9;
            metricTopo[o] = 1;
            metricTopo[o + 1] = 0;
            metricTopo[o + 2] = 0;
            metricTopo[o + 3] = 0;
            metricTopo[o + 4] = 1;
            metricTopo[o + 5] = 0;
            metricTopo[o + 6] = 0;
            metricTopo[o + 7] = 0;
            metricTopo[o + 8] = 1;
        }
    }

    private static bool BuildTopoCurvatureData(
        Mesh mesh,
        out float[] px,
        out float[] py,
        out float[] pz,
        out float[] nx,
        out float[] ny,
        out float[] nz,
        out int[] adjFlat,
        out float[] cotW,
        out int[] rowOff,
        out float[] mixedArea,
        out float[] angleSum,
        out byte[] topoBoundary,
        out int nTopo)
    {
        var tv = mesh.TopologyVertices;
        nTopo = tv.Count;
        px = new float[nTopo];
        py = new float[nTopo];
        pz = new float[nTopo];
        nx = new float[nTopo];
        ny = new float[nTopo];
        nz = new float[nTopo];
        mixedArea = new float[nTopo];
        angleSum = new float[nTopo];
        topoBoundary = new byte[nTopo];

        for (int i = 0; i < nTopo; i++)
        {
            Point3d p = tv[i];
            px[i] = (float)p.X;
            py[i] = (float)p.Y;
            pz[i] = (float)p.Z;
        }

        var norms = mesh.Normals;
        for (int ti = 0; ti < nTopo; ti++)
        {
            int[] mv = tv.MeshVertexIndices(ti);
            Vector3f sum = Vector3f.Zero;
            int cnt = 0;
            for (int k = 0; k < mv.Length; k++)
            {
                int vi = mv[k];
                if (norms != null && vi < norms.Count)
                {
                    sum += norms[vi];
                    cnt++;
                }
            }

            if (cnt > 0)
                sum /= cnt;
            if (sum.Length < 1e-20f)
                sum = new Vector3f(0, 0, 1);
            else
                sum.Unitize();
            nx[ti] = sum.X;
            ny[ti] = sum.Y;
            nz[ti] = sum.Z;
        }

        var cotSum = new Dictionary<long, double>();
        var mixedD = new double[nTopo];
        var angleD = new double[nTopo];
        for (int fi = 0; fi < mesh.Faces.Count; fi++)
        {
            MeshFace f = mesh.Faces[fi];
            if (f.IsTriangle)
                ProcessTriangle(mesh, f.A, f.B, f.C, cotSum, angleD, mixedD);
            else if (f.IsQuad)
            {
                ProcessTriangle(mesh, f.A, f.B, f.C, cotSum, angleD, mixedD);
                ProcessTriangle(mesh, f.A, f.C, f.D, cotSum, angleD, mixedD);
            }
        }

        for (int i = 0; i < nTopo; i++)
        {
            mixedArea[i] = (float)mixedD[i];
            angleSum[i] = (float)angleD[i];
            topoBoundary[i] = (byte)(IsBoundaryTopoVertex(mesh, i) ? 1 : 0);
        }

        int[][] nb = MeshTopologyNeighbors.NeighborsFromEdges(mesh);
        MeshTopologyNeighbors.ToCsr(nb, out adjFlat, out rowOff);
        int nnz = adjFlat.Length;
        cotW = new float[nnz];
        for (int i = 0; i < nTopo; i++)
        {
            int a0 = rowOff[i];
            int a1 = rowOff[i + 1];
            for (int k = a0; k < a1; k++)
            {
                int j = adjFlat[k];
                int a = i < j ? i : j;
                int b = i < j ? j : i;
                long key = ((long)a << 32) | (uint)b;
                double cw = cotSum.TryGetValue(key, out double c) ? c : 0.0;
                cotW[k] = (float)cw;
            }
        }

        return true;
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

        double a = 0.5 * Vector3d.CrossProduct(pb - pa, pc - pa).Length;
        if (a < 1e-30)
            return;

        AddCotEdge(ta, tb, pa, pb, pc, cotSum);
        AddCotEdge(tb, tc, pb, pc, pa, cotSum);
        AddCotEdge(tc, ta, pc, pa, pb, cotSum);

        angleSum[ta] += AngleAt(pc, pb, pa);
        angleSum[tb] += AngleAt(pa, pc, pb);
        angleSum[tc] += AngleAt(pb, pa, pc);

        double third = a / 3.0;
        mixedArea[ta] += third;
        mixedArea[tb] += third;
        mixedArea[tc] += third;
    }

    private static void AddCotEdge(int ta, int tb, Point3d pa, Point3d pb, Point3d pc, Dictionary<long, double> cotSum)
    {
        var u = pa - pc;
        var v = pb - pc;
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
        var u = p0 - apex;
        var v = p1 - apex;
        if (u.IsTiny() || v.IsTiny())
            return 0;
        u.Unitize();
        v.Unitize();
        double c = Math.Clamp(u * v, -1, 1);
        return Math.Acos(c);
    }

    private static bool TryBuildNakedSegments(
        Mesh mesh,
        out float[] ax,
        out float[] ay,
        out float[] az,
        out float[] bx,
        out float[] by,
        out float[] bz,
        out int nSeg)
    {
        var te = mesh.TopologyEdges;
        var lax = new List<float>();
        var lay = new List<float>();
        var laz = new List<float>();
        var lbx = new List<float>();
        var lby = new List<float>();
        var lbz = new List<float>();
        var tv = mesh.TopologyVertices;
        for (int ei = 0; ei < te.Count; ei++)
        {
            if (te.GetConnectedFaces(ei).Length != 1)
                continue;
            var ends = te.GetTopologyVertices(ei);
            Point3d a = tv[ends.I];
            Point3d b = tv[ends.J];
            lax.Add((float)a.X);
            lay.Add((float)a.Y);
            laz.Add((float)a.Z);
            lbx.Add((float)b.X);
            lby.Add((float)b.Y);
            lbz.Add((float)b.Z);
        }

        ax = lax.ToArray();
        ay = lay.ToArray();
        az = laz.ToArray();
        bx = lbx.ToArray();
        by = lby.ToArray();
        bz = lbz.ToArray();
        nSeg = ax.Length;
        return nSeg > 0;
    }

    private static bool TriangleTouchesBoundary(Mesh mesh, int fi)
    {
        MeshFace f = mesh.Faces[fi];
        if (!f.IsTriangle)
            return false;
        int ta = mesh.TopologyVertices.TopologyVertexIndex(f.A);
        int tb = mesh.TopologyVertices.TopologyVertexIndex(f.B);
        int tc = mesh.TopologyVertices.TopologyVertexIndex(f.C);
        return EdgeIsNaked(mesh, ta, tb) || EdgeIsNaked(mesh, tb, tc) || EdgeIsNaked(mesh, tc, ta);
    }

    private static bool EdgeIsNaked(Mesh mesh, int topoA, int topoB)
    {
        int ei = FindTopologyEdgeIndex(mesh, topoA, topoB);
        if (ei < 0)
            return true;
        return mesh.TopologyEdges.GetConnectedFaces(ei).Length < 2;
    }

    private static int FindTopologyEdgeIndex(Mesh mesh, int topoA, int topoB)
    {
        var te = mesh.TopologyEdges;
        for (int ei = 0; ei < te.Count; ei++)
        {
            var ends = te.GetTopologyVertices(ei);
            if ((ends.I == topoA && ends.J == topoB) || (ends.I == topoB && ends.J == topoA))
                return ei;
        }

        return -1;
    }

    private static void AreaWeightedSeeding(
        Mesh mesh,
        double[] triAreas,
        double totalArea,
        int nPart,
        Random rnd,
        bool boundaryFixed,
        float[] posX,
        float[] posY,
        float[] posZ,
        byte[] fixedMask,
        byte[] boundaryParticle,
        float[] metricTopo,
        int nTopo,
        float[] metricP)
    {
        int triCount = mesh.Faces.Count;
        var cum = new double[triCount];
        double s = 0.0;
        for (int i = 0; i < triCount; i++)
        {
            s += triAreas[i];
            cum[i] = s;
        }

        var tv = mesh.TopologyVertices;
        for (int p = 0; p < nPart; p++)
        {
            double t = rnd.NextDouble() * totalArea;
            int lo = 0, hi = triCount - 1;
            while (lo < hi)
            {
                int mid = (lo + hi) / 2;
                if (cum[mid] < t)
                    lo = mid + 1;
                else
                    hi = mid;
            }

            int fi = lo;
            MeshFace f = mesh.Faces[fi];
            Point3d a = mesh.Vertices[f.A];
            Point3d b = mesh.Vertices[f.B];
            Point3d c = mesh.Vertices[f.C];
            double r1 = rnd.NextDouble();
            double r2 = rnd.NextDouble();
            if (r1 + r2 > 1.0)
            {
                r1 = 1.0 - r1;
                r2 = 1.0 - r2;
            }

            double w = 1.0 - r1 - r2;
            Point3d pt = a * w + b * r1 + c * r2;
            posX[p] = (float)pt.X;
            posY[p] = (float)pt.Y;
            posZ[p] = (float)pt.Z;

            bool onB = TriangleTouchesBoundary(mesh, fi);
            if (boundaryFixed && onB)
            {
                fixedMask[p] = 1;
                boundaryParticle[p] = 1;
            }
            else
            {
                fixedMask[p] = 0;
                boundaryParticle[p] = 0;
            }

            int t0 = tv.TopologyVertexIndex(f.A);
            int t1 = tv.TopologyVertexIndex(f.B);
            int t2 = tv.TopologyVertexIndex(f.C);
            int o = p * 9;
            for (int k = 0; k < 9; k++)
            {
                metricP[o + k] = (float)(w * metricTopo[t0 * 9 + k] + r1 * metricTopo[t1 * 9 + k]
                    + r2 * metricTopo[t2 * 9 + k]);
            }
        }
    }

    private static void CpuParticleRelaxation(
        Mesh mesh,
        float[] posX,
        float[] posY,
        float[] posZ,
        byte[] fixedMask,
        int n,
        double targetSpacing,
        int iterations,
        float bbMinX,
        float bbMinY,
        float bbMinZ,
        int dimX,
        int dimY,
        int dimZ,
        float cellSize,
        float invCell)
    {
        double neighR = targetSpacing * 2.0;
        double neighR2 = neighR * neighR;
        var ox = new float[n];
        var oy = new float[n];
        var oz = new float[n];
        var oti = new int[n];
        var od2 = new float[n];

        int vc = mesh.Vertices.Count;
        var vx = new float[vc];
        var vy = new float[vc];
        var vz = new float[vc];
        for (int i = 0; i < vc; i++)
        {
            Point3f p = mesh.Vertices[i];
            vx[i] = p.X;
            vy[i] = p.Y;
            vz[i] = p.Z;
        }

        int triCount = mesh.Faces.Count;
        var triIdx = new int[triCount * 3];
        for (int fi = 0; fi < triCount; fi++)
        {
            MeshFace f = mesh.Faces[fi];
            triIdx[3 * fi] = f.A;
            triIdx[3 * fi + 1] = f.B;
            triIdx[3 * fi + 2] = f.C;
        }

        IntPtr ctx = IntPtr.Zero;
        MetalSharedContext.TryGetContext(out ctx);

        for (int it = 0; it < iterations; it++)
        {
            CpuLaplacianSpatialHash(
                posX,
                posY,
                posZ,
                fixedMask,
                n,
                bbMinX,
                bbMinY,
                bbMinZ,
                dimX,
                dimY,
                dimZ,
                invCell,
                neighR2,
                0.3);

            if (ctx != IntPtr.Zero)
            {
                int c = MetalBridge.ClosestPointsMesh(
                    ctx,
                    posX,
                    posY,
                    posZ,
                    n,
                    vx,
                    vy,
                    vz,
                    vc,
                    triIdx,
                    triCount,
                    ox,
                    oy,
                    oz,
                    od2,
                    oti);
                if (c == 0)
                {
                    Array.Copy(ox, posX, n);
                    Array.Copy(oy, posY, n);
                    Array.Copy(oz, posZ, n);
                    continue;
                }
            }

            for (int i = 0; i < n; i++)
            {
                var q = new Point3d(posX[i], posY[i], posZ[i]);
                MeshPoint mp = mesh.ClosestMeshPoint(q, double.MaxValue);
                posX[i] = (float)mp.Point.X;
                posY[i] = (float)mp.Point.Y;
                posZ[i] = (float)mp.Point.Z;
            }
        }
    }

    private static void CpuLaplacianSpatialHash(
        float[] x,
        float[] y,
        float[] z,
        byte[] fixedMask,
        int n,
        float bbMinX,
        float bbMinY,
        float bbMinZ,
        int dimX,
        int dimY,
        int dimZ,
        float invCell,
        double neighR2,
        double strength)
    {
        int nCells = dimX * dimY * dimZ;
        var cellLists = new List<int>[nCells];
        for (int i = 0; i < nCells; i++)
            cellLists[i] = new List<int>();

        static int CellIdx(float px, float py, float pz, float bx, float by, float bz, float inv, int dx, int dy, int dz)
        {
            int ix = (int)Math.Floor((px - bx) * inv);
            int iy = (int)Math.Floor((py - by) * inv);
            int iz = (int)Math.Floor((pz - bz) * inv);
            ix = Math.Clamp(ix, 0, Math.Max(0, dx - 1));
            iy = Math.Clamp(iy, 0, Math.Max(0, dy - 1));
            iz = Math.Clamp(iz, 0, Math.Max(0, dz - 1));
            return ix + dx * (iy + dy * iz);
        }

        for (int i = 0; i < n; i++)
        {
            int ci = CellIdx(x[i], y[i], z[i], bbMinX, bbMinY, bbMinZ, invCell, dimX, dimY, dimZ);
            cellLists[ci].Add(i);
        }

        var nx = new float[n];
        var ny = new float[n];
        var nz = new float[n];
        Array.Copy(x, nx, n);
        Array.Copy(y, ny, n);
        Array.Copy(z, nz, n);

        for (int i = 0; i < n; i++)
        {
            if (fixedMask[i] != 0)
                continue;
            int ix = (int)Math.Floor((x[i] - bbMinX) * invCell);
            int iy = (int)Math.Floor((y[i] - bbMinY) * invCell);
            int iz = (int)Math.Floor((z[i] - bbMinZ) * invCell);
            ix = Math.Clamp(ix, 0, Math.Max(0, dimX - 1));
            iy = Math.Clamp(iy, 0, Math.Max(0, dimY - 1));
            iz = Math.Clamp(iz, 0, Math.Max(0, dimZ - 1));
            double sx = 0, sy = 0, sz = 0;
            int cnt = 0;
            for (int oz = -1; oz <= 1; oz++)
            {
                for (int oy = -1; oy <= 1; oy++)
                {
                    for (int ox = -1; ox <= 1; ox++)
                    {
                        int cx = ix + ox;
                        int cy = iy + oy;
                        int cz = iz + oz;
                        if (cx < 0 || cy < 0 || cz < 0 || cx >= dimX || cy >= dimY || cz >= dimZ)
                            continue;
                        int cidx = cx + dimX * (cy + dimY * cz);
                        List<int> lst = cellLists[cidx];
                        for (int t = 0; t < lst.Count; t++)
                        {
                            int j = lst[t];
                            if (j == i)
                                continue;
                            double dx = x[i] - x[j];
                            double dy = y[i] - y[j];
                            double dz = z[i] - z[j];
                            double d2 = dx * dx + dy * dy + dz * dz;
                            if (d2 <= neighR2 && d2 > 0)
                            {
                                sx += x[j];
                                sy += y[j];
                                sz += z[j];
                                cnt++;
                            }
                        }
                    }
                }
            }

            if (cnt <= 0)
                continue;
            sx /= cnt;
            sy /= cnt;
            sz /= cnt;
            nx[i] = (float)(x[i] + strength * (sx - x[i]));
            ny[i] = (float)(y[i] + strength * (sy - y[i]));
            nz[i] = (float)(z[i] + strength * (sz - z[i]));
        }

        Array.Copy(nx, x, n);
        Array.Copy(ny, y, n);
        Array.Copy(nz, z, n);
    }

    private static bool TryDelaunayRemesh(
        GH_Component owner,
        List<Point3d> pts,
        double targetSpacing,
        double circumradiusFactor,
        out Mesh? mesh)
    {
        mesh = null;
        int n = pts.Count;
        if (n < 3)
            return false;

        var pl = new Point3dList(pts);
        PlaneFitResult fit = Plane.FitPlaneToPoints(pl, out Plane pln);
        if ((fit != PlaneFitResult.Success && fit != PlaneFitResult.Inconclusive) || !pln.IsValid)
            return false;

        var uv = new List<Vector2d>(n);
        for (int i = 0; i < n; i++)
        {
            pln.ClosestParameter(pts[i], out double u, out double v);
            uv.Add(new Vector2d(u, v));
        }

        List<int> tris = AnisoCvtDelaunay2D.BowyerWatson(uv);
        if (tris.Count < 9)
            return false;

        int nTriIn = tris.Count / 3;
        List<int> keepTris;

        if (circumradiusFactor <= 0)
        {
            keepTris = new List<int>(tris);
        }
        else
        {
            double rMax = circumradiusFactor * targetSpacing;
            keepTris = new List<int>();
            for (int t = 0; t < tris.Count; t += 3)
            {
                int ia = tris[t];
                int ib = tris[t + 1];
                int ic = tris[t + 2];
                Point3d a = pts[ia];
                Point3d b = pts[ib];
                Point3d c = pts[ic];
                double rad = AnisoCvtDelaunay2D.Circumradius(a, b, c);
                if (rad <= rMax)
                {
                    keepTris.Add(ia);
                    keepTris.Add(ib);
                    keepTris.Add(ic);
                }
            }

            int nKeep = keepTris.Count / 3;
            // Planar Delaunay of n sites is ~2n−2−b triangles; aggressive rMax removes boundary/large faces first → holes.
            int minKeep = Math.Max(4, Math.Min(nTriIn, (n * 2) / 3));
            if (nKeep < minKeep)
            {
                owner.AddRuntimeMessage(
                    GH_RuntimeMessageLevel.Warning,
                    $"Circumradius filter kept only {nKeep} of {nTriIn} Delaunay triangles (limit {rMax:0.####}). Using full Delaunay. Increase CircumradiusFactor (e.g. 3–6) if you want both thinning and coverage.");
                keepTris = new List<int>(tris);
            }
        }

        if (keepTris.Count < 9)
            return false;

        mesh = new Mesh();
        for (int i = 0; i < n; i++)
            mesh.Vertices.Add(pts[i]);
        for (int t = 0; t < keepTris.Count; t += 3)
            mesh.Faces.AddFace(keepTris[t], keepTris[t + 1], keepTris[t + 2]);

        mesh.Faces.ConvertQuadsToTriangles();
        mesh.Normals.ComputeNormals();
        mesh.UnifyNormals();
        mesh.Weld(Math.Max(1e-8, targetSpacing * 0.1));
        return mesh.IsValid;
    }
}
