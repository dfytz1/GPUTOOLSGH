using System.Drawing;
using System.Threading;
using System.Threading.Tasks;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;
using GHGPUPlugin.NativeInterop;
using GHGPUPlugin.Utilities;

namespace GHGPUPlugin.Components.DataRelationships;

/// <summary>
/// Batched triangle–triangle tests: one Metal thread per (mesh index A, mesh index B), all meshes packed into
/// two SoA buffers. Empty MeshesB reuses the A pack (same list) for A×A tests including per-mesh self (upper-triangle pairs).
/// </summary>
public class GH_MeshCollisionGPU : GH_Component
{
    private const long MaxTrianglePairTests = 50_000_000L;

    private readonly struct TriD
    {
        internal readonly double X0, Y0, Z0, X1, Y1, Z1, X2, Y2, Z2;

        /// <param name="globalTriIndex">Triangle index in the concatenated triangle buffer (not mesh-local).</param>
        internal TriD(float[] px, float[] py, float[] pz, int[] tri, int globalTriIndex)
        {
            int o = globalTriIndex * 3;
            int i0 = tri[o];
            int i1 = tri[o + 1];
            int i2 = tri[o + 2];
            X0 = px[i0];
            Y0 = py[i0];
            Z0 = pz[i0];
            X1 = px[i1];
            Y1 = py[i1];
            Z1 = pz[i1];
            X2 = px[i2];
            Y2 = py[i2];
            Z2 = pz[i2];
        }
    }

    public GH_MeshCollisionGPU()
        : base(
            "Mesh Collision GPU",
            "MeshHitGPU",
            "Test many meshes in one solve: provide lists MeshesA and MeshesB (or leave B empty to use A×A). "
                + "One GPU thread runs all triangle tests for a single (mesh A index, mesh B index) pair, so one dispatch covers the whole batch. "
                + "When B is empty and the same packed list is used, diagonal pairs (i,i) run self intersection (unique triangle pairs only). "
                + "Optional skip skips those diagonal mesh pairs entirely.",
            "GPUTools",
            "Mesh")
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddMeshParameter("MeshesA", "A", "Meshes in set A (list).", GH_ParamAccess.list);
        pManager.AddMeshParameter("MeshesB", "B", "Meshes in set B (list). Leave empty to test A against A (same pack).", GH_ParamAccess.list);
        pManager.AddIntegerParameter("MaxHits", "MaxHits", "Maximum intersecting triangle hits to record (each hit includes mesh indices).", GH_ParamAccess.item, 4096);
        pManager.AddBooleanParameter("SkipIntraMeshPair", "SkipDiag", "When B is empty (A×A), skip mesh pairs with the same index (no mesh self-test).", GH_ParamAccess.item, false);
        pManager.AddBooleanParameter("UseGPU", "GPU", "Use Metal when available.", GH_ParamAccess.item, true);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddBooleanParameter("Colliding", "C", "True if at least one triangle pair intersects.", GH_ParamAccess.item);
        pManager.AddIntegerParameter("HitCount", "N", "Total hits (can exceed MaxHits).", GH_ParamAccess.item);
        pManager.AddIntegerParameter("MeshIndexA", "mA", "Index into MeshesA for each recorded hit.", GH_ParamAccess.list);
        pManager.AddIntegerParameter("MeshIndexB", "mB", "Index into MeshesB (or MeshesA if B was empty) for each recorded hit.", GH_ParamAccess.list);
        pManager.AddIntegerParameter("TriangleIndexA", "iA", "Triangle index local to that mesh A entry.", GH_ParamAccess.list);
        pManager.AddIntegerParameter("TriangleIndexB", "iB", "Triangle index local to that mesh B entry.", GH_ParamAccess.list);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        NativeLoader.EnsureLoaded();

        if (!TryCollectMeshes(DA, 0, "MeshesA", out List<Mesh> meshesA))
            return;

        var ghB = new List<GH_Mesh>();
        DA.GetDataList(1, ghB);
        List<Mesh>? meshesBResolved = null;
        if (ghB.Count > 0)
        {
            meshesBResolved = new List<Mesh>();
            foreach (GH_Mesh? g in ghB)
            {
                if (g?.Value != null && g.Value.IsValid)
                    meshesBResolved.Add(g.Value);
            }

            if (meshesBResolved.Count == 0)
                meshesBResolved = null;
        }

        int maxHits = 4096;
        DA.GetData(2, ref maxHits);
        if (maxHits < 1)
            maxHits = 1;

        bool skipIntraMeshPair = false;
        DA.GetData(3, ref skipIntraMeshPair);

        bool useGpu = true;
        DA.GetData(4, ref useGpu);

        bool samePacked = meshesBResolved == null;
        List<Mesh> meshesBEffective = meshesBResolved ?? meshesA;

        if (!TryPackMeshList(meshesA, out float[] ax, out float[] ay, out float[] az, out int[] triA, out int[] meshTriStartA, out int nVertA, out int nTriA, out int nMeshA))
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Failed to pack MeshesA (need triangle meshes).");
            return;
        }

        if (!TryPackMeshList(meshesBEffective, out float[] bx, out float[] by, out float[] bz, out int[] triB, out int[] meshTriStartB, out int nVertB, out int nTriB, out int nMeshB))
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Failed to pack MeshesB (need triangle meshes).");
            return;
        }

        long triWork = CountTrianglePairTests(nMeshA, nMeshB, meshTriStartA, meshTriStartB, samePacked, skipIntraMeshPair);
        if (triWork > MaxTrianglePairTests)
        {
            AddRuntimeMessage(
                GH_RuntimeMessageLevel.Error,
                $"Too many triangle pair tests ({triWork}); limit is {MaxTrianglePairTests}. Reduce mesh counts or face counts.");
            return;
        }

        int sameFlag = samePacked ? 1 : 0;
        int skipFlag = skipIntraMeshPair ? 1 : 0;

        var outMa = new int[maxHits];
        var outMb = new int[maxHits];
        var outTa = new int[maxHits];
        var outTb = new int[maxHits];
        var totalBuf = new int[1];

        bool ranGpu = false;
        if (useGpu && NativeLoader.IsMetalAvailable && MetalSharedContext.TryGetContext(out IntPtr ctx))
        {
            int code = MetalBridge.MeshBatchTriangleHits(
                ctx,
                ax,
                ay,
                az,
                triA,
                meshTriStartA,
                nMeshA,
                nVertA,
                nTriA,
                bx,
                by,
                bz,
                triB,
                meshTriStartB,
                nMeshB,
                nVertB,
                nTriB,
                sameFlag,
                skipFlag,
                maxHits,
                outMa,
                outMb,
                outTa,
                outTb,
                totalBuf);
            if (code != 0)
            {
                AddRuntimeMessage(
                    GH_RuntimeMessageLevel.Error,
                    $"Metal batch mesh collision failed with code {code}.");
                return;
            }

            ranGpu = true;
        }

        if (!ranGpu)
        {
            if (useGpu)
            {
                AddRuntimeMessage(
                    GH_RuntimeMessageLevel.Warning,
                    "GPU batch collision did not run — using CPU (same tests).");
            }

            var hitWriteLock = new object();
            RunBatchCpu(
                ax,
                ay,
                az,
                triA,
                meshTriStartA,
                nMeshA,
                bx,
                by,
                bz,
                triB,
                meshTriStartB,
                nMeshB,
                samePacked,
                skipIntraMeshPair,
                maxHits,
                outMa,
                outMb,
                outTa,
                outTb,
                totalBuf,
                hitWriteLock);
        }

        int total = totalBuf[0];
        bool colliding = total > 0;
        int recorded = Math.Min(total, maxHits);

        if (total > maxHits)
        {
            AddRuntimeMessage(
                GH_RuntimeMessageLevel.Remark,
                $"Recorded {maxHits} of {total} hits. Increase MaxHits to store more.");
        }

        var listMa = new List<GH_Integer>(recorded);
        var listMb = new List<GH_Integer>(recorded);
        var listTa = new List<GH_Integer>(recorded);
        var listTb = new List<GH_Integer>(recorded);
        for (int i = 0; i < recorded; i++)
        {
            listMa.Add(new GH_Integer(outMa[i]));
            listMb.Add(new GH_Integer(outMb[i]));
            listTa.Add(new GH_Integer(outTa[i]));
            listTb.Add(new GH_Integer(outTb[i]));
        }

        DA.SetData(0, colliding);
        DA.SetData(1, total);
        DA.SetDataList(2, listMa);
        DA.SetDataList(3, listMb);
        DA.SetDataList(4, listTa);
        DA.SetDataList(5, listTb);
    }

    private bool TryCollectMeshes(IGH_DataAccess DA, int index, string label, out List<Mesh> meshes)
    {
        meshes = new List<Mesh>();
        var gh = new List<GH_Mesh>();
        if (!DA.GetDataList(index, gh) || gh.Count == 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"{label}: need at least one mesh.");
            return false;
        }

        foreach (GH_Mesh? g in gh)
        {
            if (g?.Value != null && g.Value.IsValid)
                meshes.Add(g.Value);
        }

        if (meshes.Count == 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"{label}: no valid meshes.");
            return false;
        }

        return true;
    }

    private static bool TryPackMeshList(
        IReadOnlyList<Mesh> meshes,
        out float[] vx,
        out float[] vy,
        out float[] vz,
        out int[] tri,
        out int[] meshTriStart,
        out int totalVerts,
        out int totalTris,
        out int nMesh)
    {
        vx = vy = vz = Array.Empty<float>();
        tri = Array.Empty<int>();
        meshTriStart = Array.Empty<int>();
        totalVerts = totalTris = nMesh = 0;

        nMesh = meshes.Count;
        if (nMesh < 1)
            return false;

        meshTriStart = new int[nMesh + 1];
        var vxL = new List<float>(256);
        var vyL = new List<float>(256);
        var vzL = new List<float>(256);
        var triL = new List<int>(512);

        int vOff = 0;
        meshTriStart[0] = 0;
        for (int mi = 0; mi < nMesh; mi++)
        {
            if (!MeshTriangleUtils.TryGetTriangleMeshForClosest(meshes[mi], out Mesh work))
                return false;

            if (!work.IsValid || work.Vertices.Count == 0 || work.Faces.Count == 0)
                return false;

            int vc = work.Vertices.Count;
            int fc = work.Faces.Count;
            for (int i = 0; i < vc; i++)
            {
                Point3f p = work.Vertices[i];
                vxL.Add(p.X);
                vyL.Add(p.Y);
                vzL.Add(p.Z);
            }

            for (int fi = 0; fi < fc; fi++)
            {
                MeshFace f = work.Faces[fi];
                if (!f.IsTriangle)
                    return false;
                triL.Add(f.A + vOff);
                triL.Add(f.B + vOff);
                triL.Add(f.C + vOff);
            }

            vOff += vc;
            meshTriStart[mi + 1] = meshTriStart[mi] + fc;
        }

        vx = vxL.ToArray();
        vy = vyL.ToArray();
        vz = vzL.ToArray();
        tri = triL.ToArray();
        totalVerts = vOff;
        totalTris = meshTriStart[nMesh];
        return totalTris > 0 && meshTriStart[nMesh] * 3 == triL.Count;
    }

    private static long CountTrianglePairTests(
        int nMeshA,
        int nMeshB,
        int[] startA,
        int[] startB,
        bool samePacked,
        bool skipIntra)
    {
        long t = 0;
        for (int ma = 0; ma < nMeshA; ma++)
        {
            int na = startA[ma + 1] - startA[ma];
            for (int mb = 0; mb < nMeshB; mb++)
            {
                if (skipIntra && samePacked && ma == mb)
                    continue;

                int nb = startB[mb + 1] - startB[mb];
                if (samePacked && ma == mb)
                    t += (long)na * (na - 1) / 2;
                else
                    t += (long)na * nb;
            }
        }

        return t;
    }

    private static void RunBatchCpu(
        float[] ax,
        float[] ay,
        float[] az,
        int[] triA,
        int[] meshTriStartA,
        int nMeshA,
        float[] bx,
        float[] by,
        float[] bz,
        int[] triB,
        int[] meshTriStartB,
        int nMeshB,
        bool samePacked,
        bool skipIntraMeshPair,
        int maxHits,
        int[] outMa,
        int[] outMb,
        int[] outTa,
        int[] outTb,
        int[] totalBuf,
        object hitWriteLock)
    {
        int nB = nMeshB;
        var totalHitsBox = new int[1];

        Parallel.For(0, nMeshA * nB, gid =>
        {
            int ma = gid / nB;
            int mb = gid % nB;

            if (skipIntraMeshPair && samePacked && ma == mb)
                return;

            int t0a = meshTriStartA[ma];
            int t1a = meshTriStartA[ma + 1];
            int t0b = meshTriStartB[mb];
            int t1b = meshTriStartB[mb + 1];
            int na = t1a - t0a;
            int nb = t1b - t0b;

            bool intra = samePacked && ma == mb;

            if (intra)
            {
                for (int ta = 0; ta < na; ta++)
                {
                    for (int tb = ta + 1; tb < na; tb++)
                    {
                        int gta = t0a + ta;
                        int gtb = t0a + tb;
                        TryRecordHit(
                            ax,
                            ay,
                            az,
                            triA,
                            gta,
                            ax,
                            ay,
                            az,
                            triA,
                            gtb,
                            ma,
                            mb,
                            ta,
                            tb,
                            maxHits,
                            totalHitsBox,
                            outMa,
                            outMb,
                            outTa,
                            outTb,
                            hitWriteLock);
                    }
                }
            }
            else
            {
                for (int ta = 0; ta < na; ta++)
                {
                    for (int tb = 0; tb < nb; tb++)
                    {
                        int gta = t0a + ta;
                        int gtb = t0b + tb;
                        TryRecordHit(
                            ax,
                            ay,
                            az,
                            triA,
                            gta,
                            bx,
                            by,
                            bz,
                            triB,
                            gtb,
                            ma,
                            mb,
                            ta,
                            tb,
                            maxHits,
                            totalHitsBox,
                            outMa,
                            outMb,
                            outTa,
                            outTb,
                            hitWriteLock);
                    }
                }
            }
        });

        totalBuf[0] = totalHitsBox[0];
    }

    private static void TryRecordHit(
        float[] ax,
        float[] ay,
        float[] az,
        int[] triA,
        int gta,
        float[] bx,
        float[] by,
        float[] bz,
        int[] triB,
        int gtb,
        int ma,
        int mb,
        int localTa,
        int localTb,
        int maxHits,
        int[] totalHitsBox,
        int[] outMa,
        int[] outMb,
        int[] outTa,
        int[] outTb,
        object hitWriteLock)
    {
        var triDa = new TriD(ax, ay, az, triA, gta);
        var triDb = new TriD(bx, by, bz, triB, gtb);

        if (!AabbOverlap(triDa, triDb))
            return;

        if (!TrianglesIntersectSat(triDa, triDb))
            return;

        int t = Interlocked.Increment(ref totalHitsBox[0]);
        if (t <= maxHits)
        {
            lock (hitWriteLock)
            {
                outMa[t - 1] = ma;
                outMb[t - 1] = mb;
                outTa[t - 1] = localTa;
                outTb[t - 1] = localTb;
            }
        }
    }

    private static bool AabbOverlap(TriD a, TriD b)
    {
        double minAx = Math.Min(Math.Min(a.X0, a.X1), a.X2);
        double maxAx = Math.Max(Math.Max(a.X0, a.X1), a.X2);
        double minAy = Math.Min(Math.Min(a.Y0, a.Y1), a.Y2);
        double maxAy = Math.Max(Math.Max(a.Y0, a.Y1), a.Y2);
        double minAz = Math.Min(Math.Min(a.Z0, a.Z1), a.Z2);
        double maxAz = Math.Max(Math.Max(a.Z0, a.Z1), a.Z2);

        double minBx = Math.Min(Math.Min(b.X0, b.X1), b.X2);
        double maxBx = Math.Max(Math.Max(b.X0, b.X1), b.X2);
        double minBy = Math.Min(Math.Min(b.Y0, b.Y1), b.Y2);
        double maxBy = Math.Max(Math.Max(b.Y0, b.Y1), b.Y2);
        double minBz = Math.Min(Math.Min(b.Z0, b.Z1), b.Z2);
        double maxBz = Math.Max(Math.Max(b.Z0, b.Z1), b.Z2);

        return maxAx >= minBx && maxBx >= minAx && maxAy >= minBy && maxBy >= minAy && maxAz >= minBz && maxBz >= minAz;
    }

    private static void Cross(double ax, double ay, double az, double bx, double by, double bz, out double ox, out double oy, out double oz)
    {
        ox = ay * bz - az * by;
        oy = az * bx - ax * bz;
        oz = ax * by - ay * bx;
    }

    private static double DotP(double ax, double ay, double az, double px, double py, double pz) => ax * px + ay * py + az * pz;

    private static bool SeparatedByAxis(double ax, double ay, double az, TriD a, TriD b)
    {
        double al2 = ax * ax + ay * ay + az * az;
        if (al2 < 1e-60)
            return false;
        double p0 = DotP(ax, ay, az, a.X0, a.Y0, a.Z0);
        double p1 = DotP(ax, ay, az, a.X1, a.Y1, a.Z1);
        double p2 = DotP(ax, ay, az, a.X2, a.Y2, a.Z2);
        double minA = Math.Min(Math.Min(p0, p1), p2);
        double maxA = Math.Max(Math.Max(p0, p1), p2);
        double q0 = DotP(ax, ay, az, b.X0, b.Y0, b.Z0);
        double q1 = DotP(ax, ay, az, b.X1, b.Y1, b.Z1);
        double q2 = DotP(ax, ay, az, b.X2, b.Y2, b.Z2);
        double minB = Math.Min(Math.Min(q0, q1), q2);
        double maxB = Math.Max(Math.Max(q0, q1), q2);
        return maxA < minB || maxB < minA;
    }

    private static bool TrianglesIntersectSat(TriD a, TriD b)
    {
        double e1ax = a.X1 - a.X0, e1ay = a.Y1 - a.Y0, e1az = a.Z1 - a.Z0;
        double e1bx = a.X2 - a.X1, e1by = a.Y2 - a.Y1, e1bz = a.Z2 - a.Z1;
        double e1cx = a.X0 - a.X2, e1cy = a.Y0 - a.Y2, e1cz = a.Z0 - a.Z2;

        double e2ax = b.X1 - b.X0, e2ay = b.Y1 - b.Y0, e2az = b.Z1 - b.Z0;
        double e2bx = b.X2 - b.X1, e2by = b.Y2 - b.Y1, e2bz = b.Z2 - b.Z1;
        double e2cx = b.X0 - b.X2, e2cy = b.Y0 - b.Y2, e2cz = b.Z0 - b.Z2;

        Cross(e1ax, e1ay, e1az, e1bx, e1by, e1bz, out double n1x, out double n1y, out double n1z);
        Cross(e2ax, e2ay, e2az, e2bx, e2by, e2bz, out double n2x, out double n2y, out double n2z);

        if (n1x * n1x + n1y * n1y + n1z * n1z < 1e-60 || n2x * n2x + n2y * n2y + n2z * n2z < 1e-60)
            return false;

        if (SeparatedByAxis(n1x, n1y, n1z, a, b))
            return false;
        if (SeparatedByAxis(n2x, n2y, n2z, a, b))
            return false;

        Span<double> e1xs = stackalloc double[3];
        Span<double> e1ys = stackalloc double[3];
        Span<double> e1zs = stackalloc double[3];
        Span<double> e2xs = stackalloc double[3];
        Span<double> e2ys = stackalloc double[3];
        Span<double> e2zs = stackalloc double[3];
        e1xs[0] = e1ax;
        e1xs[1] = e1bx;
        e1xs[2] = e1cx;
        e1ys[0] = e1ay;
        e1ys[1] = e1by;
        e1ys[2] = e1cy;
        e1zs[0] = e1az;
        e1zs[1] = e1bz;
        e1zs[2] = e1cz;
        e2xs[0] = e2ax;
        e2xs[1] = e2bx;
        e2xs[2] = e2cx;
        e2ys[0] = e2ay;
        e2ys[1] = e2by;
        e2ys[2] = e2cy;
        e2zs[0] = e2az;
        e2zs[1] = e2bz;
        e2zs[2] = e2cz;

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Cross(e1xs[i], e1ys[i], e1zs[i], e2xs[j], e2ys[j], e2zs[j], out double cxx, out double cxy, out double cxz);
                if (cxx * cxx + cxy * cxy + cxz * cxz < 1e-60)
                    continue;
                if (SeparatedByAxis(cxx, cxy, cxz, a, b))
                    return false;
            }
        }

        return true;
    }

    protected override Bitmap Icon => ComponentIcons24.MeshCollision;

    public override Guid ComponentGuid => new("7a2e9c41-b0d3-4f8e-9c12-6d5e8f1a2b3c");
}
