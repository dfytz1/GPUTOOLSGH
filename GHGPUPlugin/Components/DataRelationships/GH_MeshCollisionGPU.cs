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
/// Triangle–triangle intersection between two meshes (or self) on Metal, with CPU fallback.
/// The Metal kernel applies a per-pair AABB test before SAT (no mesh-level BVH yet).
/// </summary>
public class GH_MeshCollisionGPU : GH_Component
{
    private const long MaxPairTests = 50_000_000L;

    private readonly struct TriD
    {
        internal readonly double X0, Y0, Z0, X1, Y1, Z1, X2, Y2, Z2;

        internal TriD(float[] px, float[] py, float[] pz, int[] tri, int ti)
        {
            int o = ti * 3;
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
            "Test triangle–triangle intersection between two triangle meshes (or one mesh against itself). "
                + "GPU tests use an axis-aligned bounding box per triangle pair before SAT. "
                + "Reports whether any pair intersects and records up to MaxHits intersecting face index pairs.",
            "GPUTools",
            "Mesh")
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddMeshParameter("MeshA", "A", "First triangle mesh (quads are triangulated once).", GH_ParamAccess.item);
        pManager.AddMeshParameter("MeshB", "B", "Second mesh; leave empty for self-collision on MeshA.", GH_ParamAccess.item);
        pManager.AddIntegerParameter("MaxHits", "M", "Maximum intersecting triangle pairs to record in outputs.", GH_ParamAccess.item, 4096);
        pManager.AddBooleanParameter("SkipSameTriangleIndex", "Skip", "For A×B mode, skip pairs with the same triangle index (e.g. duplicated mesh).", GH_ParamAccess.item, true);
        pManager.AddBooleanParameter("UseGPU", "GPU", "Use Metal when available.", GH_ParamAccess.item, true);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddBooleanParameter("Colliding", "C", "True if at least one triangle pair intersects.", GH_ParamAccess.item);
        pManager.AddIntegerParameter("HitCount", "N", "Total intersecting pairs found (can exceed MaxHits).", GH_ParamAccess.item);
        pManager.AddIntegerParameter("TriangleIndexA", "iA", "Triangle index on mesh A for each recorded hit.", GH_ParamAccess.list);
        pManager.AddIntegerParameter("TriangleIndexB", "iB", "Triangle index on mesh B (or A in self mode) for each recorded hit.", GH_ParamAccess.list);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        NativeLoader.EnsureLoaded();

        Mesh? meshA = null;
        if (!DA.GetData(0, ref meshA) || meshA == null)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "MeshA is required.");
            return;
        }

        Mesh? meshB = null;
        bool hasB = DA.GetData(1, ref meshB) && meshB != null;

        int maxHits = 4096;
        DA.GetData(2, ref maxHits);
        if (maxHits < 1)
            maxHits = 1;

        bool skipSame = true;
        DA.GetData(3, ref skipSame);

        bool useGpu = true;
        DA.GetData(4, ref useGpu);

        if (!MeshTriangleUtils.TryGetTriangleMeshForClosest(meshA, out Mesh workA))
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "MeshA could not be reduced to triangles.");
            return;
        }

        Mesh workB;
        bool self;
        if (!hasB)
        {
            workB = workA;
            self = true;
        }
        else
        {
            if (!MeshTriangleUtils.TryGetTriangleMeshForClosest(meshB!, out workB!))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "MeshB could not be reduced to triangles.");
                return;
            }

            self = false;
        }

        if (!TryBuildTriSoa(workA, out float[] ax, out float[] ay, out float[] az, out int[] triA, out int nVa, out int nTa))
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "MeshA is not usable.");
            return;
        }

        if (!TryBuildTriSoa(workB, out float[] bx, out float[] by, out float[] bz, out int[] triB, out int nVb, out int nTb))
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "MeshB is not usable.");
            return;
        }

        long pairs = self ? (long)nTa * (nTa - 1) / 2 : (long)nTa * nTb;
        if (pairs > MaxPairTests)
        {
            AddRuntimeMessage(
                GH_RuntimeMessageLevel.Error,
                $"Too many triangle pairs ({pairs}); limit is {MaxPairTests}. Simplify meshes or use a broad-phase externally.");
            return;
        }

        int skipFlag = (!self && skipSame) ? 1 : 0;
        int selfFlag = self ? 1 : 0;

        var outIa = new int[maxHits];
        var outIb = new int[maxHits];
        var totalBuf = new int[1];

        bool ranGpu = false;
        if (useGpu && NativeLoader.IsMetalAvailable && MetalSharedContext.TryGetContext(out IntPtr ctx))
        {
            int code = MetalBridge.MeshMeshTriangleHits(
                ctx,
                ax,
                ay,
                az,
                triA,
                nVa,
                nTa,
                bx,
                by,
                bz,
                triB,
                nVb,
                nTb,
                selfFlag,
                skipFlag,
                maxHits,
                outIa,
                outIb,
                totalBuf);
            if (code != 0)
            {
                AddRuntimeMessage(
                    GH_RuntimeMessageLevel.Error,
                    $"Metal mesh collision failed with code {code}.");
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
                    "GPU mesh collision did not run — using CPU (same tests).");
            }

            var hitWriteLock = new object();
            RunCpuTriangleHits(
                ax,
                ay,
                az,
                triA,
                nTa,
                bx,
                by,
                bz,
                triB,
                nTb,
                self,
                skipFlag != 0,
                maxHits,
                outIa,
                outIb,
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
                $"Recorded {maxHits} of {total} intersecting triangle pairs. Increase MaxHits to store more.");
        }

        var listA = new List<GH_Integer>(recorded);
        var listB = new List<GH_Integer>(recorded);
        for (int i = 0; i < recorded; i++)
        {
            listA.Add(new GH_Integer(outIa[i]));
            listB.Add(new GH_Integer(outIb[i]));
        }

        DA.SetData(0, colliding);
        DA.SetData(1, total);
        DA.SetDataList(2, listA);
        DA.SetDataList(3, listB);
    }

    private static bool TryBuildTriSoa(Mesh work, out float[] vx, out float[] vy, out float[] vz, out int[] triIdx, out int vCount, out int triCount)
    {
        vx = vy = vz = Array.Empty<float>();
        triIdx = Array.Empty<int>();
        vCount = triCount = 0;
        if (!work.IsValid || work.Vertices.Count == 0 || work.Faces.Count == 0)
            return false;

        vCount = work.Vertices.Count;
        triCount = work.Faces.Count;
        vx = new float[vCount];
        vy = new float[vCount];
        vz = new float[vCount];
        for (int i = 0; i < vCount; i++)
        {
            Point3f p = work.Vertices[i];
            vx[i] = p.X;
            vy[i] = p.Y;
            vz[i] = p.Z;
        }

        triIdx = new int[triCount * 3];
        for (int fi = 0; fi < triCount; fi++)
        {
            MeshFace f = work.Faces[fi];
            if (!f.IsTriangle)
                return false;
            triIdx[3 * fi] = f.A;
            triIdx[3 * fi + 1] = f.B;
            triIdx[3 * fi + 2] = f.C;
        }

        return true;
    }

    private static void RunCpuTriangleHits(
        float[] ax,
        float[] ay,
        float[] az,
        int[] triA,
        int nTa,
        float[] bx,
        float[] by,
        float[] bz,
        int[] triB,
        int nTb,
        bool self,
        bool skipSame,
        int maxHits,
        int[] outIa,
        int[] outIb,
        int[] totalBuf,
        object hitWriteLock)
    {
        int np = self ? checked(nTa * (nTa - 1) / 2) : checked(nTa * nTb);
        int totalHits = 0;

        Parallel.For(0, np, gid =>
        {
            int ia;
            int ib;
            if (self)
                DecodeTriPair(gid, nTa, out ia, out ib);
            else
            {
                ia = gid / nTb;
                ib = gid % nTb;
                if (skipSame && ia == ib)
                    return;
            }

            var ta = new TriD(ax, ay, az, triA, ia);
            var tb = new TriD(bx, by, bz, triB, ib);

            if (!AabbOverlap(ta, tb))
                return;

            if (!TrianglesIntersectSat(ta, tb))
                return;

            int t = Interlocked.Increment(ref totalHits);
            if (t <= maxHits)
            {
                lock (hitWriteLock)
                {
                    outIa[t - 1] = ia;
                    outIb[t - 1] = ib;
                }
            }
        });

        totalBuf[0] = totalHits;
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

    private static bool SeparatedByAxis(
        double ax,
        double ay,
        double az,
        TriD a,
        TriD b)
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

    private static void DecodeTriPair(int k, int n, out int i, out int j)
    {
        int lo = 0;
        int hi = n - 2;
        while (lo < hi)
        {
            int mid = (lo + hi + 1) >> 1;
            int baseMid = mid * (2 * n - mid - 1) / 2;
            if (baseMid <= k)
                lo = mid;
            else
                hi = mid - 1;
        }

        i = lo;
        int baseI = i * (2 * n - i - 1) / 2;
        j = i + 1 + (k - baseI);
    }

    protected override Bitmap Icon => ComponentIcons24.MeshCollision;

    public override Guid ComponentGuid => new("7a2e9c41-b0d3-4f8e-9c12-6d5e8f1a2b3c");
}
