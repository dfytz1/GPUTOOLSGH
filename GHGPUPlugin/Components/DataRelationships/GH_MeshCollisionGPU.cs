using System.Drawing;
using System.Threading;
using System.Threading.Tasks;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;
using GHGPUPlugin.NativeInterop;

namespace GHGPUPlugin.Components.DataRelationships;

/// <summary>Brute-force triangle–triangle intersection between two meshes (or self) on Metal, with CPU fallback.</summary>
public class GH_MeshCollisionGPU : GH_Component
{
    private const long MaxPairTests = 50_000_000L;

    public GH_MeshCollisionGPU()
        : base(
            "Mesh Collision GPU",
            "MeshHitGPU",
            "Test triangle–triangle intersection between two triangle meshes (or one mesh against itself). "
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
        pManager.AddIntegerParameter("TriangleA", "iA", "Triangle index on mesh A for each recorded hit.", GH_ParamAccess.list);
        pManager.AddIntegerParameter("TriangleB", "iB", "Triangle index on mesh B (or A in self mode) for each recorded hit.", GH_ParamAccess.list);
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

        if (!GH_ClosestPointGPU.TryGetTriangleMeshForClosest(meshA, out Mesh workA))
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
            if (!GH_ClosestPointGPU.TryGetTriangleMeshForClosest(meshB!, out workB!))
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
                totalBuf);
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
        int[] totalBuf)
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

            GetTri(ax, ay, az, triA, ia, out Point3d a0, out Point3d a1, out Point3d a2);
            GetTri(bx, by, bz, triB, ib, out Point3d b0, out Point3d b1, out Point3d b2);

            if (!AabbOverlap(a0, a1, a2, b0, b1, b2))
                return;

            if (!TrianglesIntersectSat(a0, a1, a2, b0, b1, b2))
                return;

            int t = Interlocked.Increment(ref totalHits);
            if (t <= maxHits)
            {
                outIa[t - 1] = ia;
                outIb[t - 1] = ib;
            }
        });

        totalBuf[0] = totalHits;
    }

    private static void GetTri(
        float[] px,
        float[] py,
        float[] pz,
        int[] tri,
        int ti,
        out Point3d a,
        out Point3d b,
        out Point3d c)
    {
        int o = ti * 3;
        int i0 = tri[o];
        int i1 = tri[o + 1];
        int i2 = tri[o + 2];
        a = new Point3d(px[i0], py[i0], pz[i0]);
        b = new Point3d(px[i1], py[i1], pz[i1]);
        c = new Point3d(px[i2], py[i2], pz[i2]);
    }

    private static bool AabbOverlap(Point3d a0, Point3d a1, Point3d a2, Point3d b0, Point3d b1, Point3d b2)
    {
        double minAx = Math.Min(Math.Min(a0.X, a1.X), a2.X);
        double maxAx = Math.Max(Math.Max(a0.X, a1.X), a2.X);
        double minAy = Math.Min(Math.Min(a0.Y, a1.Y), a2.Y);
        double maxAy = Math.Max(Math.Max(a0.Y, a1.Y), a2.Y);
        double minAz = Math.Min(Math.Min(a0.Z, a1.Z), a2.Z);
        double maxAz = Math.Max(Math.Max(a0.Z, a1.Z), a2.Z);

        double minBx = Math.Min(Math.Min(b0.X, b1.X), b2.X);
        double maxBx = Math.Max(Math.Max(b0.X, b1.X), b2.X);
        double minBy = Math.Min(Math.Min(b0.Y, b1.Y), b2.Y);
        double maxBy = Math.Max(Math.Max(b0.Y, b1.Y), b2.Y);
        double minBz = Math.Min(Math.Min(b0.Z, b1.Z), b2.Z);
        double maxBz = Math.Max(Math.Max(b0.Z, b1.Z), b2.Z);

        return maxAx >= minBx && maxBx >= minAx && maxAy >= minBy && maxBy >= minAy && maxAz >= minBz && maxBz >= minAz;
    }

    private static double Dot(Vector3d axis, Point3d p) => axis.X * p.X + axis.Y * p.Y + axis.Z * p.Z;

    private static bool SeparatedByAxis(
        Vector3d axis,
        Point3d a0,
        Point3d a1,
        Point3d a2,
        Point3d b0,
        Point3d b1,
        Point3d b2)
    {
        if (axis.SquareLength < 1e-60)
            return false;
        double p0 = Dot(axis, a0);
        double p1 = Dot(axis, a1);
        double p2 = Dot(axis, a2);
        double minA = Math.Min(Math.Min(p0, p1), p2);
        double maxA = Math.Max(Math.Max(p0, p1), p2);
        double q0 = Dot(axis, b0);
        double q1 = Dot(axis, b1);
        double q2 = Dot(axis, b2);
        double minB = Math.Min(Math.Min(q0, q1), q2);
        double maxB = Math.Max(Math.Max(q0, q1), q2);
        return maxA < minB || maxB < minA;
    }

    private static bool TrianglesIntersectSat(Point3d a0, Point3d a1, Point3d a2, Point3d b0, Point3d b1, Point3d b2)
    {
        Vector3d n1 = Vector3d.CrossProduct(a1 - a0, a2 - a0);
        Vector3d n2 = Vector3d.CrossProduct(b1 - b0, b2 - b0);
        if (n1.SquareLength < 1e-60 || n2.SquareLength < 1e-60)
            return false;

        if (SeparatedByAxis(n1, a0, a1, a2, b0, b1, b2))
            return false;
        if (SeparatedByAxis(n2, a0, a1, a2, b0, b1, b2))
            return false;

        Vector3d[] e1 = { a1 - a0, a2 - a1, a0 - a2 };
        Vector3d[] e2 = { b1 - b0, b2 - b1, b0 - b2 };
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Vector3d ax = Vector3d.CrossProduct(e1[i], e2[j]);
                if (ax.SquareLength < 1e-60)
                    continue;
                if (SeparatedByAxis(ax, a0, a1, a2, b0, b1, b2))
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

    protected override Bitmap Icon => null!;

    public override Guid ComponentGuid => new("7a2e9c41-b0d3-4f8e-9c12-6d5e8f1a2b3c");
}
