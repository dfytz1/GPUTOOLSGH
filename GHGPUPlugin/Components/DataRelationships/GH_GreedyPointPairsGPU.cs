using System.Drawing;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;
using GHGPUPlugin.NativeInterop;

namespace GHGPUPlugin.Components.DataRelationships;

/// <summary>Greedy shortest-edge matching with optional 2-opt; GPU fills pairwise distances when pair count is moderate.</summary>
public class GH_GreedyPointPairsGPU : GH_Component
{
    private const long MaxGpuPairs = 10_000_000;

    private struct Edge
    {
        internal double Dist;
        internal int A;
        internal int B;
    }

    public GH_GreedyPointPairsGPU()
        : base(
            "Greedy Point Pairs GPU",
            "GreedyPtPairsGPU",
            "Pair points into disjoint edges by ascending distance (greedy), then optional 2-opt. "
                + "Uses Metal for all pairwise squared distances when n is moderate; otherwise RTree radius search.",
            "GPUTools",
            "Point")
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddPointParameter("Points", "P", "Point cloud to pair.", GH_ParamAccess.list);
        pManager.AddNumberParameter("MaxDistance", "D", "Maximum edge length; non-positive = unlimited.", GH_ParamAccess.item, 0);
        pManager.AddIntegerParameter("MaxIterations", "N", "2-opt outer iterations (non-positive defaults to 100).", GH_ParamAccess.item, 100);
        pManager.AddBooleanParameter("UseGPU", "GPU", "Use Metal for pairwise distances when eligible.", GH_ParamAccess.item, true);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddLineParameter("Lines", "L", "One line per matched pair.", GH_ParamAccess.list);
        pManager.AddIntegerParameter("IndicesA", "A", "First point index per pair.", GH_ParamAccess.list);
        pManager.AddIntegerParameter("IndicesB", "B", "Second point index per pair.", GH_ParamAccess.list);
        pManager.AddNumberParameter("Distances", "Dist", "Edge length per pair.", GH_ParamAccess.list);
        pManager.AddPointParameter("Unmatched", "U", "Points not in any pair.", GH_ParamAccess.list);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        NativeLoader.EnsureLoaded();

        var points = new List<Point3d>();
        if (!DA.GetDataList(0, points) || points.Count < 2)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Need at least 2 points.");
            return;
        }

        double maxDist = 0;
        DA.GetData(1, ref maxDist);

        int maxIterations = 100;
        DA.GetData(2, ref maxIterations);
        if (maxIterations < 1)
            maxIterations = 100;

        bool useGpu = true;
        DA.GetData(3, ref useGpu);

        int n = points.Count;
        long pairCountLong = (long)n * (n - 1) / 2;
        double maxD2 = maxDist > 0 ? maxDist * maxDist : double.MaxValue;

        List<int> sortedPairIndices;
        if (useGpu
            && NativeLoader.IsMetalAvailable
            && pairCountLong > 0
            && pairCountLong <= MaxGpuPairs
            && MetalSharedContext.TryGetContext(out IntPtr ctx))
        {
            var x = new float[n];
            var y = new float[n];
            var z = new float[n];
            for (int i = 0; i < n; i++)
            {
                x[i] = (float)points[i].X;
                y[i] = (float)points[i].Y;
                z[i] = (float)points[i].Z;
            }

            var distSq = new float[pairCountLong];
            int code = MetalBridge.PairwiseUpperDistSq(ctx, x, y, z, n, distSq);
            if (code != 0)
            {
                AddRuntimeMessage(
                    GH_RuntimeMessageLevel.Error,
                    $"Metal pairwise distances failed with code {code}.");
                return;
            }

            sortedPairIndices = new List<int>((int)pairCountLong);
            for (int k = 0; k < (int)pairCountLong; k++)
            {
                if (maxDist <= 0 || distSq[k] <= maxD2)
                    sortedPairIndices.Add(k);
            }

            sortedPairIndices.Sort((a, b) => distSq[a].CompareTo(distSq[b]));
            AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, "Candidate pairs (GPU): " + sortedPairIndices.Count);
        }
        else
        {
            if (useGpu && pairCountLong > MaxGpuPairs)
            {
                AddRuntimeMessage(
                    GH_RuntimeMessageLevel.Remark,
                    "Point count yields too many pairs for GPU path; using RTree radius search.");
            }
            else if (useGpu && !NativeLoader.IsMetalAvailable)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Metal unavailable; using RTree radius search.");
            }
            else if (useGpu && pairCountLong <= MaxGpuPairs && NativeLoader.IsMetalAvailable)
            {
                AddRuntimeMessage(
                    GH_RuntimeMessageLevel.Warning,
                    "Metal context unavailable; using RTree radius search.");
            }

            sortedPairIndices = BuildCandidatesRTree(points, n, maxDist);
        }

        if (sortedPairIndices.Count == 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "No pairs found.");
            return;
        }

        var pairA = new List<int>();
        var pairB = new List<int>();
        bool[] used = new bool[n];
        foreach (int k in sortedPairIndices)
        {
            if (k < 0 || k >= pairCountLong)
                continue;
            DecodePairIndex(k, n, out int a, out int b);
            if (used[a] || used[b])
                continue;
            used[a] = true;
            used[b] = true;
            pairA.Add(a);
            pairB.Add(b);
        }

        int optIters = RunTwoOpt(points, maxDist, maxIterations, pairA, pairB);
        AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, "2-opt iterations: " + optIters);

        var outLines = new List<Line>();
        var outA = new List<int>();
        var outB = new List<int>();
        var outDist = new List<double>();
        var matched = new bool[n];

        for (int i = 0; i < pairA.Count; i++)
        {
            int a = pairA[i];
            int b = pairB[i];
            double d = points[a].DistanceTo(points[b]);
            outLines.Add(new Line(points[a], points[b]));
            outA.Add(a);
            outB.Add(b);
            outDist.Add(d);
            matched[a] = true;
            matched[b] = true;
        }

        var unmatchedPts = new List<Point3d>();
        for (int i = 0; i < n; i++)
        {
            if (!matched[i])
                unmatchedPts.Add(points[i]);
        }

        AddRuntimeMessage(GH_RuntimeMessageLevel.Remark, "Unmatched points: " + unmatchedPts.Count);

        DA.SetDataList(0, outLines);
        DA.SetDataList(1, outA);
        DA.SetDataList(2, outB);
        DA.SetDataList(3, outDist);
        DA.SetDataList(4, unmatchedPts);
    }

    private List<int> BuildCandidatesRTree(List<Point3d> points, int n, double maxDist)
    {
        var candidates = new List<Edge>(n * 4);
        var tree = new RTree();
        for (int i = 0; i < n; i++)
            tree.Insert(points[i], i);

        double searchRadius = maxDist > 0 ? maxDist : double.MaxValue;

        for (int i = 0; i < n; i++)
        {
            Sphere sphere = new Sphere(points[i], searchRadius);
            tree.Search(sphere, (_, args) =>
            {
                int j = args.Id;
                if (j <= i)
                    return;
                double d = points[i].DistanceTo(points[j]);
                if (maxDist <= 0 || d <= maxDist)
                    candidates.Add(new Edge { Dist = d, A = i, B = j });
            });
        }

        if (candidates.Count == 0)
            return new List<int>();

        candidates.Sort((x, y) => x.Dist.CompareTo(y.Dist));

        var fake = new List<int>(candidates.Count);
        long pc = (long)n * (n - 1) / 2;
        foreach (var e in candidates)
        {
            int k = EncodePairIndex(e.A, e.B, n);
            if (k >= 0 && k < pc)
                fake.Add(k);
        }

        return fake;
    }

    private static int RunTwoOpt(
        List<Point3d> points,
        double maxDist,
        int maxIterations,
        List<int> pairA,
        List<int> pairB)
    {
        bool improved = true;
        int iter = 0;

        while (improved && iter < maxIterations)
        {
            improved = false;
            iter++;

            for (int i = 0; i < pairA.Count; i++)
            {
                for (int j = i + 1; j < pairA.Count; j++)
                {
                    int a = pairA[i];
                    int b = pairB[i];
                    int c = pairA[j];
                    int d = pairB[j];

                    double current =
                        points[a].DistanceTo(points[b]) +
                        points[c].DistanceTo(points[d]);

                    double opt1 =
                        points[a].DistanceTo(points[c]) +
                        points[b].DistanceTo(points[d]);

                    double opt2 =
                        points[a].DistanceTo(points[d]) +
                        points[b].DistanceTo(points[c]);

                    if (opt1 < current && opt1 <= opt2)
                    {
                        if (maxDist <= 0
                            || (points[a].DistanceTo(points[c]) <= maxDist
                                && points[b].DistanceTo(points[d]) <= maxDist))
                        {
                            pairA[i] = a;
                            pairB[i] = c;
                            pairA[j] = b;
                            pairB[j] = d;
                            improved = true;
                        }
                    }
                    else if (opt2 < current)
                    {
                        if (maxDist <= 0
                            || (points[a].DistanceTo(points[d]) <= maxDist
                                && points[b].DistanceTo(points[c]) <= maxDist))
                        {
                            pairA[i] = a;
                            pairB[i] = d;
                            pairA[j] = b;
                            pairB[j] = c;
                            improved = true;
                        }
                    }
                }
            }
        }

        return iter;
    }

    private static void DecodePairIndex(int k, int n, out int i, out int j)
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

    private static int EncodePairIndex(int i, int j, int n)
    {
        if (i > j)
            (i, j) = (j, i);
        if (i == j)
            return -1;
        return i * (2 * n - i - 1) / 2 + (j - i - 1);
    }

    protected override Bitmap Icon => null!;

    public override Guid ComponentGuid => new("c4d8e1f2-9a3b-4c5d-8e6f-0a1b2c3d4e5f");
}
