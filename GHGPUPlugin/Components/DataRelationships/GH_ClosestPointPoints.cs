using System.Threading.Tasks;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;
using GHGPUPlugin.NativeInterop;
using System.Drawing;

namespace GHGPUPlugin.Components.DataRelationships;

/// <summary>For each query, closest point in a target point set (brute force; Metal optional).</summary>
public class GH_ClosestPointPoints : GH_Component
{
    public GH_ClosestPointPoints()
        : base(
            "Closest Point Points GPU",
            "CptPtsGPU",
            "For each query point, find the closest target point (Euclidean). Optional Metal brute force.",
            "GPUTools",
            "Point")
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddPointParameter("QueryPoints", "QueryPoints", "Points to test.", GH_ParamAccess.list);
        pManager.AddPointParameter("TargetPoints", "TargetPoints", "Pool of candidate points.", GH_ParamAccess.list);
        pManager.AddBooleanParameter("UseGPU", "UseGPU", "Use Metal when available.", GH_ParamAccess.item, true);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddPointParameter("ClosestPoint", "ClosestPoint", "Closest target point for each query.", GH_ParamAccess.list);
        pManager.AddNumberParameter("Distance", "Distance", "Distance to closest target.", GH_ParamAccess.list);
        pManager.AddIntegerParameter("TargetIndex", "TargetIndex", "Index of closest point in TargetPoints list.", GH_ParamAccess.list);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        NativeLoader.EnsureLoaded();

        var queries = new List<Point3d>();
        if (!DA.GetDataList("QueryPoints", queries) || queries.Count == 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "No query points provided.");
            return;
        }

        var targets = new List<Point3d>();
        if (!DA.GetDataList("TargetPoints", targets) || targets.Count == 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "No target points provided.");
            return;
        }

        bool useGpu = true;
        DA.GetData("UseGPU", ref useGpu);

        int qn = queries.Count;
        int tn = targets.Count;

        var qx = new float[qn];
        var qy = new float[qn];
        var qz = new float[qn];
        for (int i = 0; i < qn; i++)
        {
            qx[i] = (float)queries[i].X;
            qy[i] = (float)queries[i].Y;
            qz[i] = (float)queries[i].Z;
        }

        var px = new float[tn];
        var py = new float[tn];
        var pz = new float[tn];
        for (int i = 0; i < tn; i++)
        {
            px[i] = (float)targets[i].X;
            py[i] = (float)targets[i].Y;
            pz[i] = (float)targets[i].Z;
        }

        var outCx = new float[qn];
        var outCy = new float[qn];
        var outCz = new float[qn];
        var outD2 = new float[qn];
        var outIdx = new int[qn];

        bool ranGpu = false;
        if (useGpu)
        {
            if (!MetalGuard.EnsureReady(this))
                return;

            MetalSharedContext.TryGetContext(out IntPtr ctx);
            int code = MetalBridge.ClosestPointsCloud(
                ctx,
                qx,
                qy,
                qz,
                qn,
                px,
                py,
                pz,
                tn,
                outCx,
                outCy,
                outCz,
                outD2,
                outIdx);
            if (code != 0)
            {
                AddRuntimeMessage(
                    GH_RuntimeMessageLevel.Error,
                    $"Metal closest-point cloud failed with code {code}.");
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
                    "GPU search did not run — using CPU parallel search.");
            }

            var opts = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };
            Parallel.For(0, qn, opts, qi =>
            {
                double best = double.MaxValue;
                int bestJ = -1;
                Point3d bestP = Point3d.Unset;
                Point3d q = queries[qi];
                for (int j = 0; j < tn; j++)
                {
                    double d2 = q.DistanceToSquared(targets[j]);
                    if (d2 < best)
                    {
                        best = d2;
                        bestJ = j;
                        bestP = targets[j];
                    }
                }

                outCx[qi] = (float)bestP.X;
                outCy[qi] = (float)bestP.Y;
                outCz[qi] = (float)bestP.Z;
                outD2[qi] = (float)best;
                outIdx[qi] = bestJ;
            });
        }

        var pts = new List<GH_Point>(qn);
        var dists = new List<GH_Number>(qn);
        var idxGh = new List<GH_Integer>(qn);
        for (int i = 0; i < qn; i++)
        {
            pts.Add(new GH_Point(new Point3d(outCx[i], outCy[i], outCz[i])));
            dists.Add(new GH_Number(Math.Sqrt(Math.Max(0, outD2[i]))));
            idxGh.Add(new GH_Integer(outIdx[i]));
        }

        DA.SetDataList(0, pts);
        DA.SetDataList(1, dists);
        DA.SetDataList(2, idxGh);
    }

    protected override Bitmap Icon => null!;

    public override Guid ComponentGuid => new("b87c95c2-6456-4b67-88b1-ca84cd327050");
}
