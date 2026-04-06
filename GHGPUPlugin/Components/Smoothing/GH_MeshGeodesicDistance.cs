using System.Drawing;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using GHGPUPlugin.Algorithms;
using GHGPUPlugin.NativeInterop;
using GHGPUPlugin.Utilities;
using Rhino.Geometry;

namespace GHGPUPlugin.Components.Smoothing;

public class GH_MeshGeodesicDistance : GH_Component
{
    private bool _approximationNoted;

    public GH_MeshGeodesicDistance()
        : base(
            "Mesh Geodesic Distance GPU",
            "MeshGeoDistGPU",
            "Approximate geodesic distance (heat diffusion + divergence + Laplacian). Seeds are 3D points snapped to the nearest mesh corner on the closest face. Outputs M and Scalars S wire to Mesh Isolines GPU (M→M, S→S).",
            "GPUTools",
            "Mesh")
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddMeshParameter("Mesh", "M", "Triangle or quad mesh.", GH_ParamAccess.item);
        pManager.AddPointParameter("Seeds", "SP", "Points snapped to the nearest mesh vertex (on the closest face); those vertices get distance 0.", GH_ParamAccess.list);
        pManager.AddIntegerParameter("Iterations", "I", "Distance diffusion iterations.", GH_ParamAccess.item, 30);
        pManager.AddNumberParameter("Strength", "S", "Distance diffusion strength per iteration.", GH_ParamAccess.item, 0.5);
        pManager.AddBooleanParameter("Normalise", "N", "Remap output distances to 0…1.", GH_ParamAccess.item, true);
        pManager.AddBooleanParameter("UseGPU", "GPU", "Use Metal Laplacian when available.", GH_ParamAccess.item, true);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddMeshParameter("Mesh", "M", "Copy of input mesh (same vertex order/count as Scalars); connect to Mesh Isolines GPU → Mesh.", GH_ParamAccess.item);
        pManager.AddNumberParameter("Scalars", "S", "Per-vertex distance field; connect to Mesh Isolines GPU → Scalars.", GH_ParamAccess.list);
        pManager.AddMeshParameter("ColourMesh", "CM", "Same mesh with vertex colours from Scalars.", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        NativeLoader.EnsureLoaded();

        Mesh? mesh = null;
        if (!DA.GetData("Mesh", ref mesh) || mesh == null)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Mesh is required.");
            return;
        }

        if (!mesh.IsValid)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Mesh is not valid.");
            return;
        }

        var seedPoints = new List<Point3d>();
        DA.GetDataList("Seeds", seedPoints);

        int iterations = 30;
        DA.GetData("Iterations", ref iterations);

        double strength = 0.5;
        DA.GetData("Strength", ref strength);

        bool normalise = true;
        DA.GetData("Normalise", ref normalise);

        bool useGpu = true;
        DA.GetData("UseGPU", ref useGpu);

        if (useGpu && !MetalGuard.EnsureReady(this))
            return;

        if (iterations < 1)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Iterations must be at least 1.");
            return;
        }

        if (strength < 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Strength clamped to 0.");
            strength = 0;
        }

        if (seedPoints.Count == 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Provide at least one seed point.");
            return;
        }

        var seeds = new List<int>();
        int skipped = 0;
        foreach (Point3d p in seedPoints)
        {
            if (!p.IsValid)
            {
                skipped++;
                continue;
            }

            if (!TryNearestMeshVertex(mesh, p, out int mvi))
            {
                skipped++;
                continue;
            }

            seeds.Add(mvi);
        }

        if (skipped > 0)
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, $"{skipped} seed point(s) were invalid or off the mesh and were skipped.");

        if (seeds.Count == 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "No valid seed points on the mesh.");
            return;
        }

        if (!_approximationNoted)
        {
            AddRuntimeMessage(
                GH_RuntimeMessageLevel.Remark,
                "Approximate geodesic via heat diffusion. Results are smooth but not exact. Accuracy improves with more iterations.");
            _approximationNoted = true;
        }

        if (!ApproximateHeatGeodesic.TryCompute(mesh, seeds, iterations, strength, useGpu, out double[]? dist, out string? err)
            || dist == null)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, err ?? "Geodesic computation failed.");
            return;
        }

        int vc = mesh.Vertices.Count;
        if (dist.Length < vc)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Internal distance buffer too short.");
            return;
        }

        double[] outD = new double[vc];
        Array.Copy(dist, outD, vc);

        if (normalise)
        {
            double lo = double.MaxValue, hi = double.MinValue;
            for (int i = 0; i < vc; i++)
            {
                double v = outD[i];
                if (v < lo) lo = v;
                if (v > hi) hi = v;
            }

            if (hi > lo + 1e-30)
            {
                double inv = 1.0 / (hi - lo);
                for (int i = 0; i < vc; i++)
                    outD[i] = (outD[i] - lo) * inv;
            }
            else
            {
                for (int i = 0; i < vc; i++)
                    outD[i] = 0;
            }
        }

        Mesh meshForIsolines = mesh.DuplicateMesh();
        DA.SetData(0, meshForIsolines);

        var ghScalars = new List<GH_Number>(vc);
        for (int i = 0; i < vc; i++)
            ghScalars.Add(new GH_Number(outD[i]));
        DA.SetDataList(1, ghScalars);

        Mesh colourMesh = MeshColourHelper.ColourByScalar(meshForIsolines, outD, normaliseMinMax: true);
        DA.SetData(2, colourMesh);
    }

    /// <summary>Nearest mesh corner of the face hit by <see cref="Mesh.ClosestMeshPoint"/>.</summary>
    private static bool TryNearestMeshVertex(Mesh mesh, Point3d p, out int meshVertexIndex)
    {
        meshVertexIndex = -1;
        var mp = mesh.ClosestMeshPoint(p, double.MaxValue);
        if (mp.FaceIndex < 0 || mp.FaceIndex >= mesh.Faces.Count)
            return false;

        MeshFace f = mesh.Faces[mp.FaceIndex];
        int best = f.A;
        double bestD = p.DistanceToSquared(mesh.Vertices[f.A]);

        void Consider(int mv)
        {
            double d2 = p.DistanceToSquared(mesh.Vertices[mv]);
            if (d2 < bestD)
            {
                bestD = d2;
                best = mv;
            }
        }

        Consider(f.B);
        Consider(f.C);
        if (f.IsQuad)
            Consider(f.D);

        meshVertexIndex = best;
        return true;
    }

    protected override Bitmap Icon => null!;

    public override Guid ComponentGuid => new("92215169-c171-4c07-8df1-812dd3512703");
}
