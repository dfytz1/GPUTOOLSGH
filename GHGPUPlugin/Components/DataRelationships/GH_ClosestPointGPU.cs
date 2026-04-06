using Grasshopper.Kernel;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;
using GHGPUPlugin.NativeInterop;
using System.Drawing;

namespace GHGPUPlugin.Components.DataRelationships;

/// <summary>Closest point on mesh triangles; GPU brute-force per query with CPU fallback.</summary>
public class GH_ClosestPointGPU : GH_Component
{
    public GH_ClosestPointGPU()
        : base(
            "Closest Point Mesh GPU",
            "CptMeshGPU",
            "Closest point on mesh surface (triangle brute force on Metal, or Rhino mesh query on CPU).",
            "GPUTools",
            "Mesh")
    {
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddPointParameter("QueryPoints", "QueryPoints", "Points to project onto the mesh.", GH_ParamAccess.list);
        pManager.AddMeshParameter("TargetMesh", "TargetMesh", "Mesh to measure against (quads are triangulated only if needed).", GH_ParamAccess.item);
        pManager.AddBooleanParameter("UseGPU", "UseGPU", "Use Metal when available.", GH_ParamAccess.item, true);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddPointParameter("ClosestPoint", "ClosestPoint", "Closest point on the mesh.", GH_ParamAccess.list);
        pManager.AddNumberParameter("Distance", "Distance", "Euclidean distance from query to closest point.", GH_ParamAccess.list);
        pManager.AddIntegerParameter("TriangleIndex", "TriangleIndex", "Triangle face index in the working mesh.", GH_ParamAccess.list);
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

        Mesh? meshIn = null;
        if (!DA.GetData("TargetMesh", ref meshIn) || meshIn == null)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "No mesh provided.");
            return;
        }

        bool useGpu = true;
        DA.GetData("UseGPU", ref useGpu);

        if (!TryGetTriangleMeshForClosest(meshIn, out Mesh work))
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Mesh has non-triangle faces that could not be triangulated.");
            return;
        }

        if (!work.IsValid || work.Vertices.Count == 0 || work.Faces.Count == 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Mesh is not usable.");
            return;
        }

        int vCount = work.Vertices.Count;
        int triCount = work.Faces.Count;
        var vx = new float[vCount];
        var vy = new float[vCount];
        var vz = new float[vCount];
        for (int i = 0; i < vCount; i++)
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
            if (!f.IsTriangle)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Mesh still contains non-triangle faces.");
                return;
            }

            triIdx[3 * fi] = f.A;
            triIdx[3 * fi + 1] = f.B;
            triIdx[3 * fi + 2] = f.C;
        }

        int qn = queries.Count;
        var qx = new float[qn];
        var qy = new float[qn];
        var qz = new float[qn];
        for (int i = 0; i < qn; i++)
        {
            qx[i] = (float)queries[i].X;
            qy[i] = (float)queries[i].Y;
            qz[i] = (float)queries[i].Z;
        }

        var outCx = new float[qn];
        var outCy = new float[qn];
        var outCz = new float[qn];
        var outD2 = new float[qn];
        var outTi = new int[qn];

        bool ranGpu = false;
        if (useGpu)
        {
            if (!MetalGuard.EnsureReady(this))
                return;

            MetalSharedContext.TryGetContext(out IntPtr ctx);
            int code = MetalBridge.ClosestPointsMesh(
                ctx,
                qx,
                qy,
                qz,
                qn,
                vx,
                vy,
                vz,
                vCount,
                triIdx,
                triCount,
                outCx,
                outCy,
                outCz,
                outD2,
                outTi);
            if (code != 0)
            {
                AddRuntimeMessage(
                    GH_RuntimeMessageLevel.Error,
                    $"Metal closest-point failed with code {code}.");
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
                    "GPU closest point did not run — using Rhino mesh closest point.");
            }

            for (int i = 0; i < qn; i++)
            {
                Point3d q = queries[i];
                int fi = work.ClosestPoint(q, out Point3d onMesh, double.MaxValue);
                outCx[i] = (float)onMesh.X;
                outCy[i] = (float)onMesh.Y;
                outCz[i] = (float)onMesh.Z;
                outD2[i] = (float)q.DistanceToSquared(onMesh);
                outTi[i] = fi;
            }
        }

        var pts = new List<GH_Point>(qn);
        var dists = new List<GH_Number>(qn);
        var tris = new List<GH_Integer>(qn);
        for (int i = 0; i < qn; i++)
        {
            pts.Add(new GH_Point(new Point3d(outCx[i], outCy[i], outCz[i])));
            dists.Add(new GH_Number(Math.Sqrt(Math.Max(0, outD2[i]))));
            tris.Add(new GH_Integer(outTi[i]));
        }

        DA.SetDataList(0, pts);
        DA.SetDataList(1, dists);
        DA.SetDataList(2, tris);
    }

    /// <summary>Use input mesh directly when already all triangles; otherwise duplicate once and quad-split.</summary>
    internal static bool TryGetTriangleMeshForClosest(Mesh input, out Mesh triangleMesh)
    {
        triangleMesh = input;
        int fc = input.Faces.Count;
        for (int i = 0; i < fc; i++)
        {
            if (!input.Faces[i].IsTriangle)
            {
                Mesh dup = input.DuplicateMesh();
                dup.Faces.ConvertQuadsToTriangles();
                triangleMesh = dup;
                for (int j = 0; j < dup.Faces.Count; j++)
                {
                    if (!dup.Faces[j].IsTriangle)
                        return false;
                }

                return true;
            }
        }

        return true;
    }

    protected override Bitmap Icon => null!;

    public override Guid ComponentGuid => new("aed0a06f-d4c9-4875-98fa-39a34d9d94e0");
}
